import torch
import math
from sparse_attn_fwd import sparse_attn_prefill, sparse_attn_decode

# =====================================================================
# 1. HCA Compression (kernel=m_prime, stride=m_prime, no overlap)
# =====================================================================

def compress_kv_simple(k, v, cu_seqlens, m_prime):
    """HCA-style non-overlapping average pooling.

    kernel_size = stride = m_prime. Each compressed token is the mean of
    up to m_prime consecutive original tokens.

    Args:
        k, v: [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: [num_seqs + 1]
        m_prime: compression ratio

    Returns:
        compressed_k, compressed_v: [total_compressed, num_kv_heads, head_dim]
        comp_offsets: [num_seqs + 1] cumulative offsets
    """
    num_seqs = cu_seqlens.numel() - 1
    num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    device = k.device
    dtype = k.dtype

    comp_k_chunks = []
    comp_v_chunks = []
    comp_offsets = [0]

    for s in range(num_seqs):
        start = cu_seqlens[s].item()
        end = cu_seqlens[s + 1].item()
        seq_len = end - start
        if seq_len == 0:
            comp_offsets.append(comp_offsets[-1])
            continue

        k_s = k[start:end]
        v_s = v[start:end]

        n_compressed = max(1, math.ceil(seq_len / m_prime))

        k_pools = []
        v_pools = []
        for i in range(n_compressed):
            win_start = i * m_prime
            win_end = min(win_start + m_prime, seq_len)
            k_win = k_s[win_start:win_end]
            v_win = v_s[win_start:win_end]
            k_pools.append(k_win.mean(dim=0))
            v_pools.append(v_win.mean(dim=0))

        comp_k_chunks.append(torch.stack(k_pools))
        comp_v_chunks.append(torch.stack(v_pools))
        comp_offsets.append(comp_offsets[-1] + n_compressed)

    if comp_k_chunks:
        compressed_k = torch.cat(comp_k_chunks, dim=0)
        compressed_v = torch.cat(comp_v_chunks, dim=0)
    else:
        compressed_k = torch.zeros(0, num_kv_heads, head_dim, dtype=dtype, device=device)
        compressed_v = torch.zeros(0, num_kv_heads, head_dim, dtype=dtype, device=device)

    comp_offsets_t = torch.tensor(comp_offsets, dtype=torch.int32, device=device)
    return compressed_k, compressed_v, comp_offsets_t


# =====================================================================
# 2. HCA topk_idxs Builder (no indexer, all causal compressed positions)
# =====================================================================

def build_hca_topk_idxs(cu_seqlens, comp_offsets, total_tokens, n_win, m_prime):
    """Build topk_idxs for HCA.

    All causally-legal compressed positions are included (no indexer/top-k).
    Entries beyond the causal bound are padded with -1.

    topk_idxs layout: [n_win sliding-window positions] + [n_compressed positions]

    The number of compressed entries per token varies: a token at position p
    can attend to ceil(p / m_prime) compressed tokens.

    K_TOPK = n_win + max(n_compressed), where max is across all tokens.

    Args:
        cu_seqlens: [num_seqs + 1]
        comp_offsets: [num_seqs + 1]
        total_tokens: total original tokens
        n_win: sliding-window size
        m_prime: HCA compression ratio

    Returns:
        topk_idxs: [total_tokens, n_win + max_n_compressed] int32, -1 for padding
    """
    num_seqs = cu_seqlens.numel() - 1
    device = cu_seqlens.device

    # Find max compressed tokens any query can attend to
    max_n_comp = 0
    for s in range(num_seqs):
        seq_start = cu_seqlens[s].item()
        seq_end = cu_seqlens[s + 1].item()
        seq_len = seq_end - seq_start
        n_comp_s = comp_offsets[s + 1].item() - comp_offsets[s].item()
        # Token at last position can attend to all compressed tokens
        # compressed token i covers [i*m_prime, (i+1)*m_prime)
        # valid for query at position p if (i+1)*m_prime <= p (strict causal)
        # max i for p=seq_len-1: floor((seq_len-1)/m_prime)
        # Actually: valid if i*m_prime < p (at least one original token before p)
        max_for_seq = min(n_comp_s, math.ceil(seq_len / m_prime))
        max_n_comp = max(max_n_comp, max_for_seq)

    K_TOPK = n_win + max_n_comp
    topk_idxs = torch.full((total_tokens, K_TOPK), -1, dtype=torch.int32, device=device)
    full_kv_len = total_tokens

    for s in range(num_seqs):
        seq_start = cu_seqlens[s].item()
        seq_end = cu_seqlens[s + 1].item()
        seq_len = seq_end - seq_start
        if seq_len == 0:
            continue

        comp_start_s = comp_offsets[s].item()

        for local_pos in range(seq_len):
            t = seq_start + local_pos

            # Sliding window
            for wi in range(n_win):
                src = local_pos - n_win + 1 + wi
                if 0 <= src <= local_pos:
                    topk_idxs[t, wi] = seq_start + src

            # All causally-legal compressed positions
            # Compressed token i covers [i*m_prime, (i+1)*m_prime)
            # Valid if i*m_prime < local_pos (at least one pooled token is before query)
            n_causal = min((local_pos + m_prime - 1) // m_prime, max_n_comp)
            for ci in range(n_causal):
                topk_idxs[t, n_win + ci] = full_kv_len + comp_start_s + ci

    return topk_idxs


# =====================================================================
# 3. HCA Forward Wrappers
# =====================================================================

def hca_prefill_forward(q, k, v, cu_seqlens, n_win, m_prime, attn_sink=None):
    """HCA prefill forward (TND layout).

    1. Compress K/V with kernel=m_prime, stride=m_prime
    2. Build topk_idxs (sliding window + all causal compressed positions)
    3. Call sparse_attn_prefill

    Args:
        q: [total_tokens, num_q_heads, head_dim]
        k, v: [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: [num_seqs + 1]
        n_win: sliding-window size
        m_prime: HCA compression ratio
        attn_sink: optional [num_q_heads]
    """
    compressed_k, compressed_v, comp_offsets = compress_kv_simple(k, v, cu_seqlens, m_prime)

    total_tokens = q.shape[0]
    topk_idxs = build_hca_topk_idxs(cu_seqlens, comp_offsets, total_tokens, n_win, m_prime)

    k_full = torch.cat([k, compressed_k], dim=0)
    v_full = torch.cat([v, compressed_v], dim=0)

    return sparse_attn_prefill(q, k_full, v_full, topk_idxs, cu_seqlens, attn_sink=attn_sink)


def hca_decode_forward(q, k_cache, v_cache, block_tables, context_lens,
                       n_win, m_prime, block_size=16, attn_sink=None):
    """HCA decode forward (paged attention layout).

    1. Gather full KV from paged cache
    2. Compress with HCA simple pooling
    3. Build topk_idxs (sliding window + all causal compressed positions)
    4. Call sparse_attn_decode

    Args:
        q: [num_seqs, num_q_heads, head_dim]
        k_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]
        n_win: sliding-window size
        m_prime: HCA compression ratio
        block_size: page block size
        attn_sink: optional [num_q_heads]
    """
    num_seqs = q.shape[0]
    num_kv_heads = k_cache.shape[1]
    head_dim = q.shape[2]
    device = q.device

    k_seq_list = []
    v_seq_list = []
    cu_seqlens = [0]
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        parts_k = []
        parts_v = []
        for blk in range(math.ceil(ctx_len / block_size)):
            phys = block_tables[s, blk].item()
            valid = min(block_size, ctx_len - blk * block_size)
            parts_k.append(k_cache[phys, :, :valid, :])
            parts_v.append(v_cache[phys, :, :valid, :])
        k_s = torch.cat(parts_k, dim=1) if parts_k else torch.empty(
            num_kv_heads, 0, head_dim, dtype=q.dtype, device=device)
        v_s = torch.cat(parts_v, dim=1) if parts_v else torch.empty(
            num_kv_heads, 0, head_dim, dtype=q.dtype, device=device)
        k_seq_list.append(k_s)
        v_seq_list.append(v_s)
        cu_seqlens.append(cu_seqlens[-1] + ctx_len)

    total_kv = cu_seqlens[-1]
    if total_kv == 0:
        return torch.zeros_like(q)

    k_tnd = torch.cat(k_seq_list, dim=1).transpose(0, 1).contiguous()
    v_tnd = torch.cat(v_seq_list, dim=1).transpose(0, 1).contiguous()
    cu_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    compressed_k, compressed_v, comp_offsets = compress_kv_simple(k_tnd, v_tnd, cu_t, m_prime)

    # Build decode topk_idxs
    max_n_comp = 0
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        n_comp_s = comp_offsets[s + 1].item() - comp_offsets[s].item()
        max_for_seq = min(n_comp_s, math.ceil(ctx_len / m_prime))
        max_n_comp = max(max_n_comp, max_for_seq)

    K_TOPK = n_win + max_n_comp
    topk_idxs = torch.full((num_seqs, K_TOPK), -1, dtype=torch.int32, device=device)

    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        comp_start_s = comp_offsets[s].item()
        n_comp_s = comp_offsets[s + 1].item() - comp_start_s

        # Sliding window
        for wi in range(n_win):
            pos = ctx_len - n_win + wi
            if pos >= 0:
                topk_idxs[s, wi] = pos

        # All causal compressed positions
        n_causal = min((ctx_len + m_prime - 1) // m_prime, n_comp_s, max_n_comp)
        for ci in range(n_causal):
            topk_idxs[s, n_win + ci] = ctx_len + comp_start_s + ci

    return sparse_attn_decode(q, k_cache, v_cache, compressed_k, compressed_v,
                               block_tables, topk_idxs, context_lens,
                               num_win=n_win, block_size=block_size, attn_sink=attn_sink)


# =====================================================================
# 4. Reference Implementations
# =====================================================================

def torch_hca_prefill_reference(q, k, v, cu_seqlens, n_win, m_prime, attn_sink=None):
    """Reference HCA prefill with GQA support."""
    compressed_k, compressed_v, comp_offsets = compress_kv_simple(k, v, cu_seqlens, m_prime)
    total_tokens = q.shape[0]
    topk_idxs = build_hca_topk_idxs(cu_seqlens, comp_offsets, total_tokens, n_win, m_prime)

    k_full = torch.cat([k, compressed_k], dim=0)
    v_full = torch.cat([v, compressed_v], dim=0)

    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    num_seqs = cu_seqlens.numel() - 1

    for s in range(num_seqs):
        seq_start = cu_seqlens[s].item()
        seq_end = cu_seqlens[s + 1].item()
        if seq_start == seq_end:
            continue
        seq_len = seq_end - seq_start
        for local_pos in range(seq_len):
            t = seq_start + local_pos
            idxs_t = topk_idxs[t]
            valid = idxs_t >= 0
            idxs_valid = idxs_t[valid].long()

            k_sel = k_full[idxs_valid]
            v_sel = v_full[idxs_valid]

            for g in range(num_kv_heads):
                k_g = k_sel[:, g, :]
                v_g = v_sel[:, g, :]
                for h_in_group in range(kv_group_num):
                    h = g * kv_group_num + h_in_group
                    q_h = q[t, h, :]
                    scores = torch.matmul(k_g, q_h) * scale
                    denom = scores.float().exp().sum()
                    if attn_sink is not None:
                        denom = denom + math.exp(attn_sink[h].item())
                    weights = (scores.float().exp() / denom).to(torch.float16)
                    out[t, h, :] = torch.matmul(weights, v_g)
    return out


def torch_hca_decode_reference(q, k_cache, v_cache, block_tables, context_lens,
                                n_win, m_prime, block_size=16, attn_sink=None):
    """Reference HCA decode with GQA support."""
    num_seqs = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = q.shape[2]
    kv_group_num = num_q_heads // num_kv_heads
    device = q.device
    scale = 1.0 / math.sqrt(head_dim)

    k_seq_list = []
    v_seq_list = []
    cu_seqlens_tnd = [0]
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        parts_k = []
        parts_v = []
        for blk in range(math.ceil(ctx_len / block_size)):
            phys = block_tables[s, blk].item()
            valid = min(block_size, ctx_len - blk * block_size)
            parts_k.append(k_cache[phys, :, :valid, :])
            parts_v.append(v_cache[phys, :, :valid, :])
        k_s = torch.cat(parts_k, dim=1) if parts_k else torch.empty(
            num_kv_heads, 0, head_dim, dtype=q.dtype, device=device)
        v_s = torch.cat(parts_v, dim=1) if parts_v else torch.empty(
            num_kv_heads, 0, head_dim, dtype=q.dtype, device=device)
        k_seq_list.append(k_s)
        v_seq_list.append(v_s)
        cu_seqlens_tnd.append(cu_seqlens_tnd[-1] + ctx_len)

    total_kv = cu_seqlens_tnd[-1]
    k_tnd = torch.cat(k_seq_list, dim=1).transpose(0, 1).contiguous() if total_kv > 0 else torch.empty(
        0, num_kv_heads, head_dim, dtype=q.dtype, device=device)
    v_tnd = torch.cat(v_seq_list, dim=1).transpose(0, 1).contiguous() if total_kv > 0 else torch.empty(
        0, num_kv_heads, head_dim, dtype=q.dtype, device=device)
    cu_t = torch.tensor(cu_seqlens_tnd, dtype=torch.int32, device=device)

    compressed_k, compressed_v, comp_offsets = compress_kv_simple(k_tnd, v_tnd, cu_t, m_prime)

    # Build topk_idxs
    max_n_comp = 0
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        n_comp_s = comp_offsets[s + 1].item() - comp_offsets[s].item()
        max_for_seq = min(n_comp_s, math.ceil(ctx_len / m_prime))
        max_n_comp = max(max_n_comp, max_for_seq)

    K_TOPK = n_win + max_n_comp
    topk_idxs = torch.full((num_seqs, K_TOPK), -1, dtype=torch.int32, device=device)
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        comp_start_s = comp_offsets[s].item()
        n_comp_s = comp_offsets[s + 1].item() - comp_start_s
        for wi in range(n_win):
            pos = ctx_len - n_win + wi
            if pos >= 0:
                topk_idxs[s, wi] = pos
        n_causal = min((ctx_len + m_prime - 1) // m_prime, n_comp_s, max_n_comp)
        for ci in range(n_causal):
            topk_idxs[s, n_win + ci] = ctx_len + comp_start_s + ci

    out = torch.zeros_like(q)
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        for g in range(num_kv_heads):
            for h_in_group in range(kv_group_num):
                h = g * kv_group_num + h_in_group
                q_h = q[s, h, :]
                ks, vs = [], []
                for ki in range(K_TOPK):
                    raw = topk_idxs[s, ki].item()
                    if raw < 0:
                        continue
                    if raw < ctx_len:
                        blk = block_tables[s, raw // block_size].item()
                        off = raw % block_size
                        ks.append(k_cache[blk, g, off, :])
                        vs.append(v_cache[blk, g, off, :])
                    else:
                        cidx = raw - ctx_len
                        ks.append(compressed_k[cidx, g, :])
                        vs.append(compressed_v[cidx, g, :])
                if not ks:
                    continue
                k_stack = torch.stack(ks)
                v_stack = torch.stack(vs)
                scores = torch.matmul(k_stack, q_h) * scale
                denom = scores.float().exp().sum()
                if attn_sink is not None:
                    denom = denom + math.exp(attn_sink[h].item())
                weights = (scores.float().exp() / denom).to(torch.float16)
                out[s, h, :] = torch.matmul(weights, v_stack)
    return out


# =====================================================================
# 5. Test Runners
# =====================================================================

def run_hca_prefill_test():
    import torch_npu
    device = torch.device("npu:0")

    head_dim = 128
    m_prime = 128
    n_win = 16

    prompt_lengths = [256, 512, 384]
    seq_list = [0]
    running_sum = 0
    for length in prompt_lengths:
        running_sum += length
        seq_list.append(running_sum)
    cu_seqlens = torch.tensor(seq_list, dtype=torch.int32, device=device)
    total_tokens = cu_seqlens[-1].item()

    num_kv_heads_vals = [4, 2, 1]
    group_sizes = [2, 4, 8]
    mode_names = ["GQA (group=2)", "GQA (group=4)", "MQA (group=8)"]

    all_pass = True
    for num_kv_heads, group_size, mode_name in zip(
            num_kv_heads_vals, group_sizes, mode_names):
        num_q_heads = num_kv_heads * group_size
        print(f"\n{'='*60}")
        print(f"  HCA Prefill {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        q = torch.randn(total_tokens, num_q_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)

        for use_sink_val in [False, True]:
            sink = (torch.randn(num_q_heads, dtype=torch.float32, device=device)
                    if use_sink_val else None)

            ref_out = torch_hca_prefill_reference(
                q, k, v, cu_seqlens, n_win, m_prime, attn_sink=sink)
            tri_out = hca_prefill_forward(
                q, k, v, cu_seqlens, n_win, m_prime, attn_sink=sink)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS]' if is_correct else '[FAILED]'
            print(f"  sink={use_sink_val:<5} | {status} | max={max_diff:.5f} mean={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  HCA PREFILL ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")
    return all_pass


def run_hca_decode_test():
    import torch_npu
    device = torch.device("npu:0")

    num_seqs = 4
    head_dim = 128
    block_size = 16
    max_blocks_per_seq = 48
    context_lens_vals = [256, 512, 128, 384]
    m_prime = 128
    n_win = 16

    num_kv_heads_vals = [4, 2, 1]
    group_sizes = [2, 4, 8]
    mode_names = ["GQA (group=2)", "GQA (group=4)", "MQA (group=8)"]

    all_pass = True
    for num_kv_heads, group_size, mode_name in zip(
            num_kv_heads_vals, group_sizes, mode_names):
        num_q_heads = num_kv_heads * group_size
        print(f"\n{'='*60}")
        print(f"  HCA Decode {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        torch.manual_seed(123)
        q = torch.randn(num_seqs, num_q_heads, head_dim, dtype=torch.float16, device=device)
        context_lens = torch.tensor(context_lens_vals, dtype=torch.int32, device=device)

        max_num_blocks = num_seqs * max_blocks_per_seq
        all_blocks = torch.randperm(max_num_blocks, dtype=torch.int32, device=device)
        block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
        idx = 0
        for i in range(num_seqs):
            needed = math.ceil(context_lens[i].item() / block_size)
            for j in range(needed):
                block_tables[i, j] = all_blocks[idx]
                idx += 1

        k_cache = torch.randn(max_num_blocks, num_kv_heads, block_size, head_dim,
                              dtype=torch.float16, device=device)
        v_cache = torch.randn(max_num_blocks, num_kv_heads, block_size, head_dim,
                              dtype=torch.float16, device=device)

        for use_sink_val in [False, True]:
            sink = (torch.randn(num_q_heads, dtype=torch.float32, device=device)
                    if use_sink_val else None)

            ref_out = torch_hca_decode_reference(
                q, k_cache, v_cache, block_tables, context_lens,
                n_win, m_prime, block_size=block_size, attn_sink=sink)
            tri_out = hca_decode_forward(
                q, k_cache, v_cache, block_tables, context_lens,
                n_win, m_prime, block_size=block_size, attn_sink=sink)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS]' if is_correct else '[FAILED]'
            print(f"  sink={use_sink_val:<5} | {status} | max={max_diff:.5f} mean={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  HCA DECODE ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("  HCA PREFILL TEST")
    print("=" * 60)
    pf_ok = run_hca_prefill_test()
    print("\n" + "=" * 60)
    print("  HCA DECODE TEST")
    print("=" * 60)
    dc_ok = run_hca_decode_test()
    print(f"\n{'='*60}")
    print(f"  HCA ALL TESTS: {'PASS' if (pf_ok and dc_ok) else 'FAIL'}")
    print(f"{'='*60}")
