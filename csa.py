import torch
import math
from sparse_attn_fwd import sparse_attn_prefill, sparse_attn_decode

# =====================================================================
# 1. CSA Compression (kernel=2m, stride=m, 50% overlap)
# =====================================================================

def compress_kv_overlap(k, v, cu_seqlens, m):
    """CSA-style overlapping average pooling.

    kernel_size = 2*m, stride = m. Each compressed token is the mean of
    up to 2*m consecutive original tokens.

    Args:
        k, v: [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: [num_seqs + 1]
        m: compression stride

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

        n_compressed = max(1, math.ceil(seq_len / m))

        k_pools = []
        v_pools = []
        for i in range(n_compressed):
            win_start = i * m
            win_end = min(win_start + 2 * m, seq_len)
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
# 2. CSA topk_idxs Builder (Lightning Indexer)
# =====================================================================

def build_csa_topk_idxs(q, compressed_k, cu_seqlens, comp_offsets, n_win, top_k, m):
    """Build topk_idxs via Lightning Indexer (Q @ compressed_K^T).

    For each query token, compute dot-product similarity with all compressed
    tokens in the same sequence, then select top-k.

    topk_idxs layout: [n_win sliding-window positions] + [top_k compressed indices]

    Args:
        q: [total_tokens, num_q_heads, head_dim]
        compressed_k: [total_compressed, num_kv_heads, head_dim]
        cu_seqlens: [num_seqs + 1]
        comp_offsets: [num_seqs + 1]
        n_win: sliding-window size
        top_k: number of compressed positions to select
        m: CSA compression stride

    Returns:
        topk_idxs: [total_tokens, n_win + top_k] int32, -1 for padding
    """
    total_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = compressed_k.shape[1]
    head_dim = q.shape[2]
    kv_group_num = num_q_heads // num_kv_heads
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    K_TOPK = n_win + top_k
    topk_idxs = torch.full((total_tokens, K_TOPK), -1, dtype=torch.int32, device=device)

    # ---- Phase 1: sliding-window indices ----
    for s in range(num_seqs):
        seq_start = cu_seqlens[s].item()
        seq_end = cu_seqlens[s + 1].item()
        seq_len = seq_end - seq_start
        for local_pos in range(seq_len):
            t = seq_start + local_pos
            for wi in range(n_win):
                src = local_pos - n_win + 1 + wi
                if 0 <= src <= local_pos:
                    topk_idxs[t, wi] = seq_start + src

    # ---- Phase 2: Lightning Indexer top-k compressed selection ----
    full_kv_len = total_tokens  # compressed indices are offset by this

    for s in range(num_seqs):
        seq_start = cu_seqlens[s].item()
        seq_end = cu_seqlens[s + 1].item()
        seq_len = seq_end - seq_start
        comp_start_s = comp_offsets[s].item()
        comp_end_s = comp_offsets[s + 1].item()
        n_comp_s = comp_end_s - comp_start_s

        if seq_len == 0 or n_comp_s == 0:
            continue

        q_s = q[seq_start:seq_end]
        comp_k_s = compressed_k[comp_start_s:comp_end_s]

        for g in range(num_kv_heads):
            h_start = g * kv_group_num
            h_end = (g + 1) * kv_group_num

            q_group = q_s[:, h_start:h_end, :]    # [seq_len, group_size, D]
            k_g = comp_k_s[:, g, :]                # [n_comp_s, D]

            # scores: [seq_len, group_size, n_comp_s]
            scores = torch.matmul(q_group, k_g.T)

            # Causal mask: compressed token i covers [i*m, i*m+2m)
            # valid if window starts before query position
            for local_pos in range(seq_len):
                causal_valid = torch.arange(n_comp_s, device=device) * m < local_pos
                scores[local_pos, :, ~causal_valid] = float('-inf')

            k_sel = min(top_k, n_comp_s)
            _, top_indices = torch.topk(scores, k_sel, dim=-1)
            # top_indices: [seq_len, group_size, k_sel]

            for local_pos in range(seq_len):
                t = seq_start + local_pos
                for h_off in range(kv_group_num):
                    h = h_start + h_off
                    for ki in range(k_sel):
                        comp_idx = top_indices[local_pos, h_off, ki].item()
                        topk_idxs[t, n_win + ki] = full_kv_len + comp_start_s + comp_idx

    return topk_idxs


# =====================================================================
# 3. CSA Forward Wrappers
# =====================================================================

def csa_prefill_forward(q, k, v, cu_seqlens, n_win, top_k, m, attn_sink=None):
    """CSA prefill forward (TND layout).

    1. Compress K/V with kernel=2m, stride=m
    2. Build topk_idxs via Lightning Indexer
    3. Call sparse_attn_prefill on concatenated (K, compressed_K)

    Args:
        q: [total_tokens, num_q_heads, head_dim]
        k, v: [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: [num_seqs + 1]
        n_win: sliding-window size
        top_k: number of compressed positions per query
        m: CSA compression stride
        attn_sink: optional [num_q_heads]
    """
    compressed_k, compressed_v, comp_offsets = compress_kv_overlap(k, v, cu_seqlens, m)

    topk_idxs = build_csa_topk_idxs(q, compressed_k, cu_seqlens, comp_offsets, n_win, top_k, m)

    k_full = torch.cat([k, compressed_k], dim=0)
    v_full = torch.cat([v, compressed_v], dim=0)

    return sparse_attn_prefill(q, k_full, v_full, topk_idxs, cu_seqlens, attn_sink=attn_sink)


def csa_decode_forward(q, k_cache, v_cache, block_tables, context_lens,
                       n_win, top_k, m, block_size=16, attn_sink=None):
    """CSA decode forward (paged attention layout).

    1. Gather full KV from paged cache
    2. Compress with CSA overlap pooling
    3. Build topk_idxs via Lightning Indexer
    4. Call sparse_attn_decode

    Args:
        q: [num_seqs, num_q_heads, head_dim]
        k_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]
        n_win: sliding-window size
        top_k: number of compressed positions per query
        m: CSA compression stride
        block_size: page block size
        attn_sink: optional [num_q_heads]
    """
    num_seqs = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = q.shape[2]
    kv_group_num = num_q_heads // num_kv_heads
    device = q.device

    # Gather full KV from paged cache for each sequence
    k_seq_list = []
    v_seq_list = []
    cu_seqlens = [0]
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        k_s_parts = []
        v_s_parts = []
        num_blocks = math.ceil(ctx_len / block_size)
        for blk in range(num_blocks):
            phys_blk = block_tables[s, blk].item()
            valid_len = min(block_size, ctx_len - blk * block_size)
            k_s_parts.append(k_cache[phys_blk, :, :valid_len, :])
            v_s_parts.append(v_cache[phys_blk, :, :valid_len, :])
        k_s = torch.cat(k_s_parts, dim=1) if k_s_parts else torch.empty(
            num_kv_heads, 0, head_dim, dtype=q.dtype, device=device)
        v_s = torch.cat(v_s_parts, dim=1) if v_s_parts else torch.empty(
            num_kv_heads, 0, head_dim, dtype=q.dtype, device=device)
        k_seq_list.append(k_s)
        v_seq_list.append(v_s)
        cu_seqlens.append(cu_seqlens[-1] + ctx_len)

    # Concatenate into TND layout for compression
    if sum(context_lens).item() == 0:
        return torch.zeros_like(q)

    k_tnd = torch.cat(k_seq_list, dim=1).transpose(0, 1).contiguous()
    v_tnd = torch.cat(v_seq_list, dim=1).transpose(0, 1).contiguous()
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # Compress
    compressed_k, compressed_v, comp_offsets = compress_kv_overlap(k_tnd, v_tnd, cu_seqlens_t, m)

    # Build topk_idxs (decode: one query per sequence)
    K_TOPK = n_win + top_k
    topk_idxs = torch.full((num_seqs, K_TOPK), -1, dtype=torch.int32, device=device)
    total_kv_len = k_tnd.shape[0]

    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        comp_start_s = comp_offsets[s].item()
        comp_end_s = comp_offsets[s + 1].item()
        n_comp_s = comp_end_s - comp_start_s

        # Sliding window
        for wi in range(n_win):
            pos = ctx_len - n_win + wi
            if pos >= 0:
                topk_idxs[s, wi] = pos

        if n_comp_s == 0:
            continue

        # Indexer scores: one query token per sequence
        q_s = q[s:s+1]  # [1, num_q_heads, D]
        comp_k_s = compressed_k[comp_start_s:comp_end_s]  # [n_comp_s, num_kv_heads, D]

        for g in range(num_kv_heads):
            h_start = g * kv_group_num
            h_end = (g + 1) * kv_group_num

            q_group = q_s[:, h_start:h_end, :]  # [1, group_size, D]
            k_g = comp_k_s[:, g, :]              # [n_comp_s, D]

            scores = torch.matmul(q_group.squeeze(0), k_g.T)  # [group_size, n_comp_s]

            # Causal mask
            causal_valid = torch.arange(n_comp_s, device=device) * m < ctx_len
            scores[:, ~causal_valid] = float('-inf')

            k_sel = min(top_k, n_comp_s)
            _, top_indices = torch.topk(scores, k_sel, dim=-1)

            for h_off in range(kv_group_num):
                h = h_start + h_off
                for ki in range(k_sel):
                    comp_idx = top_indices[h_off, ki].item()
                    topk_idxs[s, n_win + ki] = ctx_len + comp_start_s + comp_idx

    return sparse_attn_decode(q, k_cache, v_cache, compressed_k, compressed_v,
                               block_tables, topk_idxs, context_lens,
                               num_win=n_win, block_size=block_size, attn_sink=attn_sink)


# =====================================================================
# 4. Reference Implementations
# =====================================================================

def torch_csa_prefill_reference(q, k, v, cu_seqlens, n_win, top_k, m, attn_sink=None):
    """Reference CSA prefill with GQA support."""
    compressed_k, compressed_v, comp_offsets = compress_kv_overlap(k, v, cu_seqlens, m)
    k_full = torch.cat([k, compressed_k], dim=0)
    v_full = torch.cat([v, compressed_v], dim=0)
    topk_idxs = build_csa_topk_idxs(q, compressed_k, cu_seqlens, comp_offsets, n_win, top_k, m)

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


def torch_csa_decode_reference(q, k_cache, v_cache, block_tables, context_lens,
                                n_win, top_k, m, block_size=16, attn_sink=None):
    """Reference CSA decode with GQA support."""
    num_seqs = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = q.shape[2]
    kv_group_num = num_q_heads // num_kv_heads
    device = q.device
    scale = 1.0 / math.sqrt(head_dim)

    K_TOPK = n_win + top_k
    topk_idxs = torch.full((num_seqs, K_TOPK), -1, dtype=torch.int32, device=device)

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

    compressed_k, compressed_v, comp_offsets = compress_kv_overlap(k_tnd, v_tnd, cu_t, m)

    # Build topk_idxs
    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        for wi in range(n_win):
            pos = ctx_len - n_win + wi
            if pos >= 0:
                topk_idxs[s, wi] = pos

    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        nc_s = comp_offsets[s+1].item() - comp_offsets[s].item()
        if nc_s == 0:
            continue
        comp_k_s = compressed_k[comp_offsets[s].item():comp_offsets[s+1].item()]
        n_comp_s = comp_k_s.shape[0]
        q_s = q[s:s+1]

        for g in range(num_kv_heads):
            h_start = g * kv_group_num
            h_end = (g + 1) * kv_group_num
            q_group = q_s[:, h_start:h_end, :].squeeze(0)
            k_g = comp_k_s[:, g, :]
            scores = torch.matmul(q_group, k_g.T)
            causal_valid = torch.arange(n_comp_s, device=device) * m < ctx_len
            scores[:, ~causal_valid] = float('-inf')
            k_sel = min(top_k, n_comp_s)
            _, top_idx = torch.topk(scores, k_sel, dim=-1)
            for h_off in range(kv_group_num):
                h = h_start + h_off
                for ki in range(k_sel):
                    topk_idxs[s, n_win + ki] = ctx_len + comp_offsets[s].item() + top_idx[h_off, ki].item()

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

def run_csa_prefill_test():
    import torch_npu
    device = torch.device("npu:0")

    head_dim = 128
    m = 4
    n_win = 16
    top_k_val = 4

    prompt_lengths = [64, 128, 96]
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
        print(f"  CSA Prefill {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        q = torch.randn(total_tokens, num_q_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)

        for use_sink_val in [False, True]:
            sink = (torch.randn(num_q_heads, dtype=torch.float32, device=device)
                    if use_sink_val else None)

            ref_out = torch_csa_prefill_reference(
                q, k, v, cu_seqlens, n_win, top_k_val, m, attn_sink=sink)
            tri_out = csa_prefill_forward(
                q, k, v, cu_seqlens, n_win, top_k_val, m, attn_sink=sink)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS]' if is_correct else '[FAILED]'
            print(f"  sink={use_sink_val:<5} | {status} | max={max_diff:.5f} mean={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  CSA PREFILL ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")
    return all_pass


def run_csa_decode_test():
    import torch_npu
    device = torch.device("npu:0")

    num_seqs = 4
    head_dim = 128
    block_size = 16
    max_blocks_per_seq = 8
    context_lens_vals = [32, 48, 16, 56]
    m = 4
    n_win = 8
    top_k_val = 4

    num_kv_heads_vals = [4, 2, 1]
    group_sizes = [2, 4, 8]
    mode_names = ["GQA (group=2)", "GQA (group=4)", "MQA (group=8)"]

    all_pass = True
    for num_kv_heads, group_size, mode_name in zip(
            num_kv_heads_vals, group_sizes, mode_names):
        num_q_heads = num_kv_heads * group_size
        print(f"\n{'='*60}")
        print(f"  CSA Decode {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
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

            ref_out = torch_csa_decode_reference(
                q, k_cache, v_cache, block_tables, context_lens,
                n_win, top_k_val, m, block_size=block_size, attn_sink=sink)
            tri_out = csa_decode_forward(
                q, k_cache, v_cache, block_tables, context_lens,
                n_win, top_k_val, m, block_size=block_size, attn_sink=sink)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS]' if is_correct else '[FAILED]'
            print(f"  sink={use_sink_val:<5} | {status} | max={max_diff:.5f} mean={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  CSA DECODE ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("  CSA PREFILL TEST")
    print("=" * 60)
    pf_ok = run_csa_prefill_test()
    print("\n" + "=" * 60)
    print("  CSA DECODE TEST")
    print("=" * 60)
    dc_ok = run_csa_decode_test()
    print(f"\n{'='*60}")
    print(f"  CSA ALL TESTS: {'PASS' if (pf_ok and dc_ok) else 'FAIL'}")
    print(f"{'='*60}")
