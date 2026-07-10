import torch
import triton
import triton.language as tl
import math

# =====================================================================
# 1. Prefill Sparse Attention Kernel (TND layout)
# =====================================================================

@triton.jit
def _sparse_prefill_kernel(
    Q_ptr, K_ptr, V_ptr,
    cu_seqlens_ptr,
    token_to_seq_ptr,
    topk_idxs_ptr,
    attn_sink_ptr,
    Out_ptr,
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_idx_t, stride_idx_k,
    stride_o_t, stride_o_h, stride_o_d,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TOPK: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    USE_SINK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // KV_GROUP_NUM

    seq_idx = tl.load(token_to_seq_ptr + token_idx)
    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    offs_d = tl.arange(0, HEAD_DIM)

    q_offs = token_idx * stride_q_t + head_idx * stride_q_h + offs_d * stride_q_d
    q = tl.load(Q_ptr + q_offs)

    m_i = tl.full([1], value=NEG_INF, dtype=tl.float32)
    l_i = tl.full([1], value=0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, K_TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < K_TOPK

        idx_offs = token_idx * stride_idx_t + offs_n * stride_idx_k
        k_idxs = tl.load(topk_idxs_ptr + idx_offs, mask=mask_n, other=-1)
        valid_k = (k_idxs >= 0) & mask_n
        safe_idxs = tl.where(valid_k, k_idxs, 0)

        k_offs = (safe_idxs[:, None] * stride_k_t +
                  kv_head_idx * stride_k_h +
                  offs_d[None, :] * stride_k_d)
        k_block = tl.load(K_ptr + k_offs, mask=valid_k[:, None], other=0.0)

        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(valid_k, scores, NEG_INF)

        m_ij = tl.max(scores, axis=0)
        m_next = tl.maximum(m_i, m_ij)

        p = tl.exp(scores[:, None] - m_next)
        p = tl.where(valid_k[:, None], p, 0.0)
        p_sum = tl.sum(p, axis=0)

        l_alpha = tl.exp(m_i - m_next)
        l_next = l_i * l_alpha + p_sum

        v_offs = (safe_idxs[:, None] * stride_v_t +
                  kv_head_idx * stride_v_h +
                  offs_d[None, :] * stride_v_d)
        v_block = tl.load(V_ptr + v_offs, mask=valid_k[:, None], other=0.0)

        acc = acc * l_alpha + tl.sum(p * v_block, axis=0)
        m_i = m_next
        l_i = l_next

    if USE_SINK:
        sink = tl.load(attn_sink_ptr + head_idx)
        m_next = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_next)
        sink_term = tl.exp(sink - m_next)
        acc = acc * alpha
        l_i = l_i * alpha + sink_term

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe

    out_offs = token_idx * stride_o_t + head_idx * stride_o_h + offs_d * stride_o_d
    tl.store(Out_ptr + out_offs, acc.to(tl.float16))


# =====================================================================
# 2. Decode Sparse Attention Kernel (paged attention layout)
# =====================================================================

@triton.jit
def _sparse_decode_kernel(
    Q_ptr,
    K_cache_ptr, V_cache_ptr,
    CompK_ptr, CompV_ptr,
    block_tables_ptr,
    topk_idxs_ptr,
    context_lens_ptr,
    attn_sink_ptr,
    Out_ptr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_ck_t, stride_ck_h, stride_ck_d,
    stride_cv_t, stride_cv_h, stride_cv_d,
    stride_bt_s, stride_bt_b,
    stride_idx_s, stride_idx_k,
    stride_o_s, stride_o_h, stride_o_d,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_WIN: tl.constexpr,
    K_TOPK: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    USE_SINK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // KV_GROUP_NUM

    cur_seq_len = tl.load(context_lens_ptr + seq_idx)
    n_compressed_base = cur_seq_len

    offs_d = tl.arange(0, HEAD_DIM)

    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + offs_d * stride_q_d
    q = tl.load(Q_ptr + q_offs)

    m_i = tl.full([1], value=NEG_INF, dtype=tl.float32)
    l_i = tl.full([1], value=0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # ---- Phase 1: Sliding window (paged KV cache) ----
    for start_n in range(0, NUM_WIN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < NUM_WIN

        idx_offs = seq_idx * stride_idx_s + offs_n * stride_idx_k
        raw_idxs = tl.load(topk_idxs_ptr + idx_offs, mask=mask_n, other=-1)
        valid_k = (raw_idxs >= 0) & (raw_idxs < cur_seq_len) & mask_n
        safe_idxs = tl.where(valid_k, raw_idxs, 0)

        logical_block_idx = safe_idxs // BLOCK_SIZE
        intra_block_offs = safe_idxs % BLOCK_SIZE

        bt_offset = seq_idx * stride_bt_s + logical_block_idx * stride_bt_b
        physical_block_id = tl.load(block_tables_ptr + bt_offset, mask=valid_k, other=0)
        safe_physical = tl.where(valid_k, physical_block_id, 0)

        k_base = (safe_physical[:, None] * stride_k_b +
                  kv_head_idx * stride_k_h +
                  intra_block_offs[:, None] * stride_k_s +
                  offs_d[None, :] * stride_k_d)
        k_block = tl.load(K_cache_ptr + k_base, mask=valid_k[:, None], other=0.0)

        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(valid_k, scores, NEG_INF)

        m_ij = tl.max(scores, axis=0)
        m_next = tl.maximum(m_i, m_ij)

        p = tl.exp(scores[:, None] - m_next)
        p = tl.where(valid_k[:, None], p, 0.0)
        p_sum = tl.sum(p, axis=0)

        l_alpha = tl.exp(m_i - m_next)
        l_next = l_i * l_alpha + p_sum

        v_base = (safe_physical[:, None] * stride_v_b +
                  kv_head_idx * stride_v_h +
                  intra_block_offs[:, None] * stride_v_s +
                  offs_d[None, :] * stride_v_d)
        v_block = tl.load(V_cache_ptr + v_base, mask=valid_k[:, None], other=0.0)

        acc = acc * l_alpha + tl.sum(p * v_block, axis=0)
        m_i = m_next
        l_i = l_next

    # ---- Phase 2: Compressed KV (contiguous) ----
    for start_n in range(NUM_WIN, K_TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < K_TOPK

        idx_offs = seq_idx * stride_idx_s + offs_n * stride_idx_k
        raw_idxs = tl.load(topk_idxs_ptr + idx_offs, mask=mask_n, other=-1)
        valid_k = (raw_idxs >= cur_seq_len) & mask_n
        safe_idxs = tl.where(valid_k, raw_idxs, 0)
        compressed_idx = safe_idxs - cur_seq_len

        ck_offs = (compressed_idx[:, None] * stride_ck_t +
                   kv_head_idx * stride_ck_h +
                   offs_d[None, :] * stride_ck_d)
        k_block = tl.load(CompK_ptr + ck_offs, mask=valid_k[:, None], other=0.0)

        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(valid_k, scores, NEG_INF)

        m_ij = tl.max(scores, axis=0)
        m_next = tl.maximum(m_i, m_ij)

        p = tl.exp(scores[:, None] - m_next)
        p = tl.where(valid_k[:, None], p, 0.0)
        p_sum = tl.sum(p, axis=0)

        l_alpha = tl.exp(m_i - m_next)
        l_next = l_i * l_alpha + p_sum

        cv_offs = (compressed_idx[:, None] * stride_cv_t +
                   kv_head_idx * stride_cv_h +
                   offs_d[None, :] * stride_cv_d)
        v_block = tl.load(CompV_ptr + cv_offs, mask=valid_k[:, None], other=0.0)

        acc = acc * l_alpha + tl.sum(p * v_block, axis=0)
        m_i = m_next
        l_i = l_next

    if USE_SINK:
        sink = tl.load(attn_sink_ptr + head_idx)
        m_next = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_next)
        sink_term = tl.exp(sink - m_next)
        acc = acc * alpha
        l_i = l_i * alpha + sink_term

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe

    out_offs = seq_idx * stride_o_s + head_idx * stride_o_h + offs_d * stride_o_d
    tl.store(Out_ptr + out_offs, acc.to(tl.float16))


# =====================================================================
# 3. Python Host Wrappers
# =====================================================================

def build_token_to_seq(cu_seqlens, total_tokens, device):
    num_seqs = cu_seqlens.numel() - 1
    token_to_seq = torch.zeros(total_tokens, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        if e > s:
            token_to_seq[s:e] = i
    return token_to_seq


def sparse_attn_prefill(q, k, v, topk_idxs, cu_seqlens, attn_sink=None):
    """Sparse attention prefill with GQA support (TND layout).

    Args:
        q:  [total_tokens, num_q_heads, head_dim]
        k:  [total_kv_tokens, num_kv_heads, head_dim]
        v:  [total_kv_tokens, num_kv_heads, head_dim]
        topk_idxs:  [total_tokens, K_TOPK] — indices into k/v rows, -1 = masked
        cu_seqlens: [num_seqs + 1]
        attn_sink: optional [num_q_heads] per-head sink logit
    """
    assert q.is_npu, "Q must be on NPU"
    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]

    assert num_q_heads % num_kv_heads == 0
    assert k.shape == v.shape
    assert topk_idxs.shape[0] == total_tokens

    kv_group_num = num_q_heads // num_kv_heads
    K_TOPK = topk_idxs.shape[1]
    use_sink = attn_sink is not None

    token_to_seq = build_token_to_seq(cu_seqlens, total_tokens, q.device)

    if use_sink:
        assert attn_sink.shape == (num_q_heads,)
        sink = attn_sink.contiguous()
    else:
        sink = torch.empty(1, device=q.device)

    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    grid = (total_tokens, num_q_heads)

    _sparse_prefill_kernel[grid](
        q, k, v,
        cu_seqlens, token_to_seq, topk_idxs, sink, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_idxs.stride(0), topk_idxs.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        HEAD_DIM=head_dim,
        BLOCK_N=64,
        K_TOPK=K_TOPK,
        KV_GROUP_NUM=kv_group_num,
        USE_SINK=use_sink,
        NEG_INF=-1.0e30,
    )
    return out


def sparse_attn_decode(q, k_cache, v_cache, compressed_k, compressed_v,
                       block_tables, topk_idxs, context_lens,
                       num_win, block_size=16, attn_sink=None):
    """Sparse attention decode with GQA support (paged attention layout).

    Args:
        q:  [num_seqs, num_q_heads, head_dim]
        k_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        compressed_k: [num_compressed, num_kv_heads, head_dim]
        compressed_v: [num_compressed, num_kv_heads, head_dim]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        topk_idxs: [num_seqs, K_TOPK] — first num_win entries < context_len
                   (sliding window positions), remaining entries >= context_len
                   (compressed KV indices + context_len offset)
        context_lens: [num_seqs]
        num_win: number of sliding window entries in topk_idxs
        block_size: page block size
        attn_sink: optional [num_q_heads] per-head sink logit
    """
    assert q.is_npu, "Q must be on NPU"
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]

    assert num_q_heads % num_kv_heads == 0
    assert k_cache.shape == v_cache.shape
    assert compressed_k.shape == compressed_v.shape

    kv_group_num = num_q_heads // num_kv_heads
    K_TOPK = topk_idxs.shape[1]
    use_sink = attn_sink is not None

    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    grid = (num_seqs, num_q_heads)

    if use_sink:
        assert attn_sink.shape == (num_q_heads,)
        sink = attn_sink.contiguous()
    else:
        sink = torch.empty(1, device=q.device)

    _sparse_decode_kernel[grid](
        q,
        k_cache, v_cache,
        compressed_k, compressed_v,
        block_tables, topk_idxs, context_lens, sink, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        compressed_k.stride(0), compressed_k.stride(1), compressed_k.stride(2),
        compressed_v.stride(0), compressed_v.stride(1), compressed_v.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        topk_idxs.stride(0), topk_idxs.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=16,
        NUM_WIN=num_win,
        K_TOPK=K_TOPK,
        KV_GROUP_NUM=kv_group_num,
        USE_SINK=use_sink,
        NEG_INF=-1.0e30,
    )
    return out


# =====================================================================
# 4. Reference Implementations
# =====================================================================

def torch_sparse_prefill_reference(q, k, v, topk_idxs, cu_seqlens, attn_sink=None):
    """Pure-PyTorch reference: sparse prefill attention with GQA (TND layout)."""
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    num_seqs = cu_seqlens.numel() - 1

    for s in range(num_seqs):
        seq_start, seq_end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
        if seq_start == seq_end:
            continue
        seq_len = seq_end - seq_start
        for local_pos in range(seq_len):
            t = seq_start + local_pos
            idxs_t = topk_idxs[t]  # [K_TOPK]
            valid = idxs_t >= 0
            idxs_valid = idxs_t[valid].long()

            # Gather KV: [num_valid, num_kv_heads, D] → focus per KV head
            k_sel = k[idxs_valid]   # [num_valid, num_kv_heads, D]
            v_sel = v[idxs_valid]   # [num_valid, num_kv_heads, D]

            for g in range(num_kv_heads):
                k_g = k_sel[:, g, :]  # [num_valid, D]
                v_g = v_sel[:, g, :]  # [num_valid, D]
                for h_in_group in range(kv_group_num):
                    h = g * kv_group_num + h_in_group
                    q_h = q[t, h, :]  # [D]
                    scores = torch.matmul(k_g, q_h) * scale  # [num_valid]
                    denom = scores.float().exp().sum()
                    if attn_sink is not None:
                        denom = denom + math.exp(attn_sink[h].item())
                    weights = (scores.float().exp() / denom).to(torch.float16)
                    ctx = torch.matmul(weights, v_g)  # [D]
                    out[t, h, :] = ctx
    return out


def torch_sparse_decode_reference(q, k_cache, v_cache, compressed_k, compressed_v,
                                  block_tables, topk_idxs, context_lens,
                                  block_size=16, attn_sink=None):
    """Pure-PyTorch reference: sparse decode attention with GQA (paged layout)."""
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    K_TOPK = topk_idxs.shape[1]
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        for g in range(num_kv_heads):
            for h_in_group in range(kv_group_num):
                h = g * kv_group_num + h_in_group
                q_h = q[s, h, :]
                ks, vs = [], []
                for ki in range(K_TOPK):
                    raw_idx = topk_idxs[s, ki].item()
                    if raw_idx < 0:
                        continue
                    if raw_idx < ctx_len:
                        blk = block_tables[s, raw_idx // block_size].item()
                        off = raw_idx % block_size
                        ks.append(k_cache[blk, g, off, :])
                        vs.append(v_cache[blk, g, off, :])
                    else:
                        cidx = raw_idx - ctx_len
                        ks.append(compressed_k[cidx, g, :])
                        vs.append(compressed_v[cidx, g, :])
                if not ks:
                    continue
                k_stack = torch.stack(ks)  # [N, D]
                v_stack = torch.stack(vs)  # [N, D]
                scores = torch.matmul(k_stack, q_h) * scale
                denom = scores.float().exp().sum()
                if attn_sink is not None:
                    denom = denom + math.exp(attn_sink[h].item())
                weights = (scores.float().exp() / denom).to(torch.float16)
                out[s, h, :] = torch.matmul(weights, v_stack)
    return out


# =====================================================================
# 5. Test Helpers
# =====================================================================

def _build_sliding_window_idxs(total_tokens, cu_seqlens, n_win, device):
    """Build sliding-window indices for TND layout.

    Returns [total_tokens, n_win] where each entry is an absolute position
    in [0, total_tokens) or -1.
    """
    num_seqs = cu_seqlens.numel() - 1
    idxs = torch.full((total_tokens, n_win), -1, dtype=torch.int32, device=device)
    for s in range(num_seqs):
        seq_start = cu_seqlens[s].item()
        seq_end = cu_seqlens[s + 1].item()
        seq_len = seq_end - seq_start
        if seq_len == 0:
            continue
        for pos in range(seq_len):
            t = seq_start + pos
            for win_off in range(n_win):
                src = pos - n_win + 1 + win_off
                if 0 <= src <= pos:
                    idxs[t, win_off] = seq_start + src
    return idxs


# =====================================================================
# 6. Prefill Test Runner
# =====================================================================

def run_prefill_test():
    import torch_npu
    device = torch.device("npu:0")

    head_dim = 128
    prompt_lengths = [64, 128, 96]
    seq_list = [0]
    running_sum = 0
    for length in prompt_lengths:
        running_sum += length
        seq_list.append(running_sum)
    cu_seqlens = torch.tensor(seq_list, dtype=torch.int32, device=device)
    total_tokens = cu_seqlens[-1].item()
    num_seqs = cu_seqlens.numel() - 1

    n_win = 32
    n_compressed = 8
    K_TOPK = n_win + n_compressed

    num_kv_heads_vals = [4, 2, 1]
    group_sizes = [2, 4, 8]
    mode_names = ["GQA (group=2)", "GQA (group=4)", "MQA (group=8)"]

    all_pass = True
    for num_kv_heads, group_size, mode_name in zip(
            num_kv_heads_vals, group_sizes, mode_names):
        num_q_heads = num_kv_heads * group_size
        print(f"\n{'='*60}")
        print(f"  {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        q = torch.randn(total_tokens, num_q_heads, head_dim,
                        dtype=torch.float16, device=device)
        k_full = torch.randn(total_tokens + num_seqs * n_compressed,
                             num_kv_heads, head_dim,
                             dtype=torch.float16, device=device)
        v_full = torch.randn(total_tokens + num_seqs * n_compressed,
                             num_kv_heads, head_dim,
                             dtype=torch.float16, device=device)

        win_idxs = _build_sliding_window_idxs(total_tokens, cu_seqlens,
                                               n_win, device)
        comp_start = total_tokens
        comp_idxs = torch.full((total_tokens, n_compressed), -1,
                               dtype=torch.int32, device=device)
        for s in range(num_seqs):
            seq_start = cu_seqlens[s].item()
            seq_end = cu_seqlens[s + 1].item()
            seq_len = seq_end - seq_start
            s_comp_offset = s * n_compressed
            for pos in range(seq_len):
                t = seq_start + pos
                for ci in range(n_compressed):
                    comp_idxs[t, ci] = comp_start + s_comp_offset + ci

        topk_idxs = torch.cat([win_idxs, comp_idxs], dim=1)

        for use_sink_val in [False, True]:
            sink = (torch.randn(num_q_heads, dtype=torch.float32, device=device)
                    if use_sink_val else None)

            ref_out = torch_sparse_prefill_reference(
                q, k_full, v_full, topk_idxs, cu_seqlens, attn_sink=sink)
            tri_out = sparse_attn_prefill(
                q, k_full, v_full, topk_idxs, cu_seqlens, attn_sink=sink)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS]' if is_correct else '[FAILED]'
            print(f"  sink={use_sink_val:<5} | {status} | max={max_diff:.5f} mean={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  PREFILL ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")


# =====================================================================
# 7. Decode Test Runner
# =====================================================================

def run_decode_test():
    import torch_npu
    device = torch.device("npu:0")

    num_seqs = 4
    head_dim = 128
    block_size = 16
    max_blocks_per_seq = 8
    context_lens_vals = [32, 48, 16, 56]

    n_win = 16
    n_compressed = 8
    K_TOPK = n_win + n_compressed

    num_kv_heads_vals = [4, 2, 1]
    group_sizes = [2, 4, 8]
    mode_names = ["GQA (group=2)", "GQA (group=4)", "MQA (group=8)"]

    all_pass = True
    for num_kv_heads, group_size, mode_name in zip(
            num_kv_heads_vals, group_sizes, mode_names):
        num_q_heads = num_kv_heads * group_size
        print(f"\n{'='*60}")
        print(f"  {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        torch.manual_seed(123)
        q = torch.randn(num_seqs, num_q_heads, head_dim,
                        dtype=torch.float16, device=device)
        context_lens = torch.tensor(context_lens_vals, dtype=torch.int32, device=device)

        max_num_blocks = num_seqs * max_blocks_per_seq
        all_blocks = torch.randperm(max_num_blocks, dtype=torch.int32, device=device)
        block_tables = torch.zeros(num_seqs, max_blocks_per_seq,
                                   dtype=torch.int32, device=device)
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

        comp_k = torch.randn(num_seqs * n_compressed, num_kv_heads, head_dim,
                             dtype=torch.float16, device=device)
        comp_v = torch.randn(num_seqs * n_compressed, num_kv_heads, head_dim,
                             dtype=torch.float16, device=device)

        topk_idxs = torch.full((num_seqs, K_TOPK), -1,
                               dtype=torch.int32, device=device)
        for s in range(num_seqs):
            ctx = context_lens[s].item()
            for wi in range(n_win):
                pos = ctx - n_win + wi
                if pos >= 0:
                    topk_idxs[s, wi] = pos
            s_comp_offset = s * n_compressed
            for ci in range(n_compressed):
                topk_idxs[s, n_win + ci] = ctx + s_comp_offset + ci

        for use_sink_val in [False, True]:
            sink = (torch.randn(num_q_heads, dtype=torch.float32, device=device)
                    if use_sink_val else None)

            ref_out = torch_sparse_decode_reference(
                q, k_cache, v_cache, comp_k, comp_v,
                block_tables, topk_idxs, context_lens,
                block_size=block_size, attn_sink=sink)
            tri_out = sparse_attn_decode(
                q, k_cache, v_cache, comp_k, comp_v,
                block_tables, topk_idxs, context_lens,
                num_win=n_win, block_size=block_size, attn_sink=sink)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS]' if is_correct else '[FAILED]'
            print(f"  sink={use_sink_val:<5} | {status} | max={max_diff:.5f} mean={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  DECODE ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SPARSE ATTENTION PREFILL TEST")
    print("=" * 60)
    run_prefill_test()
    print("\n" + "=" * 60)
    print("  SPARSE ATTENTION DECODE TEST")
    print("=" * 60)
    run_decode_test()
