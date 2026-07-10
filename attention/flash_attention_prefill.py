import torch
import triton
import triton.language as tl
import math

# =====================================================================
# 1. Triton-Ascend Prefill TND Attention Kernel
# =====================================================================
# 支持 MHA / GQA / MQA：通过 kv_group_num = num_q_heads / num_kv_heads 映射 Q head → KV head

@triton.jit
def flash_attention_prefill_kernel(
    Q_ptr, K_ptr, V_ptr,     # Q: [total_tokens, num_q_heads, head_dim]
                               # K: [total_tokens, num_kv_heads, head_dim]
                               # V: [total_tokens, num_kv_heads, head_dim]
    cu_seqlens_ptr,          # [num_seqs + 1]
    seq_block_info_ptr,      # [total_blocks, 2]: (seq_idx, local_block_m_idx) per block
    Out_ptr,                 # [total_tokens, num_q_heads, head_dim]
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr   # = num_q_heads / num_kv_heads
):
    block_idx = tl.program_id(0)   # 全局 block 编号
    head_idx = tl.program_id(1)    # Q head 索引 (0..num_q_heads-1)

    # GQA 映射：连续 KV_GROUP_NUM 个 Q head 共享同一个 KV head
    kv_head_idx = head_idx // KV_GROUP_NUM

    # 从预计算的查找表获取当前 block 所属的序列和局部块号
    info_offs = block_idx * 2
    seq_idx = tl.load(seq_block_info_ptr + info_offs).to(tl.int32)
    local_block_m_idx = tl.load(seq_block_info_ptr + info_offs + 1).to(tl.int32)

    cur_seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    cur_seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len = cur_seq_end - cur_seq_start

    # 基于序列内局部位置计算 Q 行偏移
    local_m_offset = local_block_m_idx * BLOCK_M
    offs_m = local_m_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    offs_d = tl.arange(0, HEAD_DIM)

    global_q_tokens = cur_seq_start + offs_m
    # Q 按 Q head 索引寻址
    q_offs = global_q_tokens[:, None] * stride_q_t + head_idx * stride_q_h + offs_d[None, :] * stride_q_d
    q_block = tl.load(Q_ptr + q_offs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M, 1], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        global_kv_tokens = cur_seq_start + offs_n

        # K/V 按 KV head 索引寻址（GQA 核心）
        k_offs = global_kv_tokens[None, :] * stride_k_t + kv_head_idx * stride_k_h + offs_d[:, None] * stride_k_d
        k_block = tl.load(K_ptr + k_offs, mask=mask_n[None, :], other=0.0)

        scores = tl.dot(q_block.to(tl.float16), k_block.to(tl.float16), out_dtype=tl.float32) * scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], scores, -float("inf"))
        else:
            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)[:, None]
        m_next = tl.maximum(m_i, m_ij)

        p = tl.exp(scores - m_next)
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        p_sum = tl.sum(p, axis=1)[:, None]

        l_alpha = tl.exp(m_i - m_next)
        l_next = l_i * l_alpha + p_sum

        v_offs = global_kv_tokens[:, None] * stride_v_t + kv_head_idx * stride_v_h + offs_d[None, :] * stride_v_d
        v_block = tl.load(V_ptr + v_offs, mask=mask_n[:, None], other=0.0)

        acc = acc * l_alpha
        acc = tl.dot(p.to(tl.float16), v_block.to(tl.float16), acc=acc, out_dtype=tl.float32)

        m_i = m_next
        l_i = l_next

    acc = acc / (l_i + 1e-6)
    acc = tl.where(mask_m[:, None], acc, 0.0)

    # Output 按 Q head 索引写入
    out_offs = global_q_tokens[:, None] * stride_o_t + head_idx * stride_o_h + offs_d[None, :] * stride_o_d
    tl.store(Out_ptr + out_offs, acc.to(tl.float16), mask=mask_m[:, None])

# =====================================================================
# 2. Python Host Wrapper
# =====================================================================

def flash_attention_prefill(q, k, v, cu_seqlens, causal=True):
    """
    Prefill attention with GQA/MQA/MHA support (TND layout).

    Args:
        q: [total_tokens, num_q_heads, head_dim]
        k: [total_tokens, num_kv_heads, head_dim]
        v: [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: [num_seqs + 1]
        causal: whether to apply causal mask

    Supports:
        MHA: num_q_heads == num_kv_heads (group_size=1)
        GQA: num_q_heads > num_kv_heads, num_q_heads % num_kv_heads == 0
        MQA: num_kv_heads == 1
    """
    assert q.is_npu and k.is_npu and v.is_npu, "Tensors must be on NPU"

    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]

    assert num_q_heads % num_kv_heads == 0, \
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    assert k.shape == v.shape, "K and V must have the same shape"
    assert k.shape[0] == total_tokens and k.shape[2] == head_dim, \
        "K/V shape must be [total_tokens, num_kv_heads, head_dim]"

    out = torch.empty_like(q)  # [total_tokens, num_q_heads, head_dim]
    scale = 1.0 / math.sqrt(head_dim)
    kv_group_num = num_q_heads // num_kv_heads

    BLOCK_M = 64
    BLOCK_N = 64

    num_seqs = cu_seqlens.numel() - 1

    # 预计算每个 block 所属的 (seq_idx, local_block_m_idx) 查找表
    block_info = []
    for i in range(num_seqs):
        seq_len = (cu_seqlens[i+1] - cu_seqlens[i]).item()
        num_blocks = math.ceil(seq_len / BLOCK_M) if seq_len > 0 else 0
        for b in range(num_blocks):
            block_info.append([i, b])

    total_blocks_m = len(block_info)
    seq_block_info = torch.tensor(block_info, dtype=torch.int32, device=q.device)

    # Grid 第二维 = num_q_heads：每个 Q head 独立一个 program
    grid = (total_blocks_m, num_q_heads)

    flash_attention_prefill_kernel[grid](
        q, k, v, cu_seqlens, seq_block_info, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        KV_GROUP_NUM=kv_group_num
    )
    return out

# =====================================================================
# 3. Reference Implementation (GQA-aware)
# =====================================================================

def torch_prefill_tnd_reference(q, k, v, cu_seqlens, causal=True):
    """
    Reference prefill attention with GQA support.

    q: [total_tokens, num_q_heads, head_dim]
    k: [total_tokens, num_kv_heads, head_dim]
    v: [total_tokens, num_kv_heads, head_dim]
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    out = torch.zeros_like(q)
    num_seqs = cu_seqlens.numel() - 1
    scale = 1.0 / math.sqrt(q.shape[-1])

    for i in range(num_seqs):
        start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
        if start == end:
            continue

        q_s = q[start:end, :, :]    # [seq_len, num_q_heads, D]
        k_s = k[start:end, :, :]    # [seq_len, num_kv_heads, D]
        v_s = v[start:end, :, :]    # [seq_len, num_kv_heads, D]
        seq_len = end - start

        # 转为 [heads, seq_len, D]
        q_s = q_s.transpose(0, 1).contiguous()   # [num_q_heads, seq_len, D]
        k_s = k_s.transpose(0, 1).contiguous()   # [num_kv_heads, seq_len, D]
        v_s = v_s.transpose(0, 1).contiguous()   # [num_kv_heads, seq_len, D]

        # 按 KV group 计算：每组 Q heads 共享同一个 KV head
        for g in range(num_kv_heads):
            q_group = q_s[g * kv_group_num : (g + 1) * kv_group_num]  # [group_size, seq_len, D]
            k_group = k_s[g:g+1]   # [1, seq_len, D]
            v_group = v_s[g:g+1]   # [1, seq_len, D]

            # [group_size, seq_len, D] @ [1, D, seq_len] → [group_size, seq_len, seq_len]
            scores = torch.matmul(q_group, k_group.transpose(-2, -1)) * scale

            if causal:
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
                scores = scores + mask

            weights = torch.softmax(scores.float(), dim=-1).to(torch.float16)
            # [group_size, seq_len, seq_len] @ [1, seq_len, D] → [group_size, seq_len, D]
            context = torch.matmul(weights, v_group)

            # 写回对应 Q head 位置
            out[start:end, g * kv_group_num : (g + 1) * kv_group_num, :] = \
                context.transpose(0, 1).contiguous()

    return out

# =====================================================================
# 4. Test Runner
# =====================================================================

def run_prefill_test():
    import torch_npu
    device = torch.device("npu:0")

    prompt_lengths = [128, 35, 256]

    seq_list = [0]
    running_sum = 0
    for length in prompt_lengths:
        running_sum += length
        seq_list.append(running_sum)
    cu_seqlens = torch.tensor(seq_list, dtype=torch.int32, device=device)
    total_tokens = cu_seqlens[-1].item()

    head_dim = 128

    # ------------------------------------------------------------------
    # 测试配置: (num_q_heads, num_kv_heads, 模式名)
    # ------------------------------------------------------------------
    test_configs = [
        (8, 8, "MHA (group_size=1)"),
        (8, 4, "GQA (group_size=2)"),
        (8, 2, "GQA (group_size=4)"),
        (8, 1, "MQA (group_size=8)"),
    ]

    all_pass = True
    for num_q_heads, num_kv_heads, mode_name in test_configs:
        print(f"\n{'='*60}")
        print(f"  {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        torch.manual_seed(123)
        q = torch.randn(total_tokens, num_q_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)

        for causal_flag in [True, False]:
            ref_out = torch_prefill_tnd_reference(q, k, v, cu_seqlens, causal=causal_flag)
            tri_out = flash_attention_prefill(q, k, v, cu_seqlens, causal=causal_flag)

            is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
            max_diff = (tri_out - ref_out).abs().max().item()
            mean_diff = (tri_out - ref_out).abs().mean().item()

            status = '[SUCCESS] Pass!' if is_correct else '[FAILED] Error!'
            print(f"  CAUSAL={causal_flag:<5} | {status} | max_diff={max_diff:.5f}, mean_diff={mean_diff:.5f}")
            all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  ALL TESTS: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_prefill_test()