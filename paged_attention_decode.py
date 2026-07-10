import torch
import triton
import triton.language as tl
import math

# =====================================================================
# 1. Triton-Ascend Paged Attention Kernel (GQA/MQA/MHA)
# =====================================================================
# 支持 Q head 数 ≠ KV head 数：通过 KV_GROUP_NUM = num_q_heads / num_kv_heads 映射

@triton.jit
def paged_attention_decode_kernel(
    Q_ptr,             # [num_seqs, num_q_heads, head_dim]
    K_cache_ptr,       # [num_blocks, num_kv_heads, block_size, head_dim]
    V_cache_ptr,       # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    Out_ptr,           # [num_seqs, num_q_heads, head_dim]
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_bt_s, stride_bt_b,
    stride_o_s, stride_o_h, stride_o_d,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr   # = num_q_heads / num_kv_heads
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)    # Q head 索引 (0..num_q_heads-1)

    # GQA 映射：连续 KV_GROUP_NUM 个 Q head 共享同一个 KV head
    kv_head_idx = head_idx // KV_GROUP_NUM

    cur_seq_len = tl.load(context_lens_ptr + seq_idx)

    offs_d = tl.arange(0, HEAD_DIM)
    # Q 按 Q head 索引寻址
    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + offs_d * stride_q_d
    q = tl.load(Q_ptr + q_offs)

    # Initialize loop-carried variables as 1D tensors to preserve type invariance
    m_i = tl.full([1], value=-float("inf"), dtype=tl.float32)
    l_i = tl.full([1], value=0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, cur_seq_len, BLOCK_N):
        logical_block_idx = start_n // BLOCK_SIZE

        bt_offset = seq_idx * stride_bt_s + logical_block_idx * stride_bt_b
        physical_block_id = tl.load(block_tables_ptr + bt_offset)

        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < cur_seq_len

        safe_intra_block_offs = (offs_n % BLOCK_SIZE)
        safe_intra_block_offs = tl.where(mask, safe_intra_block_offs, 0)

        # Load K block [BLOCK_N, HEAD_DIM] — 按 KV head 索引寻址
        k_base_start = physical_block_id * stride_k_b + kv_head_idx * stride_k_h
        k_ptr_block = (K_cache_ptr + k_base_start +
                       safe_intra_block_offs[:, None] * stride_k_s +
                       offs_d[None, :] * stride_k_d)
        k_block = tl.load(k_ptr_block, mask=mask[:, None], other=0.0)

        # Compute Attention Scores -> Output shape is [BLOCK_N, 1]
        attn_scores = tl.sum(q[None, :] * k_block, axis=1)[:, None] * scale
        attn_scores = tl.where(mask[:, None], attn_scores, -float("inf"))

        # Local Online Softmax parameters -> Output shape is [1]
        m_ij = tl.max(attn_scores, axis=0)
        m_next = tl.maximum(m_i, m_ij)

        p = tl.exp(attn_scores - m_next)
        p_sum = tl.sum(tl.where(mask[:, None], p, 0.0), axis=0)

        l_alpha = tl.exp(m_i - m_next)
        l_next = l_i * l_alpha + p_sum

        # Load V block [BLOCK_N, HEAD_DIM] — 按 KV head 索引寻址
        v_base_start = physical_block_id * stride_v_b + kv_head_idx * stride_v_h
        v_ptr_block = (V_cache_ptr + v_base_start +
                       safe_intra_block_offs[:, None] * stride_v_s +
                       offs_d[None, :] * stride_v_d)
        v_block = tl.load(v_ptr_block, mask=mask[:, None], other=0.0)

        # Accumulate output weight distribution -> [HEAD_DIM]
        acc = acc * l_alpha + tl.sum(p * v_block, axis=0)

        m_i = m_next
        l_i = l_next

    # Final block normalization
    acc = acc / l_i
    # Output 按 Q head 索引写入
    out_offs = seq_idx * stride_o_s + head_idx * stride_o_h + offs_d * stride_o_d
    tl.store(Out_ptr + out_offs, acc.to(tl.float16))

# =====================================================================
# 2. Python Host Wrapper
# =====================================================================

def paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size=16, block_n=16):
    """
    Paged attention decode with GQA/MQA/MHA support.

    Args:
        q:        [num_seqs, num_q_heads, head_dim]
        k_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
        v_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]

    Supports:
        MHA: num_q_heads == num_kv_heads (group_size=1)
        GQA: num_q_heads > num_kv_heads, num_q_heads % num_kv_heads == 0
        MQA: num_kv_heads == 1
    """
    assert q.is_npu, "Triton-Ascend requires tensors to be on NPU"
    assert block_n <= block_size and block_size % block_n == 0, \
        f"block_n ({block_n}) must be <= block_size ({block_size}) and divide it evenly: " \
        f"a single BLOCK_N iteration cannot span multiple paged physical blocks"

    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]

    assert num_q_heads % num_kv_heads == 0, \
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    out = torch.empty_like(q)  # [num_seqs, num_q_heads, head_dim]
    scale = 1.0 / math.sqrt(head_dim)
    kv_group_num = num_q_heads // num_kv_heads

    # Grid 第二维 = num_q_heads：每个 Q head 独立一个 program
    grid = (num_seqs, num_q_heads)

    paged_attention_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        KV_GROUP_NUM=kv_group_num
    )
    return out

# =====================================================================
# 3. Reference Implementation (GQA-aware)
# =====================================================================

def torch_paged_attention_reference(q, k_cache, v_cache, block_tables, context_lens, block_size=16):
    """
    Reference paged attention with GQA support.

    q:        [num_seqs, num_q_heads, head_dim]
    k_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
    v_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
    """
    num_seqs = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])

    for i in range(num_seqs):
        seq_len = context_lens[i].item()
        k_seq, v_seq = [], []

        for logical_blk in range(math.ceil(seq_len / block_size)):
            physical_blk = block_tables[i, logical_blk].item()
            k_blk = k_cache[physical_blk, :, :block_size, :]
            v_blk = v_cache[physical_blk, :, :block_size, :]
            k_seq.append(k_blk)
            v_seq.append(v_blk)

        # k_seq/v_seq: [num_kv_heads, seq_len, head_dim]
        k_seq = torch.cat(k_seq, dim=1)[:, :seq_len, :]
        v_seq = torch.cat(v_seq, dim=1)[:, :seq_len, :]

        # 按 KV group 计算：每组 Q heads 共享同一个 KV head
        for g in range(num_kv_heads):
            k_head = k_seq[g, :, :]   # [seq_len, head_dim]
            v_head = v_seq[g, :, :]   # [seq_len, head_dim]

            for h_in_group in range(kv_group_num):
                q_head_idx = g * kv_group_num + h_in_group
                q_head = q[i, q_head_idx, :].unsqueeze(0)  # [1, head_dim]

                scores = torch.matmul(q_head, k_head.t()) * scale
                attn_weights = torch.softmax(scores.float(), dim=-1).to(torch.float16)
                context = torch.matmul(attn_weights, v_head)
                out[i, q_head_idx, :] = context.squeeze(0)

    return out

# =====================================================================
# 4. Test Runner
# =====================================================================

def run_test():
    import torch_npu
    device = torch.device("npu:0")

    num_seqs = 4
    head_dim = 128
    block_size = 16
    max_blocks_per_seq = 4
    context_lens_vals = [12, 35, 22, 54]

    # ------------------------------------------------------------------
    # 测试配置: (num_q_heads, num_kv_heads, 模式名)
    # ------------------------------------------------------------------
    test_configs = [
        (16, 16, "MHA (group_size=1)"),
        (16, 8, "GQA (group_size=2)"),
        (16, 4, "GQA (group_size=4)"),
        (16, 2, "MQA (group_size=8)"),
    ]

    all_pass = True
    for num_q_heads, num_kv_heads, mode_name in test_configs:
        print(f"\n{'='*60}")
        print(f"  {mode_name} | q_heads={num_q_heads}, kv_heads={num_kv_heads}")
        print(f"{'='*60}")

        max_num_blocks = num_seqs * max_blocks_per_seq

        torch.manual_seed(42)
        q = torch.randn(num_seqs, num_q_heads, head_dim, dtype=torch.float16, device=device)
        context_lens = torch.tensor(context_lens_vals, dtype=torch.int32, device=device)

        block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
        all_physical_blocks = torch.randperm(max_num_blocks, dtype=torch.int32, device=device)

        idx = 0
        for i in range(num_seqs):
            needed_blocks = math.ceil(context_lens[i].item() / block_size)
            for j in range(needed_blocks):
                block_tables[i, j] = all_physical_blocks[idx]
                idx += 1

        k_cache = torch.randn(max_num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)
        v_cache = torch.randn(max_num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)

        ref_out = torch_paged_attention_reference(q, k_cache, v_cache, block_tables, context_lens, block_size)
        tri_out = paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size, block_n=16)

        is_correct = torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
        max_diff = (tri_out - ref_out).abs().max().item()
        mean_diff = (tri_out - ref_out).abs().mean().item()

        status = '[SUCCESS] Pass!' if is_correct else '[FAILED] Error!'
        print(f"  {status} | max_diff={max_diff:.5f}, mean_diff={mean_diff:.5f}")
        all_pass &= is_correct

    print(f"\n{'='*60}")
    print(f"  ALL TESTS: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_test()
