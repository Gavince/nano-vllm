import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    key_ptr/value_ptr: 指向输入 K/V 张量数据的 GPU 内存指针
    key_stride/value_stride: 张量的步长信息，用于计算内存偏移
    k_cache_ptr/v_cache_ptr: 指向缓存张量数据的 GPU 内存指针
    slot_mapping_ptr: 指向位置映射数组的 GPU 内存指针
    D: 编译时常量，等于 num_heads * head_dim
    """

    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 获取key和value
    # 通过当前kv-cache缓存的位置获取相对应的key和value
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 存储kv-cache
    # 将起始偏移量和行内偏移量相加，就得到了一个完整的、包含 D 个元素的内存地址偏移量集合，精确指向了 Cache 中第 slot 行的每一个位置
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    """
    参数传递: 你将 key 这个 PyTorch Tensor 对象作为参数传递。
    “翻译”过程: Triton 的启动器（Launcher）和 PyTorch 的接口在后台进行了一次“翻译”。
    它从 key 这个复杂的 Python 对象中，仅仅提取出最核心、GPU 最需要的信息：
    
    key.data_ptr(): 实际数据存储的内存地址。
    key.stride(0): 步幅信息（一个整数）。
    D: 维度信息（一个整数）。
    下达指令: 启动器将这些被“翻译”过的、简单的信息（内存地址和几个整数）发送给 GPU。
    GPU 执行: GPU 上的 store_kvcache_kernel 内核接收到这些信息。此时，内核中的 key_ptr 变量拿到的就只是那个内存地址，而不是整个 Tensor 对象。然后，内核就可以根据这个地址和步幅信息，在 GPU 内存中飞快地进行读写操作了。
    """
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        """
        解码步骤 N:
            1. prepare_decode() → slot_mapping=[17, 50]
            2. Attention.forward() → 调用 store_kvcache()
            3. store_kvcache_kernel() → 并行写入缓存位置 17, 50
            4. flash_attn_with_kvcache() → 使用缓存计算注意力
            5. 返回输出 → 用于生成下一个 token

        解码步骤 N+1:
            1. prepare_decode() → slot_mapping=[18, 51] 
            2. Attention.forward() → 调用 store_kvcache()
            3. store_kvcache_kernel() → 并行写入缓存位置 18, 51
            4. flash_attn_with_kvcache() → 使用缓存计算注意力
            5. 返回输出 → 用于生成下一个 token
        """
        k_cache, v_cache = self.k_cache, self.v_cache
        # 初始化阶段已经完成kv-cache的初始化
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
