#!/usr/bin/env python3
"""
简单的 Flash Attention Varlen 使用示例

展示如何在实际场景中使用 flash_attn_varlen_func
"""

import torch
from nanovllm.flash_attn.flash_attn_interface import flash_attn_varlen_func

def simple_example():
    """简单示例：处理3个不同长度的文本序列"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 场景：批处理3个不同长度的文本序列
    # 序列1: "Hello world" (2个token)
    # 序列2: "How are you today" (4个token) 
    # 序列3: "Good morning" (2个token)
    
    seq_lengths = [2, 4, 2]  # 每个序列的长度
    total_tokens = sum(seq_lengths)  # 总token数: 8
    
    # 模型参数
    num_heads = 8
    head_dim = 64
    
    print(f"序列长度: {seq_lengths}")
    print(f"总token数: {total_tokens}")
    
    # 创建累积序列长度
    # cu_seqlens = [0, 2, 6, 8] 表示:
    # - 序列0: tokens 0-1
    # - 序列1: tokens 2-5  
    # - 序列2: tokens 6-7
    cu_seqlens = torch.tensor([0, 2, 6, 8], dtype=torch.int32, device=device)
    
    # 创建Q, K, V张量 (模拟经过线性变换后的结果)
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    
    print(f"Q形状: {q.shape}")
    print(f"K形状: {k.shape}")
    print(f"V形状: {v.shape}")
    
    # 调用flash_attn_varlen_func
    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=4,  # 最大序列长度
        max_seqlen_k=4,
        dropout_p=0.0,   # 推理时不使用dropout
        causal=True,     # 使用因果注意力（适合语言模型）
    )
    
    print(f"输出形状: {output.shape}")
    print("✓ 注意力计算完成")
    
    return output

def mqa_example():
    """MQA示例：查询头数 > 键值头数"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 场景：使用Multi-Query Attention
    # Q有8个头，K/V只有2个头（更节省内存）
    
    seq_lengths = [3, 5]
    total_tokens = sum(seq_lengths)
    
    num_heads_q = 8   # 查询头数
    num_heads_kv = 2  # 键值头数
    head_dim = 64
    
    print(f"\nMQA示例:")
    print(f"Q头数: {num_heads_q}, K/V头数: {num_heads_kv}")
    
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    
    # 创建不同头数的Q, K, V
    q = torch.randn(total_tokens, num_heads_q, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, num_heads_kv, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, num_heads_kv, head_dim, device=device, dtype=torch.float16)
    
    output = flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=5,
        max_seqlen_k=5,
        causal=True,
    )
    
    print(f"输出形状: {output.shape}")
    print("✓ MQA注意力计算完成")
    
    return output

def sliding_window_example():
    """滑动窗口示例：限制注意力范围"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 场景：使用滑动窗口注意力，每个位置只能看到附近的token
    # 适合长序列，减少计算复杂度
    
    seq_lengths = [8, 6]
    total_tokens = sum(seq_lengths)
    
    num_heads = 4
    head_dim = 32
    
    print(f"\n滑动窗口示例:")
    print(f"窗口大小: 左侧2个token, 右侧2个token")
    
    cu_seqlens = torch.tensor([0, 8, 14], dtype=torch.int32, device=device)
    
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    
    output = flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_k=8,
        causal=True,
        window_size=(2, 2),  # 滑动窗口: 左侧2, 右侧2
    )
    
    print(f"输出形状: {output.shape}")
    print("✓ 滑动窗口注意力计算完成")
    
    return output

if __name__ == "__main__":
    print("Flash Attention Varlen 简单示例")
    print("=" * 40)
    
    try:
        simple_example()
        mqa_example()
        sliding_window_example()
        
        print("\n" + "=" * 40)
        print("✓ 所有示例运行成功!")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

