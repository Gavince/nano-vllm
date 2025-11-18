import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    # 模型路径
    model: str
    # 每个批次最大token数
    max_num_batched_tokens: int = 16384
    # 最大序列数
    max_num_seqs: int = 512
    # 模型最大长度
    max_model_len: int = 4096
    # GPU显存利用率
    gpu_memory_utilization: float = 0.9
    # 张量并行大小
    tensor_parallel_size: int = 1
    # 是否强制使用eager模式
    enforce_eager: bool = False
    # HuggingFace配置
    hf_config: AutoConfig | None = None
    # 结束符token id
    eos: int = -1
    # KV缓存块大小
    kvcache_block_size: int = 256
    # KV缓存块数量
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
