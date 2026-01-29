#!/usr/bin/env python3
"""
PCIChain 配置文件
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"  # openai, azure, zhipu, qwen
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2048
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ChainConfig:
    """任务链配置"""
    max_corrections: int = 3  # 最大反馈修正次数
    confidence_threshold: float = 0.7  # 置信度阈值
    enable_llm_contradiction: bool = True  # 启用LLM辅助矛盾检测
    enable_rule_contradiction: bool = True  # 启用规则矛盾检测
    verbose: bool = True  # 详细输出


@dataclass
class PathConfig:
    """路径配置"""
    data_dir: str = "/Users/dsr/Desktop/paper/1213_PCI/1213_数据处理"
    dataset_file: str = "融合数据集_OCR增强版.csv"
    output_dir: str = "/Users/dsr/Desktop/paper/1213_PCI/pci_chain/experiments/results"
    
    @property
    def dataset_path(self):
        return os.path.join(self.data_dir, self.dataset_file)


# 默认配置
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_CHAIN_CONFIG = ChainConfig()
DEFAULT_PATH_CONFIG = PathConfig()
