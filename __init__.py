#!/usr/bin/env python3
"""
PCIChain: 任务链式自修正Agent框架
用于PCI患者临床决策的LLM多Agent系统
"""

from .chain import PCIChain
from .agents.base import BaseAgent
from .utils.contradiction import ContradictionDetector
from .utils.feedback import FeedbackCorrector

__version__ = "0.1.0"
__all__ = ["PCIChain", "BaseAgent", "ContradictionDetector", "FeedbackCorrector"]
