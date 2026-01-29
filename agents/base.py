#!/usr/bin/env python3
"""
Agent基类定义
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentOutput:
    """Agent输出结构"""
    content: Dict[str, Any]  # 主要输出内容
    confidence: float  # 置信度 0-1
    reasoning: str  # 推理过程
    sources: list = field(default_factory=list)  # 引用的病历片段
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    corrections: list = field(default_factory=list)  # 修正历史


class BaseAgent(ABC):
    """任务Agent基类"""
    
    def __init__(self, name: str, llm, description: str = ""):
        self.name = name
        self.llm = llm
        self.description = description
        self.output: Optional[AgentOutput] = None
        self.upstream_inputs: Dict[str, AgentOutput] = {}
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """系统提示词"""
        pass
    
    @abstractmethod
    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        """构建任务提示词"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        """解析LLM响应，返回(内容, 置信度, 推理过程)"""
        pass
    
    def process(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput] = None) -> AgentOutput:
        """执行任务处理"""
        upstream_outputs = upstream_outputs or {}
        self.upstream_inputs = upstream_outputs
        
        # 构建提示词
        prompt = self.build_task_prompt(patient_info, upstream_outputs)
        
        # 调用LLM
        response = self.llm.generate(prompt, system_prompt=self.system_prompt)
        
        # 解析响应
        content, confidence, reasoning = self.parse_response(response)
        
        # 构建输出
        self.output = AgentOutput(
            content=content,
            confidence=confidence,
            reasoning=reasoning
        )
        
        return self.output
    
    def verify(self, query: str, patient_info: Dict) -> Dict[str, Any]:
        """响应下游质疑，重新验证"""
        verify_prompt = f"""
下游任务对你的输出提出质疑：
{query}

原始病历信息：
{self._format_patient_info(patient_info)}

你之前的输出：
{self.output.content if self.output else "无"}

请重新检查原始病历，确认或修正你的判断。

输出格式：
1. 重新检查结果：[确认正确 / 需要修正]
2. 如果修正，新的输出是什么
3. 修正原因（如有）
4. 新的置信度 (0-1)
"""
        response = self.llm.generate(verify_prompt, system_prompt=self.system_prompt)
        
        # 解析验证结果
        corrected = "需要修正" in response or "修正" in response
        
        if corrected:
            new_content, new_conf, reasoning = self.parse_response(response)
            if self.output:
                self.output.corrections.append({
                    "query": query,
                    "old_content": self.output.content.copy(),
                    "new_content": new_content,
                    "reason": reasoning
                })
                self.output.content = new_content
                self.output.confidence = new_conf
            
            return {
                "corrected": True,
                "new_output": new_content,
                "new_confidence": new_conf,
                "reason": reasoning
            }
        
        return {
            "corrected": False,
            "reason": "确认原输出正确"
        }
    
    def _format_patient_info(self, patient_info: Dict) -> str:
        """格式化患者信息用于提示词"""
        formatted = []
        for key, value in patient_info.items():
            if value and str(value).strip() and str(value) != 'nan':
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted[:50])  # 限制长度
    
    def _format_upstream(self, upstream_outputs: Dict[str, AgentOutput]) -> str:
        """格式化上游输出"""
        if not upstream_outputs:
            return "无上游输入"
        
        formatted = []
        for name, output in upstream_outputs.items():
            formatted.append(f"### {name} 的输出 (置信度: {output.confidence:.2f})")
            for key, value in output.content.items():
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
