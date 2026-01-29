#!/usr/bin/env python3
"""
LLM接口封装
支持多种LLM提供商
"""

import os
import json
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """LLM基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    def generate_with_json(self, prompt: str, **kwargs) -> Dict:
        """生成JSON格式输出"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM接口"""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, temperature: float = 0.3):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 2048)
        )
        return response.choices[0].message.content
    
    def generate_with_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict:
        """生成JSON格式输出"""
        json_prompt = prompt + "\n\n请以JSON格式输出结果。"
        response = self.generate(json_prompt, system_prompt, **kwargs)
        
        # 尝试解析JSON
        try:
            # 提取JSON部分
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {"raw_response": response, "parse_error": True}


class ZhipuLLM(BaseLLM):
    """智谱AI LLM接口"""
    
    def __init__(self, model: str = "glm-4", api_key: Optional[str] = None,
                 temperature: float = 0.3):
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError("请安装zhipuai: pip install zhipuai")
        
        self.model = model
        self.temperature = temperature
        self.client = ZhipuAI(api_key=api_key or os.getenv("ZHIPUAI_API_KEY"))
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature)
        )
        return response.choices[0].message.content
    
    def generate_with_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict:
        """生成JSON格式输出"""
        json_prompt = prompt + "\n\n请以JSON格式输出结果。"
        response = self.generate(json_prompt, system_prompt, **kwargs)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {"raw_response": response, "parse_error": True}


class MockLLM(BaseLLM):
    """模拟LLM，用于测试"""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None, **kwargs):
        # Accept and ignore extra kwargs like 'model' for compatibility
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        # 根据关键词返回预设响应
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return f"[MockLLM Response #{self.call_count}]"
    
    def generate_with_json(self, prompt: str, **kwargs) -> Dict:
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(response)
        except:
            return {"response": response}


class QwenLLM(BaseLLM):
    """阿里百炼 Qwen LLM接口 (OpenAI兼容)"""
    
    def __init__(self, model: str = "qwen-plus", api_key: Optional[str] = None,
                 temperature: float = 0.3, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
        
        self.model = model
        self.temperature = temperature
        # 阿里百炼使用OpenAI兼容接口
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 2048)
        )
        return response.choices[0].message.content
    
    def generate_with_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict:
        """生成JSON格式输出"""
        json_prompt = prompt + "\n\n请以JSON格式输出结果。"
        response = self.generate(json_prompt, system_prompt, **kwargs)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {"raw_response": response, "parse_error": True}


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM接口 (通过 qnaigc.com 代理)"""
    
    def __init__(self, model: str = "deepseek/deepseek-v3.2-251201", 
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.qnaigc.com/v1",
                 temperature: float = 0.1, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url
        )
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 2048)
        )
        return response.choices[0].message.content
    
    def generate_with_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict:
        """生成JSON格式输出"""
        json_prompt = prompt + "\n\nPlease output in JSON format."
        response = self.generate(json_prompt, system_prompt, **kwargs)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {"raw_response": response, "parse_error": True}


def create_llm(provider: str = "openai", **kwargs) -> BaseLLM:
    """工厂函数：创建LLM实例"""
    providers = {
        "openai": OpenAILLM,
        "zhipu": ZhipuLLM,
        "qwen": QwenLLM,  # 阿里百炼Qwen
        "dashscope": QwenLLM,  # 别名
        "deepseek": DeepSeekLLM,  # DeepSeek via qnaigc
        "mock": MockLLM,
    }
    
    if provider not in providers:
        raise ValueError(f"未知的LLM提供商: {provider}. 支持: {list(providers.keys())}")
    
    return providers[provider](**kwargs)

