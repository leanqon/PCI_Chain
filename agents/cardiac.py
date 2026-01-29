#!/usr/bin/env python3
"""
T1: Cardiac Function Inference Agent
Predict LVEF grade based on medical record information
"""

import re
from typing import Dict, Any, Tuple
from .base import BaseAgent, AgentOutput
from ..prompts import get_prompts


class CardiacFunctionAgent(BaseAgent):
    """Cardiac Function Inference Agent (T1)"""
    
    def __init__(self, llm, language: str = "zh"):
        super().__init__(
            name="cardiac_function",
            llm=llm,
            description="Infer cardiac function status from admission and PCI records"
        )
        self.language = language
        self._prompts = get_prompts(language)
    
    @property
    def system_prompt(self) -> str:
        return self._prompts["cardiac_function"]

    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        # 输入字段 (不包含金标准 Pre_LVEF - 应从Echo_Text推断)
        relevant_fields = [
            # 心脏超声报告文本 - 主要信息来源（包含LVEF描述）
            'Pre_PCI_Echo_Text',          # 超声报告文本
            'Pre_PCI_Echo_Date',          # 超声日期
            # 辅助信息（不直接给出LVEF数值）
            'Pre_LVEDD',                  # 左室舒张末期内径
            'Pre_LA',                     # 左房大小
            # 症状和病史
            'Chief_Complaint',            # 主诉
            'Admission_Diagnosis_Western', # 入院诊断
            # 实验室检查
            'Pre_PCI_NT_proBNP',          # NT-proBNP
            'Pre_PCI_Troponin_Max',       # 肌钙蛋白
            # 兼容旧字段名
            '主诉', '现病史', '既往史', '入院诊断',
            'echocardiography', 'chief_complaint', 'history_present_illness'
            # 注意: Pre_LVEF 是金标准，不能直接作为结构化输入
        ]
        
        patient_text = []
        for field in relevant_fields:
            if field in patient_info and patient_info[field]:
                value = str(patient_info[field])
                if value and value.strip() and value != 'nan':
                    patient_text.append(f"- {field}: {value}")
        
        # Get upstream coronary results
        coronary_info = ""
        if "coronary" in upstream_outputs:
            coronary = upstream_outputs["coronary"]
            if self.language == "en":
                coronary_info = f"""
Upstream Task (Coronary Lesion Identification) Results:
- Affected vessels: {coronary.content.get('vessels', 'Unknown')}
- Lesion severity: {coronary.content.get('severity', 'Unknown')}
- Confidence: {coronary.confidence:.2f}
"""
            else:
                coronary_info = f"""
上游任务（冠脉病变识别）结果：
- 病变血管：{coronary.content.get('vessels', '未知')}
- 病变严重程度：{coronary.content.get('severity', '未知')}
- 置信度：{coronary.confidence:.2f}
"""
        
        if self.language == "en":
            return f"""Please infer the patient's cardiac function status (LVEF grade) based on the following medical record information.

## Patient Medical Record
{chr(10).join(patient_text) if patient_text else 'Limited information'}

{coronary_info}

## Please output in the following format

### Reasoning Process
(Analyze clues related to cardiac function in the medical record)

### Conclusion
- LVEF grade: [Normal/Mildly reduced/Moderately reduced/Severely reduced]
- Estimated LVEF range: [X-Y%]
- Confidence: [0.0-1.0]
- Key evidence: [List key supporting evidence]
"""
        else:
            return f"""请根据以下病历信息，推断患者的心功能状态（LVEF分级）。

## 患者病历信息
{chr(10).join(patient_text) if patient_text else '信息有限'}

{coronary_info}

## 请按以下格式输出

### 推理过程
（分析病历中与心功能相关的线索）

### 结论
- LVEF分级：[正常/轻度降低/中度降低/重度降低]
- 预估LVEF范围：[X-Y%]
- 置信度：[0.0-1.0]
- 主要依据：[列出关键证据]
"""

    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        unknown = "Unknown" if self.language == "en" else "未知"
        content = {
            "lvef_grade": unknown,
            "lvef_range": unknown,
            "key_evidence": []
        }
        confidence = 0.5
        reasoning = ""
        
        # Extract reasoning
        for marker in ["### Reasoning Process", "### 推理过程"]:
            if marker in response:
                parts = response.split(marker)
                if len(parts) > 1:
                    for end_marker in ["### Conclusion", "### 结论"]:
                        if end_marker in parts[1]:
                            reasoning = parts[1].split(end_marker)[0].strip()
                            break
                    else:
                        reasoning = parts[1].strip()
                break
        
        # Extract LVEF grade - English patterns
        en_grade_patterns = [
            (r"LVEF grade[：:]?\s*(Normal|Mildly reduced|Moderately reduced|Severely reduced)", 1),
            (r"(Normal|Mildly reduced|Moderately reduced|Severely reduced).*LVEF", 1),
        ]
        # Chinese patterns
        zh_grade_patterns = [
            (r"LVEF分级[：:]?\s*(正常|轻度降低|中度降低|重度降低)", 1),
            (r"(正常|轻度降低|中度降低|重度降低).*LVEF", 1),
        ]
        
        patterns = en_grade_patterns if self.language == "en" else zh_grade_patterns
        for pattern, group in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                content["lvef_grade"] = match.group(group)
                break
        
        # Extract LVEF range
        range_match = re.search(r"(\d{1,2})[-%～~到至](\d{1,2})%?", response)
        if range_match:
            content["lvef_range"] = f"{range_match.group(1)}-{range_match.group(2)}%"
        
        # Extract confidence
        conf_match = re.search(r"(?:Confidence|置信度)[：:]?\s*([\d.]+)", response, re.IGNORECASE)
        if conf_match:
            confidence = min(float(conf_match.group(1)), 1.0)
        
        return content, confidence, reasoning
