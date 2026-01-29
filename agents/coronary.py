#!/usr/bin/env python3
"""
T2: Coronary Lesion Identification Agent
Identify coronary lesion information from medical records
"""

import re
from typing import Dict, Any, Tuple, List
from .base import BaseAgent, AgentOutput
from ..prompts import get_prompts


class CoronaryAgent(BaseAgent):
    """Coronary Lesion Identification Agent (T2)"""
    
    def __init__(self, llm, language: str = "zh"):
        super().__init__(
            name="coronary",
            llm=llm,
            description="Identify coronary lesion location, type and severity"
        )
        self.language = language
        self._prompts = get_prompts(language)
        
        # Standard vessel names
        self.vessel_aliases = {
            "LAD": ["LAD", "前降支", "左前降支", "Left Anterior Descending"],
            "LCX": ["LCX", "回旋支", "左回旋支", "Left Circumflex"],
            "RCA": ["RCA", "右冠", "右冠状动脉", "Right Coronary"],
            "LM": ["LM", "左主干", "左冠状动脉主干", "Left Main"],
            "D1": ["D1", "第一对角支", "对角支", "Diagonal"],
            "D2": ["D2", "第二对角支"],
            "OM": ["OM", "钝缘支", "Obtuse Marginal"],
        }
    
    @property
    def system_prompt(self) -> str:
        return self._prompts["coronary"]

    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        # 输入字段 (不包含金标准 Culprit_Vessel)
        relevant_fields = [
            # 手术记录 - 主要信息来源
            'PCI_Operative_Note',           # 手术记录（包含冠脉造影描述）
            'PCI_Procedure_Type',           # 手术类型
            # 支架信息
            'Stent_Specs',                  # 支架规格
            'Stent_Count',                  # 支架数量
            # 诊断信息
            'Admission_Diagnosis_Western',  # 入院诊断
            # 兼容旧字段名
            '手术及操作名称', 'PCI_术式', 'PCI_血管部位', 'PCI_支架信息',
            '入院诊断',
            'coronary_angiography', 'pci_procedure'
            # 注意: Culprit_Vessel 是金标准，不能作为输入
        ]
        
        patient_text = []
        for field in relevant_fields:
            if field in patient_info and patient_info[field]:
                value = str(patient_info[field])
                if value and value.strip() and value != 'nan':
                    patient_text.append(f"- {field}: {value}")
        
        if self.language == "en":
            return f"""Please identify coronary artery lesion information from the following medical record.

## Patient Medical Record
{chr(10).join(patient_text) if patient_text else 'Limited information'}

## Please output in the following format

### Affected Vessels
(List all affected vessels using standard abbreviations: LM/LAD/LCX/RCA/D1/D2/OM)

### Lesion Details
(Stenosis severity and treatment for each vessel)

### Conclusion
- Affected vessel list: [LAD, LCX, ...]
- Main lesion vessel: [which vessel has the most severe lesion]
- Lesion extent: [single-vessel/two-vessel/three-vessel/left main]
- Intervention: [balloon angioplasty/stent implantation/no intervention]
- Confidence: [0.0-1.0]
"""
        else:
            return f"""请从以下病历信息中识别冠状动脉病变情况。

## 患者病历信息
{chr(10).join(patient_text) if patient_text else '信息有限'}

## 请按以下格式输出

### 病变血管
（列出所有受累血管，使用标准缩写：LM/LAD/LCX/RCA/D1/D2/OM）

### 病变详情
（每个血管的狭窄程度和处理方式）

### 结论
- 受累血管列表：[LAD, LCX, ...]
- 主要病变血管：[哪个血管病变最严重]
- 病变范围：[单支/双支/三支/左主干]
- 干预方式：[球囊扩张/支架植入/未干预]
- 置信度：[0.0-1.0]
"""

    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        unknown = "Unknown" if self.language == "en" else "未知"
        content = {
            "vessels": [],
            "main_vessel": unknown,
            "vessel_count": unknown,
            "intervention": unknown,
            "details": {}
        }
        confidence = 0.5
        reasoning = ""
        
        # Extract reasoning
        for marker in ["### Affected Vessels", "### 病变血管"]:
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
        
        # Extract affected vessels
        vessel_pattern = r"(?:Affected vessel list|受累血管列表)[：:]?\s*\[([^\]]+)\]"
        match = re.search(vessel_pattern, response, re.IGNORECASE)
        if match:
            vessels_str = match.group(1)
            vessels = []
            for alias, names in self.vessel_aliases.items():
                if any(name in vessels_str for name in names) or alias in vessels_str:
                    vessels.append(alias)
            content["vessels"] = vessels
        else:
            vessels = []
            for alias, names in self.vessel_aliases.items():
                if any(name in response for name in names):
                    vessels.append(alias)
            content["vessels"] = list(set(vessels))
        
        # Extract lesion extent
        if "single-vessel" in response.lower() or "单支" in response:
            content["vessel_count"] = "single-vessel" if self.language == "en" else "单支病变"
        elif "two-vessel" in response.lower() or "双支" in response:
            content["vessel_count"] = "two-vessel" if self.language == "en" else "双支病变"
        elif "three-vessel" in response.lower() or "三支" in response:
            content["vessel_count"] = "three-vessel" if self.language == "en" else "三支病变"
        elif "left main" in response.lower() or "左主干" in response:
            content["vessel_count"] = "left main" if self.language == "en" else "左主干病变"
        
        # Extract intervention
        if "stent" in response.lower() or "支架" in response:
            content["intervention"] = "stent implantation" if self.language == "en" else "支架植入"
        elif "balloon" in response.lower() or "球囊" in response:
            content["intervention"] = "balloon angioplasty" if self.language == "en" else "球囊扩张"
        elif "no intervention" in response.lower() or "未干预" in response or "保守" in response:
            content["intervention"] = "no intervention" if self.language == "en" else "未干预"
        
        # Extract confidence
        conf_match = re.search(r"(?:Confidence|置信度)[：:]?\s*([\d.]+)", response, re.IGNORECASE)
        if conf_match:
            confidence = min(float(conf_match.group(1)), 1.0)
        
        return content, confidence, reasoning
