#!/usr/bin/env python3
"""
T5: 风险分层Agent
预测PCI术后不良事件风险
"""

import re
from typing import Dict, Any, Tuple, List
from .base import BaseAgent, AgentOutput


class RiskAgent(BaseAgent):
    """风险分层Agent (T5)"""
    
    def __init__(self, llm):
        super().__init__(
            name="risk",
            llm=llm,
            description="Predict post-PCI adverse event risk / 预测PCI术后不良事件风险"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a cardiovascular risk assessment specialist with expertise in predicting outcomes for PCI patients.
你是一位心血管风险评估专家，专长于PCI患者预后预测。

Your task is to assess the patient's risk for:
你的任务是评估患者以下风险：
1. 30-day adverse events (30天不良事件)
2. 1-year MACE (Major Adverse Cardiovascular Events) (1年主要心血管不良事件)

Risk factors to consider / 需考虑的危险因素:
- Age / 年龄
- LVEF (Left Ventricular Ejection Fraction / 左室射血分数)
- Number of diseased vessels / 病变血管数量
- Diabetes / 糖尿病
- Renal function / 肾功能
- Prior MI or revascularization / 既往心梗或血运重建史
- Lesion complexity / 病变复杂程度

Risk stratification / 风险分层:
- Low risk / 低危: <10% event rate
- Moderate risk / 中危: 10-20% event rate  
- High risk / 高危: >20% event rate

Note: Input data may contain both Chinese and English. Process both languages appropriately.
注意：输入数据可能包含中英文，请适当处理两种语言。"""

    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        # 筛选风险相关字段
        relevant_fields = [
            '性别', 'Sex', 'Gender',
            '入院诊断', 'Admission_Diagnosis',
            '出院诊断', 'Discharge_Diagnosis',
            '既往史', 'Past_History', 'word_既往史',
            '心超_LVEF', 'LVEF',
            'PCI_术式', 'PCI_血管部位',
            'ocr_诊断'
        ]
        
        patient_text = []
        for field in relevant_fields:
            if field in patient_info and patient_info[field]:
                value = str(patient_info[field])
                if value and value.strip() and value.lower() != 'nan':
                    patient_text.append(f"【{field}】{value}")
        
        # 获取上游结果
        upstream_info = ""
        
        if "cardiac_function" in upstream_outputs:
            cardiac = upstream_outputs["cardiac_function"]
            upstream_info += f"""
## Cardiac Function / 心功能评估
- LVEF grade / LVEF分级: {cardiac.content.get('lvef_grade', 'Unknown/未知')}
- LVEF range / LVEF范围: {cardiac.content.get('lvef_range', 'Unknown/未知')}
"""
        
        if "coronary" in upstream_outputs:
            coronary = upstream_outputs["coronary"]
            upstream_info += f"""
## Coronary Lesions / 冠脉病变
- Affected vessels / 受累血管: {', '.join(coronary.content.get('vessels', ['Unknown/未知']))}
- Lesion extent / 病变范围: {coronary.content.get('vessel_count', 'Unknown/未知')}
"""
        
        if "diagnosis" in upstream_outputs:
            diag = upstream_outputs["diagnosis"]
            upstream_info += f"""
## Diagnosis / 诊断
- Main diagnosis / 主诊断: {diag.content.get('main_diagnosis', 'Unknown/未知')}
- Comorbidities / 合并症: {', '.join(diag.content.get('comorbidities', ['None/无']))}
"""
        
        if "medication" in upstream_outputs:
            med = upstream_outputs["medication"]
            upstream_info += f"""
## Medication / 用药
- Complete DAPT / 完整双抗: {'Yes/是' if med.content.get('antiplatelet') else 'No/否'}
- Statin / 他汀: {med.content.get('statin', 'Unknown/未知')}
"""
        
        return f"""Please assess the patient's cardiovascular risk based on the following information.
请根据以下信息评估患者的心血管风险。

## Patient Information / 患者信息
{chr(10).join(patient_text) if patient_text else 'Limited information / 信息有限'}

{upstream_info}

## Output Format / 输出格式

### Risk Factor Analysis / 危险因素分析
(Identify and analyze each risk factor)
(识别并分析每个危险因素)

### Risk Assessment / 风险评估

#### 30-day Risk / 30天风险
- Risk level / 风险等级: [Low/Moderate/High] / [低危/中危/高危]
- Estimated event rate / 预估事件率: [X%]
- Main risk factors / 主要危险因素: [...]

#### 1-year MACE Risk / 1年MACE风险
- Risk level / 风险等级: [Low/Moderate/High] / [低危/中危/高危]
- Estimated event rate / 预估事件率: [X%]
- Main risk factors / 主要危险因素: [...]

### Recommendations / 建议
(Risk mitigation strategies)
(风险缓解策略)

### Confidence / 置信度: [0.0-1.0]
"""

    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        content = {
            "30day_risk": "Unknown/未知",
            "30day_rate": "",
            "1year_risk": "Unknown/未知",
            "1year_rate": "",
            "risk_factors": [],
            "recommendations": []
        }
        confidence = 0.5
        reasoning = ""
        
        # 提取分析过程
        if "### Risk Factor Analysis" in response or "### 危险因素分析" in response:
            split_key = "### Risk Factor Analysis" if "### Risk Factor Analysis" in response else "### 危险因素分析"
            parts = response.split(split_key)
            if len(parts) > 1:
                end_key = "### Risk Assessment" if "### Risk Assessment" in parts[1] else "### 风险评估"
                reasoning_part = parts[1].split(end_key)[0] if end_key in parts[1] else parts[1]
                reasoning = reasoning_part.strip()
        
        # 提取30天风险
        risk_patterns_30d = [
            r"30-day Risk.*?Risk level.*?[：:]\s*(Low|Moderate|High|低危|中危|高危)",
            r"30天风险.*?风险等级.*?[：:]\s*(Low|Moderate|High|低危|中危|高危)",
        ]
        for pattern in risk_patterns_30d:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content["30day_risk"] = match.group(1)
                break
        
        # 提取1年风险
        risk_patterns_1y = [
            r"1-year MACE.*?Risk level.*?[：:]\s*(Low|Moderate|High|低危|中危|高危)",
            r"1年MACE.*?风险等级.*?[：:]\s*(Low|Moderate|High|低危|中危|高危)",
        ]
        for pattern in risk_patterns_1y:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content["1year_risk"] = match.group(1)
                break
        
        # 提取事件率
        rate_30d = re.search(r"30.*?(?:event rate|事件率)[：:]\s*(\d+\.?\d*)%?", response, re.IGNORECASE)
        if rate_30d:
            content["30day_rate"] = f"{rate_30d.group(1)}%"
        
        rate_1y = re.search(r"1.*?year.*?(?:event rate|事件率)[：:]\s*(\d+\.?\d*)%?", response, re.IGNORECASE)
        if rate_1y:
            content["1year_rate"] = f"{rate_1y.group(1)}%"
        
        # 提取置信度
        conf_match = re.search(r"(?:Confidence|置信度)[：:]\s*(\d+\.?\d*)", response, re.IGNORECASE)
        if conf_match:
            confidence = min(float(conf_match.group(1)), 1.0)
        
        return content, confidence, reasoning
