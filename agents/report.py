#!/usr/bin/env python3
"""
T5: MDT Consultation Report Generation Agent
Generate multidisciplinary team consultation report
"""

import re
from typing import Dict, Any, Tuple, List
from .base import BaseAgent, AgentOutput
from ..prompts import get_prompts


class ReportAgent(BaseAgent):
    """MDT Consultation Report Generation Agent (T5)"""
    
    def __init__(self, llm, language: str = "zh"):
        super().__init__(
            name="report",
            llm=llm,
            description="Generate multidisciplinary team consultation report"
        )
        self.language = language
        self._prompts = get_prompts(language)
    
    @property
    def system_prompt(self) -> str:
        return self._prompts["report"]

    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        basic_fields = [
            '序号', 'Study_ID', 'Patient_ID', 'patient_id',
            '性别', 'Sex', 'gender',
            '主诉', 'Chief_Complaint', 'chief_complaint',
            '入院诊断', 'Admission_Diagnosis',
            '手术及操作名称', 'PCI_术式', 'pci_procedure'
        ]
        
        patient_text = []
        for field in basic_fields:
            if field in patient_info and patient_info[field]:
                value = str(patient_info[field])
                if value and value.strip() and value.lower() != 'nan':
                    patient_text.append(f"- {field}: {value}")
        
        # Build specialist reports based on language
        specialist_reports = ""
        
        if self.language == "en":
            if "coronary" in upstream_outputs:
                coronary = upstream_outputs["coronary"]
                specialist_reports += f"""
### Interventional Cardiology
- Affected vessels: {', '.join(coronary.content.get('vessels', ['N/A']))}
- Lesion extent: {coronary.content.get('vessel_count', 'N/A')}
- Intervention: {coronary.content.get('intervention', 'N/A')}
- Confidence: {coronary.confidence:.2f}
"""
            
            if "cardiac_function" in upstream_outputs:
                cardiac = upstream_outputs["cardiac_function"]
                specialist_reports += f"""
### Cardiac Imaging
- LVEF grade: {cardiac.content.get('lvef_grade', 'N/A')}
- LVEF range: {cardiac.content.get('lvef_range', 'N/A')}
- Confidence: {cardiac.confidence:.2f}
"""
            
            if "diagnosis" in upstream_outputs:
                diag = upstream_outputs["diagnosis"]
                specialist_reports += f"""
### Comprehensive Diagnosis
- Main diagnosis: {diag.content.get('main_diagnosis', 'N/A')}
- Comorbidities: {', '.join(diag.content.get('comorbidities', ['None']))}
- Confidence: {diag.confidence:.2f}
"""
            
            if "medication" in upstream_outputs:
                med = upstream_outputs["medication"]
                antiplatelet = ', '.join(med.content.get('antiplatelet', [])) or 'N/A'
                specialist_reports += f"""
### Clinical Pharmacy
- Antiplatelet: {antiplatelet}
- Statin: {med.content.get('statin', 'N/A')}
- ACEI/ARB: {med.content.get('acei_arb', 'N/A')}
- Beta-blocker: {med.content.get('beta_blocker', 'N/A')}
- Complete quadruple therapy: {'Yes' if med.content.get('is_complete_quad') else 'No'}
- Confidence: {med.confidence:.2f}
"""
            
            return f"""Please generate a comprehensive MDT consultation report based on all specialist assessments.

## Patient Basic Information
{chr(10).join(patient_text) if patient_text else 'Limited information'}

## Specialist Assessments
{specialist_reports if specialist_reports else 'No upstream assessments available'}

## Output: MDT Consultation Report

Please generate a structured report following this format:

---
# MDT Consultation Report

## 1. Patient Summary
(Brief overview of the case)

## 2. Specialist Consensus
(Key findings agreed upon by all specialists)

## 3. Integrated Diagnosis
(Final diagnosis integrating all assessments)

## 4. Treatment Recommendations
(Comprehensive treatment plan)

## 5. Follow-up Plan
(Monitoring and follow-up schedule)

## 6. Discussion Points
(Any disagreements or areas requiring further evaluation)

## 7. MDT Conclusion
(Overall summary and key takeaways)

---

### Report Quality Confidence: [0.0-1.0]
"""
        else:
            # Chinese version
            if "coronary" in upstream_outputs:
                coronary = upstream_outputs["coronary"]
                specialist_reports += f"""
### 介入心脏病学
- 受累血管: {', '.join(coronary.content.get('vessels', ['未知']))}
- 病变范围: {coronary.content.get('vessel_count', '未知')}
- 干预方式: {coronary.content.get('intervention', '未知')}
- 置信度: {coronary.confidence:.2f}
"""
            
            if "cardiac_function" in upstream_outputs:
                cardiac = upstream_outputs["cardiac_function"]
                specialist_reports += f"""
### 心脏影像
- LVEF分级: {cardiac.content.get('lvef_grade', '未知')}
- LVEF范围: {cardiac.content.get('lvef_range', '未知')}
- 置信度: {cardiac.confidence:.2f}
"""
            
            if "diagnosis" in upstream_outputs:
                diag = upstream_outputs["diagnosis"]
                specialist_reports += f"""
### 综合诊断
- 主诊断: {diag.content.get('main_diagnosis', '未知')}
- 合并症: {', '.join(diag.content.get('comorbidities', ['无']))}
- 置信度: {diag.confidence:.2f}
"""
            
            if "medication" in upstream_outputs:
                med = upstream_outputs["medication"]
                antiplatelet = ', '.join(med.content.get('antiplatelet', [])) or '未知'
                specialist_reports += f"""
### 临床药学
- 抗血小板: {antiplatelet}
- 他汀: {med.content.get('statin', '未知')}
- ACEI/ARB: {med.content.get('acei_arb', '未知')}
- β阻滞剂: {med.content.get('beta_blocker', '未知')}
- 完整四联: {'是' if med.content.get('is_complete_quad') else '否'}
- 置信度: {med.confidence:.2f}
"""
            
            return f"""请根据所有专科评估生成一份完整的MDT会诊报告。

## 患者基本信息
{chr(10).join(patient_text) if patient_text else '信息有限'}

## 专科评估
{specialist_reports if specialist_reports else '无上游评估信息'}

## 输出：MDT会诊报告

请按以下格式生成结构化报告：

---
# MDT会诊报告

## 1. 患者摘要
（病例简述）

## 2. 专家共识
（各专科一致认同的关键发现）

## 3. 综合诊断
（整合所有评估的最终诊断）

## 4. 治疗建议
（综合治疗方案）

## 5. 随访计划
（监测和随访安排）

## 6. 讨论要点
（任何分歧或需进一步评估的领域）

## 7. MDT结论
（总体总结和关键要点）

---

### 报告质量置信度: [0.0-1.0]
"""

    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        content = {
            "patient_summary": "",
            "specialist_consensus": "",
            "integrated_diagnosis": "",
            "treatment_recommendations": "",
            "followup_plan": "",
            "discussion_points": "",
            "mdt_conclusion": "",
            "full_report": response  # Store full report
        }
        confidence = 0.5
        reasoning = ""
        
        # Extract sections - both English and Chinese patterns
        sections = {
            "patient_summary": [
                r"## 1\. Patient Summary.*?\n(.*?)(?=## 2\.|$)",
                r"## 1\. 患者摘要.*?\n(.*?)(?=## 2\.|$)"
            ],
            "specialist_consensus": [
                r"## 2\. Specialist Consensus.*?\n(.*?)(?=## 3\.|$)",
                r"## 2\. 专家共识.*?\n(.*?)(?=## 3\.|$)"
            ],
            "integrated_diagnosis": [
                r"## 3\. Integrated Diagnosis.*?\n(.*?)(?=## 4\.|$)",
                r"## 3\. 综合诊断.*?\n(.*?)(?=## 4\.|$)"
            ],
            "treatment_recommendations": [
                r"## 4\. Treatment Recommendations.*?\n(.*?)(?=## 5\.|$)",
                r"## 4\. 治疗建议.*?\n(.*?)(?=## 5\.|$)"
            ],
            "followup_plan": [
                r"## 5\. Follow-up Plan.*?\n(.*?)(?=## 6\.|$)",
                r"## 5\. 随访计划.*?\n(.*?)(?=## 6\.|$)"
            ],
            "mdt_conclusion": [
                r"## 7\. MDT Conclusion.*?\n(.*?)(?=###|$)",
                r"## 7\. MDT结论.*?\n(.*?)(?=###|$)"
            ],
        }
        
        for key, patterns in sections.items():
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    content[key] = match.group(1).strip()[:500]  # Limit length
                    break
        
        # Use patient summary as reasoning
        reasoning = content.get("patient_summary", "")
        
        # Extract confidence
        conf_match = re.search(
            r"(?:Report Quality Confidence|报告质量置信度)[：:]?\s*([\d.]+)", 
            response, 
            re.IGNORECASE
        )
        if conf_match:
            confidence = min(float(conf_match.group(1)), 1.0)
        
        return content, confidence, reasoning
