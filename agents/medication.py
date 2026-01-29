#!/usr/bin/env python3
"""
T4: 用药推荐Agent
PCI术后二级预防用药推荐
"""

import re
from typing import Dict, Any, Tuple, List
from .base import BaseAgent, AgentOutput
from ..rag import get_retriever
from ..prompts import get_prompts


class MedicationAgent(BaseAgent):
    """用药推荐Agent (T4)"""
    
    def __init__(self, llm, language: str = "zh"):
        super().__init__(
            name="medication",
            llm=llm,
            description="推荐PCI术后二级预防用药方案"
        )
        self.language = language
        self._prompts = get_prompts(language)
        
        # Initialize RAG retriever
        self.retriever = get_retriever()
        
        # PCI术后标准二级预防药物
        self.standard_medications = {
            "抗血小板药": ["阿司匹林", "氯吡格雷", "替格瑞洛", "普拉格雷", "Aspirin", "Clopidogrel", "Ticagrelor"],
            "他汀类": ["阿托伐他汀", "瑞舒伐他汀", "辛伐他汀", "普伐他汀", "Atorvastatin", "Rosuvastatin"],
            "ACEI/ARB": ["培哚普利", "贝那普利", "缬沙坦", "厄贝沙坦", "氯沙坦", "Perindopril", "Valsartan"],
            "β受体阻滞剂": ["美托洛尔", "比索洛尔", "卡维地洛", "阿替洛尔", "Metoprolol", "Bisoprolol"]
        }
    
    @property
    def system_prompt(self) -> str:
        return self._prompts["medication"]

    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        # 输入字段 (不包含金标准 Discharge_* 用药字段)
        relevant_fields = [
            # 诊断信息 - 决定用药的主要依据
            'Admission_Diagnosis_Western',  # 入院诊断
            # 心功能 (从上游获取更好，但这里作为参考)
            # Pre_LVEF 应从上游 cardiac_function Agent 获取
            # 肾功能 - 影响用药选择
            'Pre_PCI_eGFR',                 # eGFR
            'Creatinine',                   # 肌酐
            # 病史 - 影响用药禁忌
            'Hypertension_History',
            'Diabetes_History',
            'CKD_History',                  # 慢性肾病史
            # 年龄等
            'Age',
            # 兼容旧字段名
            '入院诊断', '既往史'
            # 注意: Discharge_Statin, Discharge_Antiplatelets, 
            #       Discharge_Beta_Blocker, Discharge_ACEI_ARB 是金标准，不能作为输入
        ]
        
        patient_text = []
        for field in relevant_fields:
            if field in patient_info and patient_info[field]:
                value = str(patient_info[field])
                if value and value.strip() and value != 'nan':
                    patient_text.append(f"【{field}】{value}")
        
        # Get upstream results
        upstream_info = ""
        
        if self.language == "en":
            # English version
            if "cardiac_function" in upstream_outputs:
                cardiac = upstream_outputs["cardiac_function"]
                upstream_info += f"""
Cardiac Function Assessment:
- LVEF Grade: {cardiac.content.get('lvef_grade', 'Unknown')}
- LVEF Range: {cardiac.content.get('lvef_range', 'Unknown')}
- Confidence: {cardiac.confidence:.2f}
"""
            
            if "coronary" in upstream_outputs:
                coronary = upstream_outputs["coronary"]
                upstream_info += f"""
Coronary Lesion Assessment:
- Affected Vessels: {', '.join(coronary.content.get('vessels', ['Unknown']))}
- Lesion Extent: {coronary.content.get('vessel_count', 'Unknown')}
- Intervention: {coronary.content.get('intervention', 'Unknown')}
"""
            
            if "diagnosis" in upstream_outputs:
                diag = upstream_outputs["diagnosis"]
                upstream_info += f"""
Diagnosis Results:
- Primary Diagnosis: {diag.content.get('main_diagnosis', 'Unknown')}
- Comorbidities: {', '.join(diag.content.get('comorbidities', ['Unknown']))}
"""
        else:
            # Chinese version
            if "cardiac_function" in upstream_outputs:
                cardiac = upstream_outputs["cardiac_function"]
                upstream_info += f"""
心功能评估结果：
- LVEF分级：{cardiac.content.get('lvef_grade', '未知')}
- LVEF范围：{cardiac.content.get('lvef_range', '未知')}
- 置信度：{cardiac.confidence:.2f}
"""
            
            if "coronary" in upstream_outputs:
                coronary = upstream_outputs["coronary"]
                upstream_info += f"""
冠脉病变评估：
- 受累血管：{', '.join(coronary.content.get('vessels', ['未知']))}
- 病变范围：{coronary.content.get('vessel_count', '未知')}
- 干预方式：{coronary.content.get('intervention', '未知')}
"""
            
            if "diagnosis" in upstream_outputs:
                diag = upstream_outputs["diagnosis"]
                upstream_info += f"""
诊断结果：
- 主诊断：{diag.content.get('main_diagnosis', '未知')}
- 合并症：{', '.join(diag.content.get('comorbidities', ['未知']))}
"""
        
        # RAG: Retrieve relevant guidelines based on diagnosis and comorbidities
        rag_context = ""
        try:
            main_diagnosis = ""
            comorbidities = []
            if "diagnosis" in upstream_outputs:
                diag = upstream_outputs["diagnosis"]
                main_diagnosis = diag.content.get('main_diagnosis', '')
                comorbidities = diag.content.get('comorbidities', [])
            
            # Fallback to patient_info if no upstream diagnosis
            if not main_diagnosis:
                main_diagnosis = patient_info.get('出院诊断', '') or patient_info.get('入院诊断', '')
            
            rag_context = self.retriever.retrieve_for_medication(
                diagnosis=main_diagnosis,
                comorbidities=comorbidities,
                language=self.language
            )
        except Exception as e:
            print(f"[RAG] Warning: Retrieval failed: {e}")
        
        if self.language == "en":
            return f"""Please recommend a secondary prevention medication regimen for the following post-PCI patient.

## Patient Medical Record
{chr(10).join(patient_text) if patient_text else 'Limited information'}

{upstream_info}

{rag_context}

## Please output in the following format

### Medication Analysis
(Analyze patient characteristics, contraindications, special considerations)

### Recommended Medication Regimen
1. Antiplatelet: [Drug name and dosage]
2. Statin: [Drug name and dosage]
3. ACEI/ARB: [Drug name and dosage, or explain why not needed]
4. Beta-blocker: [Drug name and dosage, or explain why not needed]

### Special Medications
(Other medications if needed: PPI, antidiabetic agents, etc.)

### Precautions
(Medication warnings, monitoring requirements)

### Conclusion
- Number of recommended medications: [X]
- Complete quadruple therapy: [Yes/No]
- Confidence: [0.0-1.0]
"""
        else:
            return f"""请为以下PCI术后患者推荐二级预防用药方案。

## 患者病历信息
{chr(10).join(patient_text) if patient_text else '信息有限'}

{upstream_info}

{rag_context}

## 请按以下格式输出

### 用药分析
（分析患者特点、禁忌症、特殊考虑）

### 推荐用药方案
1. 抗血小板：[药物名称和剂量]
2. 他汀类：[药物名称和剂量]
3. ACEI/ARB：[药物名称和剂量，或说明不需要的原因]
4. β受体阻滞剂：[药物名称和剂量，或说明不需要的原因]

### 特殊用药
（如有其他需要的药物：PPI、降糖药等）

### 注意事项
（用药警告、监测要求）

### 结论
- 推荐药物数量：[X种]
- 完整四联：[是/否]
- 置信度：[0.0-1.0]
"""


    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        content = {
            "antiplatelet": [],
            "statin": "",
            "acei_arb": "",
            "beta_blocker": "",
            "other_meds": [],
            "warnings": [],
            "is_complete_quad": False
        }
        confidence = 0.5
        reasoning = ""
        
        # 提取分析过程
        if "### 用药分析" in response:
            parts = response.split("### 用药分析")
            if len(parts) > 1:
                reasoning_part = parts[1].split("### ")[0]
                reasoning = reasoning_part.strip()
        
        # 提取抗血小板药
        for drug in self.standard_medications["抗血小板药"]:
            if drug in response:
                content["antiplatelet"].append(drug)
        
        # 提取他汀
        for drug in self.standard_medications["他汀类"]:
            if drug in response:
                content["statin"] = drug
                break
        
        # 提取ACEI/ARB
        for drug in self.standard_medications["ACEI/ARB"]:
            if drug in response:
                content["acei_arb"] = drug
                break
        
        # 提取β阻滞剂
        for drug in self.standard_medications["β受体阻滞剂"]:
            if drug in response:
                content["beta_blocker"] = drug
                break
        
        # 判断是否完整四联
        has_antiplatelet = len(content["antiplatelet"]) >= 2  # DAPT
        has_statin = bool(content["statin"])
        has_acei = bool(content["acei_arb"])
        has_beta = bool(content["beta_blocker"])
        content["is_complete_quad"] = all([has_antiplatelet, has_statin, has_acei, has_beta])
        
        # 提取置信度
        conf_match = re.search(r"置信度[：:]\s*(\d+\.?\d*)", response)
        if conf_match:
            confidence = min(float(conf_match.group(1)), 1.0)
        
        return content, confidence, reasoning
