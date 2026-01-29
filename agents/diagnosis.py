#!/usr/bin/env python3
"""
T3: 诊断综合评估Agent
整合多源信息生成完整诊断
"""

import re
from typing import Dict, Any, Tuple, List
from .base import BaseAgent, AgentOutput
from ..prompts import get_prompts


class DiagnosisAgent(BaseAgent):
    """诊断综合评估Agent (T3)"""
    
    def __init__(self, llm, language: str = "zh"):
        super().__init__(
            name="diagnosis",
            llm=llm,
            description="Integrate multi-source information to generate comprehensive diagnosis / 整合多源信息生成完整诊断"
        )
        self.language = language
        self._prompts = get_prompts(language)
    
    @property
    def system_prompt(self) -> str:
        return self._prompts["diagnosis"]

    def build_task_prompt(self, patient_info: Dict, upstream_outputs: Dict[str, AgentOutput]) -> str:
        # 匹配数据集实际字段名
        relevant_fields = [
            # 诊断字段
            'Admission_Diagnosis_Western',   # 入院诊断
            'Discharge_Diagnosis_Western',   # 出院诊断
            'Chief_Complaint',               # 主诉
            # 病史
            'Hypertension_History',          # 高血压史
            'Diabetes_History',              # 糖尿病史
            'Dyslipidemia_History',          # 血脂异常史
            'CAD_History',                   # 冠心病史
            'CKD_History',                   # 慢性肾病史
            # 手术信息
            'PCI_Procedure_Type',            # 手术类型
            'Culprit_Vessel',                # 罪犯血管
            # 兼容旧字段名
            '主诉', '现病史', '既往史', '入院诊断', '出院诊断',
        ]
        
        patient_text = []
        for field in relevant_fields:
            if field in patient_info and patient_info[field]:
                value = str(patient_info[field])
                if value and value.strip() and value.lower() != 'nan':
                    patient_text.append(f"【{field}】{value}")
        
        # 获取上游结果
        upstream_info = ""
        
        if "coronary" in upstream_outputs:
            coronary = upstream_outputs["coronary"]
            upstream_info += f"""
## Upstream: Coronary Lesion Assessment / 上游：冠脉病变评估
- Affected vessels / 受累血管: {', '.join(coronary.content.get('vessels', ['Unknown/未知']))}
- Lesion extent / 病变范围: {coronary.content.get('vessel_count', 'Unknown/未知')}
- Intervention / 干预方式: {coronary.content.get('intervention', 'Unknown/未知')}
- Confidence / 置信度: {coronary.confidence:.2f}
"""
        
        if "cardiac_function" in upstream_outputs:
            cardiac = upstream_outputs["cardiac_function"]
            upstream_info += f"""
## Upstream: Cardiac Function Assessment / 上游：心功能评估
- LVEF grade / LVEF分级: {cardiac.content.get('lvef_grade', 'Unknown/未知')}
- LVEF range / LVEF范围: {cardiac.content.get('lvef_range', 'Unknown/未知')}
- Confidence / 置信度: {cardiac.confidence:.2f}
"""
        
        return f"""Please generate a comprehensive diagnosis based on the following information.
请根据以下信息生成完整诊断。

## Patient Medical Record / 患者病历信息
{chr(10).join(patient_text) if patient_text else 'Limited information / 信息有限'}

{upstream_info}

## Output Format / 输出格式

### Clinical Analysis / 临床分析
(Analyze key findings from history, examination, and upstream agents)
(分析病史、检查和上游Agent的关键发现)

### Diagnosis / 诊断
1. Main Diagnosis / 主诊断: [ICD-10 format]
2. Comorbidities / 合并症:
   - [Comorbidity 1 / 合并症1]
   - [Comorbidity 2 / 合并症2]
3. Complications / 并发症 (if any): [...]

### Key Evidence / 关键依据
- [Evidence 1 / 依据1]
- [Evidence 2 / 依据2]

### Confidence / 置信度: [0.0-1.0]
"""

    def parse_response(self, response: str) -> Tuple[Dict[str, Any], float, str]:
        content = {
            "main_diagnosis": "",
            "comorbidities": [],
            "complications": [],
            "key_evidence": []
        }
        confidence = 0.5
        reasoning = ""
        
        # =====================================================
        # 首先尝试从JSON块中解析 (处理verify()的响应格式)
        # First try to parse from JSON block (handles verify() response)
        # =====================================================
        import json
        json_pattern = r'```json\s*([\s\S]*?)```'
        json_match = re.search(json_pattern, response)
        if json_match:
            try:
                json_content = json.loads(json_match.group(1))
                if isinstance(json_content, dict):
                    if "main_diagnosis" in json_content:
                        content["main_diagnosis"] = str(json_content.get("main_diagnosis", ""))
                    if "comorbidities" in json_content:
                        comorbidities = json_content.get("comorbidities", [])
                        if isinstance(comorbidities, list):
                            content["comorbidities"] = [str(c) for c in comorbidities]
                    if "complications" in json_content:
                        complications = json_content.get("complications", [])
                        if isinstance(complications, list):
                            content["complications"] = [str(c) for c in complications]
                    if "key_evidence" in json_content:
                        evidence = json_content.get("key_evidence", [])
                        if isinstance(evidence, list):
                            content["key_evidence"] = [str(e) for e in evidence]
                    
                    # 如果成功解析JSON且有主诊断，直接返回
                    if content["main_diagnosis"]:
                        # 提取置信度
                        conf_match = re.search(r'(?:置信度|Confidence|新的置信度)[：:\s]*[*]*(\d+\.?\d*)', response, re.IGNORECASE)
                        if conf_match:
                            confidence = min(float(conf_match.group(1)), 1.0)
                        # 使用响应开头作为reasoning
                        reasoning = response[:1500]
                        return content, confidence, reasoning
            except json.JSONDecodeError:
                pass  # JSON解析失败，继续使用正则表达式
        
        # =====================================================
        # 正则表达式解析 (处理process()的响应格式)
        # Regex parsing (handles process() response)
        # =====================================================
        
        # 提取分析过程
        analysis_patterns = [
            r"(?:### Clinical Analysis|### 临床分析|## 临床分析|临床分析)(.*?)(?=### |## |$)",
            r"(?:分析|推理过程)(.*?)(?=### |## |诊断|$)",
        ]
        for pattern in analysis_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()[:2000]  # 限制长度
                break
        
        if not reasoning:
            # 如果没找到专门的分析部分，使用整个响应的前半部分
            reasoning = response[:1500]
        
        # 提取主诊断 - 使用更多模式
        main_patterns = [
            # 标准格式
            r"(?:Main Diagnosis|主诊断)[：:\s]*\n?\s*(.+?)(?:\n|$)",
            r"1\.\s*(?:Main Diagnosis|主诊断)[：:\s]*(.+?)(?:\n|$)",
            r"主诊断[：:\s]*(.+?)(?:\n|$)",
            # 带标记的格式
            r"\*\*主诊断\*\*[：:\s]*(.+?)(?:\n|$)",
            r"\*\*Main Diagnosis\*\*[：:\s]*(.+?)(?:\n|$)",
            # 简化格式
            r"(?:诊断|Diagnosis)[：:]\s*\n?[-\d\.\s]*(.+?)(?:\n[-\d]|\n\*|\n#|$)",
            # 从综合诊断部分提取
            r"综合诊断.*?主诊断[：:\s]*(.+?)(?:\n|合并症|$)",
            # 列表格式 - 提取第一项
            r"(?:### 诊断|## Diagnosis).*?\n[-\d\.\s]*(\S.+?)(?:\n|$)",
        ]
        
        for pattern in main_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                diagnosis = match.group(1).strip()
                # 清理诊断文本 - 移除各种格式字符
                diagnosis = re.sub(r'^[\d\.\s\-\*\/\:：]+', '', diagnosis)  # 移除开头的编号和格式符
                diagnosis = re.sub(r'^\*\*主诊断\*\*[：:\s]*', '', diagnosis)  # 移除开头的markdown标记
                diagnosis = re.sub(r'^主诊断[：:\s]*', '', diagnosis)  # 移除"主诊断:"前缀
                diagnosis = re.sub(r'^\*\*', '', diagnosis)  # 移除开头的**
                diagnosis = re.sub(r'\*\*$', '', diagnosis)  # 移除结尾的**
                diagnosis = diagnosis.strip()
                if diagnosis and len(diagnosis) > 3:  # 确保有实际内容
                    content["main_diagnosis"] = diagnosis
                    break
        
        # 如果仍然没有找到主诊断，尝试从reasoning中提取关键诊断
        if not content["main_diagnosis"] and reasoning:
            # 查找常见诊断关键词
            diagnosis_keywords = [
                r"诊断(?:为|：|:)(.+?)(?:\n|，|。|$)",
                r"(?:冠心病|心绞痛|心肌梗死|STEMI|NSTEMI|ACS|冠状动脉粥样硬化性心脏病).*?(?:PCI|术后|支架|病变)",
                r"(?:稳定型心绞痛|不稳定型心绞痛|急性冠脉综合征)",
            ]
            for pattern in diagnosis_keywords:
                match = re.search(pattern, reasoning, re.IGNORECASE)
                if match:
                    content["main_diagnosis"] = match.group(0).strip()[:200]
                    break
        
        # 提取合并症
        comorbidity_section = ""
        com_patterns = [
            r"(?:Comorbidities|合并症)[：:](.*?)(?=###|##|Complications|并发症|Confidence|置信度|关键依据|Key Evidence|$)",
            r"2\.\s*(?:Comorbidities|合并症)[：:]?(.*?)(?=3\.|###|##|$)",
            r"\*\*合并症\*\*[：:]?(.*?)(?=\*\*|###|##|$)",
        ]
        for pattern in com_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                comorbidity_section = match.group(1)
                break
        
        if comorbidity_section:
            # 提取列表项
            items = re.findall(r'[-•\*]\s*(.+?)(?:\n|$)', comorbidity_section)
            if not items:
                # 尝试提取编号项
                items = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', comorbidity_section)
            content["comorbidities"] = [item.strip() for item in items if item.strip() and len(item.strip()) > 2]
        
        # 提取并发症
        comp_patterns = [
            r"(?:Complications|并发症)[：:](.*?)(?=###|##|Confidence|置信度|Key Evidence|关键依据|$)",
        ]
        for pattern in comp_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                comp_section = match.group(1)
                items = re.findall(r'[-•\*]\s*(.+?)(?:\n|$)', comp_section)
                content["complications"] = [item.strip() for item in items if item.strip() and len(item.strip()) > 2]
                break
        
        # 提取关键依据
        evidence_patterns = [
            r"(?:Key Evidence|关键依据)[：:](.*?)(?=###|##|Confidence|置信度|$)",
        ]
        for pattern in evidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                ev_section = match.group(1)
                items = re.findall(r'[-•\*]\s*(.+?)(?:\n|$)', ev_section)
                content["key_evidence"] = [item.strip() for item in items if item.strip()][:5]
                break
        
        # 提取置信度
        conf_patterns = [
            r"(?:Confidence|置信度)[：:\s]*(\d+\.?\d*)",
            r"\*\*置信度\*\*[：:\s]*(\d+\.?\d*)",
            r"置信度.*?(\d+\.?\d*)",
        ]
        for pattern in conf_patterns:
            conf_match = re.search(pattern, response, re.IGNORECASE)
            if conf_match:
                try:
                    conf_val = float(conf_match.group(1))
                    if conf_val > 1:
                        conf_val = conf_val / 100  # 处理百分比格式
                    confidence = min(max(conf_val, 0.1), 1.0)
                    break
                except:
                    pass
        
        return content, confidence, reasoning
