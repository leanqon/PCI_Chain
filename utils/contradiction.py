#!/usr/bin/env python3
"""
矛盾检测模块 (重新设计版)
Contradiction Detection Module (Redesigned)

设计原则 / Design Principles:
- 医学决策规则 → 放在Prompt中指导LLM
- 矛盾检测规则 → 检查Agent输出之间的逻辑一致性

规则类型 / Rule Types:
1. 数值一致性 - 不同Agent引用同一数值是否一致
2. 分类一致性 - 分类结果是否一致 (如病变范围)
3. 引用正确性 - 下游是否正确引用上游信息
4. 完整性检查 - 下游是否遗漏上游关键信息
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Contradiction:
    """矛盾记录 / Contradiction Record"""
    rule_name: str          # 规则名称
    description: str        # 矛盾描述
    upstream_agent: str     # 涉及的上游Agent
    downstream_agent: str   # 检测到矛盾的下游Agent
    upstream_value: str     # 上游的值
    downstream_value: str   # 下游的值
    query: str              # 反馈给上游的质疑
    severity: str = "medium"  # 严重程度: low/medium/high


class ContradictionDetector:
    """
    矛盾检测器 (逻辑一致性检测)
    Contradiction Detector (Logical Consistency Checking)
    
    不检查医学规则，只检查Agent输出之间的逻辑一致性。
    Does NOT check medical rules, only checks logical consistency between agent outputs.
    """
    
    def __init__(self, llm=None, enable_llm: bool = True, enable_rules: bool = True):
        self.llm = llm
        self.enable_llm = enable_llm and llm is not None
        self.enable_rules = enable_rules
        
        # 定义一致性检测规则 (非医学规则)
        # Define consistency rules (NOT medical rules)
        self.rules = [
            {
                "name": "vessel_count_consistency",
                "description": "病变血管数目一致性 / Vessel count consistency",
                "check": self._check_vessel_count_consistency
            },
            {
                "name": "vessel_list_consistency", 
                "description": "病变血管列表一致性 / Vessel list consistency",
                "check": self._check_vessel_list_consistency
            },
            {
                "name": "lvef_description_consistency",
                "description": "LVEF描述一致性 / LVEF description consistency",
                "check": self._check_lvef_consistency
            },
            {
                "name": "diagnosis_citation_accuracy",
                "description": "诊断引用准确性 / Diagnosis citation accuracy",
                "check": self._check_diagnosis_citation
            },
            {
                "name": "medication_completeness",
                "description": "用药信息完整性 / Medication completeness in report",
                "check": self._check_medication_completeness
            },
        ]
    
    def detect(self, current_agent_name: str, current_output: Any, 
               all_outputs: Dict[str, Any]) -> List[Contradiction]:
        """检测矛盾 / Detect contradictions"""
        contradictions = []
        
        # 规则检测
        if self.enable_rules:
            for rule in self.rules:
                result = rule["check"](current_agent_name, current_output, all_outputs)
                if result:
                    contradictions.append(result)
        
        # LLM辅助检测 (检测隐含的逻辑不一致)
        if self.enable_llm and self.llm:
            llm_contradictions = self._llm_detect(current_agent_name, current_output, all_outputs)
            contradictions.extend(llm_contradictions)
        
        return contradictions
    
    # =========================================================================
    # 一致性检测规则 (非医学规则)
    # Consistency Rules (NOT medical rules)
    # =========================================================================
    
    def _check_vessel_count_consistency(self, agent_name: str, output: Any, 
                                         all_outputs: Dict) -> Optional[Contradiction]:
        """
        检查病变血管数目一致性
        Check if vessel count is consistent between coronary and diagnosis agents
        
        示例矛盾:
        - T1冠脉: vessels=["LAD", "RCA"] (2支)
        - T3诊断: "单支病变"
        """
        if agent_name != "diagnosis":
            return None
        
        coronary_output = all_outputs.get("coronary")
        if not coronary_output:
            return None
        
        # 获取上游血管列表
        vessels = coronary_output.content.get("vessels", [])
        vessel_count_from_coronary = len(vessels)
        
        # 获取下游诊断中的病变范围描述
        main_diagnosis = str(output.content.get("main_diagnosis", ""))
        
        # 解析诊断中的病变数目
        diagnosis_vessel_count = self._extract_vessel_count_from_text(main_diagnosis)
        
        if diagnosis_vessel_count is None:
            return None  # 诊断中没有明确提及病变范围
        
        # 检查一致性
        if vessel_count_from_coronary != diagnosis_vessel_count:
            return Contradiction(
                rule_name="vessel_count_consistency",
                description=f"冠脉Agent识别{vessel_count_from_coronary}支病变，但诊断Agent描述为{diagnosis_vessel_count}支病变",
                upstream_agent="coronary",
                downstream_agent="diagnosis",
                upstream_value=f"{vessel_count_from_coronary}支 ({', '.join(vessels)})",
                downstream_value=f"{diagnosis_vessel_count}支",
                query=f"您识别了{vessel_count_from_coronary}支血管病变{vessels}，但诊断描述为{diagnosis_vessel_count}支病变。请确认血管识别是否准确？",
                severity="high"
            )
        
        return None
    
    def _check_vessel_list_consistency(self, agent_name: str, output: Any,
                                        all_outputs: Dict) -> Optional[Contradiction]:
        """
        检查病变血管列表一致性
        Check if vessel names mentioned in diagnosis match coronary output
        
        示例矛盾:
        - T1冠脉: vessels=["LAD"]
        - T3诊断: "RCA病变行PCI"
        """
        if agent_name != "diagnosis":
            return None
        
        coronary_output = all_outputs.get("coronary")
        if not coronary_output:
            return None
        
        coronary_vessels = set(coronary_output.content.get("vessels", []))
        main_diagnosis = str(output.content.get("main_diagnosis", ""))
        
        # 从诊断文本中提取提及的血管
        diagnosis_vessels = self._extract_vessels_from_text(main_diagnosis)
        
        if not diagnosis_vessels:
            return None
        
        # 检查诊断中提及的血管是否都在冠脉输出中
        extra_vessels = diagnosis_vessels - coronary_vessels
        if extra_vessels:
            return Contradiction(
                rule_name="vessel_list_consistency",
                description=f"诊断提及血管{extra_vessels}，但冠脉Agent未识别这些血管",
                upstream_agent="coronary",
                downstream_agent="diagnosis",
                upstream_value=str(coronary_vessels),
                downstream_value=str(diagnosis_vessels),
                query=f"诊断引用了血管{extra_vessels}，但您的输出中只有{coronary_vessels}。请确认是否遗漏了血管？",
                severity="medium"
            )
        
        return None
    
    def _check_lvef_consistency(self, agent_name: str, output: Any,
                                 all_outputs: Dict) -> Optional[Contradiction]:
        """
        检查LVEF描述一致性
        Check if LVEF descriptions are consistent between cardiac and diagnosis
        
        示例矛盾:
        - T2心功能: lvef_grade="正常"
        - T3诊断: "心功能不全" 或 "LVEF降低"
        """
        if agent_name != "diagnosis":
            return None
        
        cardiac_output = all_outputs.get("cardiac_function")
        if not cardiac_output:
            return None
        
        cardiac_lvef = cardiac_output.content.get("lvef_grade", "")
        main_diagnosis = str(output.content.get("main_diagnosis", ""))
        comorbidities = output.content.get("comorbidities", [])
        all_diagnosis_text = main_diagnosis + " " + " ".join(str(c) for c in comorbidities)
        
        # 定义LVEF相关关键词
        reduced_keywords = ["心功能不全", "心衰", "心力衰竭", "LVEF降低", "射血分数降低", 
                           "heart failure", "reduced EF", "HFrEF"]
        normal_keywords = ["心功能正常", "LVEF正常", "射血分数保留", "preserved EF", "HFpEF"]
        
        # 检查一致性
        diagnosis_suggests_reduced = any(kw in all_diagnosis_text for kw in reduced_keywords)
        diagnosis_suggests_normal = any(kw in all_diagnosis_text for kw in normal_keywords)
        
        if cardiac_lvef in ["正常", "Normal", "HFpEF"] and diagnosis_suggests_reduced:
            return Contradiction(
                rule_name="lvef_description_consistency",
                description=f"心功能Agent评估为'{cardiac_lvef}'，但诊断中提及心功能不全相关描述",
                upstream_agent="cardiac_function",
                downstream_agent="diagnosis",
                upstream_value=cardiac_lvef,
                downstream_value="诊断提及心功能不全",
                query=f"您评估LVEF为'{cardiac_lvef}'，但诊断中提及心功能不全。请确认LVEF评估是否准确？",
                severity="high"
            )
        
        if cardiac_lvef in ["中度降低", "重度降低", "Moderately Reduced", "Severely Reduced"] and diagnosis_suggests_normal:
            return Contradiction(
                rule_name="lvef_description_consistency",
                description=f"心功能Agent评估为'{cardiac_lvef}'，但诊断描述心功能正常",
                upstream_agent="cardiac_function",
                downstream_agent="diagnosis",
                upstream_value=cardiac_lvef,
                downstream_value="诊断描述心功能正常",
                query=f"您评估LVEF为'{cardiac_lvef}'，但诊断描述心功能正常。请确认LVEF评估是否准确？",
                severity="high"
            )
        
        return None
    
    def _check_diagnosis_citation(self, agent_name: str, output: Any,
                                   all_outputs: Dict) -> Optional[Contradiction]:
        """
        检查用药Agent是否正确引用诊断信息
        Check if medication agent correctly cites diagnosis information
        
        示例矛盾:
        - T3诊断: 无糖尿病
        - T4用药推理: "因患者有糖尿病，推荐..."
        """
        if agent_name != "medication":
            return None
        
        diagnosis_output = all_outputs.get("diagnosis")
        if not diagnosis_output:
            return None
        
        # 获取诊断中的合并症
        comorbidities = diagnosis_output.content.get("comorbidities", [])
        comorbidities_text = " ".join(str(c) for c in comorbidities).lower()
        
        # 检查用药推理中的引用 (如果有reasoning字段)
        reasoning = str(output.reasoning).lower() if output.reasoning else ""
        
        # 示例：检查糖尿病引用
        if "糖尿病" in reasoning or "diabetes" in reasoning:
            if "糖尿病" not in comorbidities_text and "diabetes" not in comorbidities_text:
                return Contradiction(
                    rule_name="diagnosis_citation_accuracy",
                    description="用药推理中提及糖尿病，但诊断中未列出糖尿病",
                    upstream_agent="diagnosis",
                    downstream_agent="medication",
                    upstream_value=f"合并症: {comorbidities}",
                    downstream_value="推理中提及糖尿病",
                    query="用药Agent提及患者有糖尿病，但您的诊断中未列出。请确认合并症是否完整？",
                    severity="medium"
                )
        
        return None
    
    def _check_medication_completeness(self, agent_name: str, output: Any,
                                        all_outputs: Dict) -> Optional[Contradiction]:
        """
        检查报告是否完整引用用药信息
        Check if report completely cites medication recommendations
        
        示例矛盾:
        - T4用药: antiplatelet=["阿司匹林", "氯吡格雷"]
        - T5报告: 只提及阿司匹林
        """
        if agent_name != "report":
            return None
        
        medication_output = all_outputs.get("medication")
        if not medication_output:
            return None
        
        # 获取用药推荐
        med_content = medication_output.content
        recommended_drugs = []
        if med_content.get("antiplatelet"):
            recommended_drugs.extend(med_content["antiplatelet"] if isinstance(med_content["antiplatelet"], list) else [med_content["antiplatelet"]])
        if med_content.get("statin"):
            recommended_drugs.append(med_content["statin"])
        if med_content.get("acei_arb"):
            recommended_drugs.append(med_content["acei_arb"])
        if med_content.get("beta_blocker"):
            recommended_drugs.append(med_content["beta_blocker"])
        
        recommended_drugs = [d for d in recommended_drugs if d]  # 过滤空值
        
        if not recommended_drugs:
            return None
        
        # 检查报告中是否提及这些药物
        report_text = str(output.content.get("treatment_recommendations", ""))
        report_text += str(output.content.get("full_report", ""))
        
        missing_drugs = []
        for drug in recommended_drugs:
            if drug and drug not in report_text:
                missing_drugs.append(drug)
        
        if len(missing_drugs) > len(recommended_drugs) // 2:  # 超过一半的药物未提及
            return Contradiction(
                rule_name="medication_completeness",
                description=f"用药Agent推荐了{len(recommended_drugs)}种药物，但报告中遗漏了{len(missing_drugs)}种",
                upstream_agent="medication",
                downstream_agent="report",
                upstream_value=str(recommended_drugs),
                downstream_value=f"遗漏: {missing_drugs}",
                query=f"您推荐了以下药物{recommended_drugs}，但报告Agent遗漏了{missing_drugs}。请确认用药推荐是否需要调整？",
                severity="low"
            )
        
        return None
    
    # =========================================================================
    # LLM辅助检测 (检测隐含的逻辑不一致)
    # LLM-assisted Detection (Detect implicit logical inconsistencies)
    # =========================================================================
    
    def _llm_detect(self, agent_name: str, output: Any,
                    all_outputs: Dict) -> List[Contradiction]:
        """使用LLM检测隐含的逻辑不一致"""
        if not self.llm:
            return []
        
        outputs_text = self._format_all_outputs(all_outputs)
        current_text = self._format_output(agent_name, output)
        
        prompt = f"""作为逻辑审核专家，请检查以下Agent输出之间是否存在**逻辑不一致**。

**注意**：不要检查医学决策是否正确（如"LVEF降低是否应该用ACEI"），只检查：
1. 事实引用是否正确（下游是否正确引用了上游的信息）
2. 数值是否一致（同一数据在不同Agent中是否相同）
3. 分类是否一致（如病变范围描述是否一致）
4. 信息是否完整（下游是否遗漏了上游的关键信息）

## 当前Agent ({agent_name}) 的输出
{current_text}

## 上游Agent的输出
{outputs_text}

## 检查要点
1. 当前Agent是否正确引用了上游Agent的输出？
2. 是否存在数值或分类不一致？
3. 是否遗漏了上游的关键信息？

## 输出格式
如果发现逻辑不一致，请输出：
- 不一致类型：[事实引用错误/数值不一致/分类不一致/信息遗漏]
- 涉及Agent：[Agent名称]
- 描述：[具体描述]
- 上游值：[上游的值]
- 下游值：[下游的值]

如果没有发现逻辑不一致，请输出：无矛盾
"""
        
        response = self.llm.generate(prompt)
        
        contradictions = []
        if "无矛盾" not in response:
            if "不一致类型" in response:
                c = Contradiction(
                    rule_name="llm_detected_inconsistency",
                    description=response[:200],
                    upstream_agent=self._extract_agent_name(response),
                    downstream_agent=agent_name,
                    upstream_value=self._extract_value(response, "上游值"),
                    downstream_value=self._extract_value(response, "下游值"),
                    query=self._extract_query(response),
                    severity="medium"
                )
                contradictions.append(c)
        
        return contradictions
    
    # =========================================================================
    # 辅助方法 / Helper Methods
    # =========================================================================
    
    def _extract_vessel_count_from_text(self, text: str) -> Optional[int]:
        """从文本中提取病变血管数目"""
        if "单支" in text or "single-vessel" in text.lower():
            return 1
        if "双支" in text or "two-vessel" in text.lower():
            return 2
        if "三支" in text or "three-vessel" in text.lower():
            return 3
        return None
    
    def _extract_vessels_from_text(self, text: str) -> set:
        """从文本中提取血管名称"""
        vessels = set()
        vessel_patterns = {
            "LAD": ["LAD", "前降支", "左前降支"],
            "LCX": ["LCX", "回旋支", "左回旋支"],
            "RCA": ["RCA", "右冠", "右冠状动脉"],
            "LM": ["LM", "左主干"],
            "D1": ["D1", "对角支"],
            "OM": ["OM", "钝缘支"],
        }
        for vessel, patterns in vessel_patterns.items():
            if any(p in text for p in patterns):
                vessels.add(vessel)
        return vessels
    
    def _format_all_outputs(self, outputs: Dict) -> str:
        """格式化所有输出"""
        lines = []
        for name, output in outputs.items():
            lines.append(f"### {name}")
            if hasattr(output, 'content'):
                for k, v in output.content.items():
                    lines.append(f"  - {k}: {v}")
                lines.append(f"  - 置信度: {output.confidence:.2f}")
        return "\n".join(lines)
    
    def _format_output(self, name: str, output: Any) -> str:
        """格式化单个输出"""
        lines = [f"### {name}"]
        if hasattr(output, 'content'):
            for k, v in output.content.items():
                lines.append(f"  - {k}: {v}")
            lines.append(f"  - 置信度: {output.confidence:.2f}")
        return "\n".join(lines)
    
    def _extract_agent_name(self, response: str) -> str:
        """从响应中提取Agent名称"""
        if "cardiac" in response.lower() or "心功能" in response:
            return "cardiac_function"
        if "coronary" in response.lower() or "冠脉" in response:
            return "coronary"
        if "diagnosis" in response.lower() or "诊断" in response:
            return "diagnosis"
        if "medication" in response.lower() or "用药" in response:
            return "medication"
        return "unknown"
    
    def _extract_value(self, response: str, key: str) -> str:
        """从响应中提取值"""
        if key in response:
            parts = response.split(key)
            if len(parts) > 1:
                value = parts[1].split("\n")[0].strip(":：[] ")
                return value[:100]
        return ""
    
    def _extract_query(self, response: str) -> str:
        """从响应中提取质疑问题"""
        return "请确认您的输出与上游信息是否一致"
