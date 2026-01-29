#!/usr/bin/env python3
"""
PCIChain 完整评估脚本
为所有5个任务提供评估指标，比较模型输出与金标准
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

# ============================================================================
# 药物标准化映射
# ============================================================================

ANTIPLATELET_DRUGS = {
    "阿司匹林": ["阿司匹林", "拜阿司匹灵", "aspirin", "拜耳"],
    "氯吡格雷": ["氯吡格雷", "波立维", "泰嘉", "clopidogrel", "plavix"],
    "替格瑞洛": ["替格瑞洛", "倍林达", "ticagrelor", "brilinta"],
    "普拉格雷": ["普拉格雷", "prasugrel"],
}

STATIN_DRUGS = {
    "阿托伐他汀": ["阿托伐他汀", "立普妥", "美达信", "atorvastatin", "lipitor"],
    "瑞舒伐他汀": ["瑞舒伐他汀", "可定", "rosuvastatin", "crestor"],
    "匹伐他汀": ["匹伐他汀", "pitavastatin"],
    "辛伐他汀": ["辛伐他汀", "舒降之", "simvastatin"],
}

ACEI_DRUGS = {
    "培哚普利": ["培哚普利", "雅施达", "perindopril"],
    "贝那普利": ["贝那普利", "洛汀新", "benazepril"],
    "依那普利": ["依那普利", "enalapril"],
    "卡托普利": ["卡托普利", "captopril"],
    "福辛普利": ["福辛普利", "fosinopril"],
    "雷米普利": ["雷米普利", "ramipril"],
}

ARB_DRUGS = {
    "缬沙坦": ["缬沙坦", "代文", "valsartan", "diovan"],
    "厄贝沙坦": ["厄贝沙坦", "安博维", "依伦平", "irbesartan"],
    "氯沙坦": ["氯沙坦", "科素亚", "losartan"],
    "坎地沙坦": ["坎地沙坦", "candesartan"],
    "替米沙坦": ["替米沙坦", "美卡素", "telmisartan"],
    "奥美沙坦": ["奥美沙坦", "olmesartan"],
}

ACEI_ARB_DRUGS = {**ACEI_DRUGS, **ARB_DRUGS}
ACEI_ARB_DRUGS["沙库巴曲缬沙坦"] = ["沙库巴曲缬沙坦", "诺欣妥", "sacubitril", "entresto"]

BETA_BLOCKER_DRUGS = {
    "美托洛尔": ["美托洛尔", "倍他乐克", "metoprolol", "betaloc"],
    "比索洛尔": ["比索洛尔", "康忻", "bisoprolol", "concor"],
    "卡维地洛": ["卡维地洛", "carvedilol"],
    "阿替洛尔": ["阿替洛尔", "atenolol"],
}

VESSEL_MAP = {
    "LAD": ["LAD", "前降支", "left anterior descending"],
    "LCX": ["LCX", "回旋支", "left circumflex"],
    "RCA": ["RCA", "右冠", "right coronary"],
    "LM": ["LM", "左主干", "left main"],
    "D1": ["D1", "第一对角支", "first diagonal"],
    "D2": ["D2", "第二对角支", "second diagonal"],
    "OM": ["OM", "钝缘支", "obtuse marginal"],
}


def normalize_drug(drug_text: str, drug_map: Dict[str, List[str]]) -> str:
    """将药物名称标准化"""
    if not drug_text:
        return ""
    drug_text_lower = drug_text.lower()
    for standard_name, variants in drug_map.items():
        for variant in variants:
            if variant.lower() in drug_text_lower:
                return standard_name
    return ""


def extract_drugs_from_text(text: str, drug_map: Dict[str, List[str]]) -> List[str]:
    """从文本中提取所有匹配的药物"""
    if not text:
        return []
    drugs = set()
    text_lower = text.lower()
    for standard_name, variants in drug_map.items():
        for variant in variants:
            if variant.lower() in text_lower:
                drugs.add(standard_name)
                break
    return list(drugs)


def normalize_vessels(text: str) -> Set[str]:
    """标准化血管名称"""
    if not text:
        return set()
    vessels = set()
    text_lower = text.lower()
    for standard, variants in VESSEL_MAP.items():
        for variant in variants:
            if variant.lower() in text_lower:
                vessels.add(standard)
                break
    return vessels


def text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（0-1）"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def extract_lvef_value(text: str) -> int:
    """从文本中提取LVEF数值"""
    if not text:
        return -1
    # 匹配LVEF或EF后的数字
    patterns = [
        r'(?:LVEF|EF|射血分数)[^\d]*(\d+)',
        r'(\d+)\s*[%％]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = int(match.group(1))
            if 10 <= val <= 99:  # 合理范围
                return val
    return -1


def categorize_lvef(lvef: int) -> str:
    """LVEF分级"""
    if lvef < 0:
        return "unknown"
    elif lvef >= 50:
        return "normal"
    elif lvef >= 40:
        return "mildly_reduced"
    elif lvef >= 30:
        return "moderately_reduced"
    else:
        return "severely_reduced"


# ============================================================================
# Task 1: 冠脉识别评估
# ============================================================================

def evaluate_coronary(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    评估冠脉识别任务
    
    指标：
    - 血管识别准确率（完全匹配）
    - 血管识别Precision/Recall/F1
    - 主要血管识别准确率
    """
    exact_match = 0
    total = 0
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Handle different output structures
        pred_content = pred.get("content", pred)  # May be directly in pred
        
        # Extract vessels from different possible formats
        pred_vessels = set()
        
        # Format 1: English - {"vessels": [{"name": "LAD"}...]}
        if "vessels" in pred_content:
            for v in pred_content.get("vessels", []):
                if isinstance(v, dict):
                    name = v.get("name", "").upper()
                else:
                    name = str(v).upper()
                if name:
                    pred_vessels.add(name)
        
        # Format 2: English - {"lesion_findings": [{"vessel": "LAD", "intervention_performed": true}...]}
        elif "lesion_findings" in pred_content:
            for v in pred_content.get("lesion_findings", []):
                if isinstance(v, dict) and v.get("intervention_performed", False):
                    name = v.get("vessel", "").upper()
                    if name:
                        pred_vessels.add(name)
        
        # Format 3: Chinese - {"受累血管": [{"血管": "LAD"}...], "干预方式": {"靶血管": "LAD"}}
        elif "受累血管" in pred_content:
            # Extract from 受累血管 list
            for v in pred_content.get("受累血管", []):
                if isinstance(v, dict):
                    vessel_name = v.get("血管", "").upper()
                    stenosis = v.get("狭窄程度", "")
                    # Only include if significant stenosis (>=50% or intervention performed)
                    if vessel_name and ("%" in stenosis or "重度" in stenosis or "中度" in stenosis):
                        # Check if >=50%
                        import re
                        percent_match = re.search(r'(\d+)%', stenosis)
                        if percent_match:
                            percent = int(percent_match.group(1))
                            if percent >= 50:
                                pred_vessels.add(vessel_name)
                        elif "重度" in stenosis or "中度" in stenosis:
                            pred_vessels.add(vessel_name)
            
            # Also extract from 干预方式/靶血管 (intervention target - most important)
            intervention = pred_content.get("干预方式", {})
            if isinstance(intervention, dict):
                target_vessel = intervention.get("靶血管", "").upper()
                if target_vessel:
                    pred_vessels.add(target_vessel)
        
        # Extract ground truth vessels
        gt_vessel_text = gt.get("Culprit_Vessel", "") or gt.get("PCI_Procedure_Type", "")
        gt_vessels = normalize_vessels(gt_vessel_text)
        
        # 完全匹配
        if pred_vessels == gt_vessels:
            exact_match += 1
        
        # F1计算
        tp = len(pred_vessels & gt_vessels)
        fp = len(pred_vessels - gt_vessels)
        fn = len(gt_vessels - pred_vessels)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        total += 1
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(exact_match / total, 4) if total > 0 else 0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
        "total": total,
    }



# ============================================================================
# Task 2: 心功能评估
# ============================================================================

def evaluate_cardiac_function(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    评估心功能任务
    
    指标：
    - LVEF分级准确率
    - LVEF数值误差（MAE）
    """
    grade_correct = 0
    value_errors = []
    total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Handle different output structures
        pred_content = pred.get("content", pred)  # May be directly in pred
        pred_range = pred_content.get("lvef_range", "") or pred_content.get("lvef_grade", "") or pred_content.get("lvef_value", "")
        
        # 从金标准中提取LVEF
        gt_echo = gt.get("Pre_PCI_Echo_Text", "") or ""
        gt_lvef = gt.get("Pre_LVEF", "")
        
        # 尝试解析
        if gt_lvef and str(gt_lvef) not in ["", "nan", "None"]:
            try:
                lvef_val = int(float(str(gt_lvef).replace("%", "").strip()))
            except:
                lvef_val = extract_lvef_value(gt_echo)
        else:
            lvef_val = extract_lvef_value(gt_echo)
        
        if lvef_val < 0:
            continue  # 无法评估
        
        gt_grade = categorize_lvef(lvef_val)
        
        # 从预测中提取分级
        pred_grade = "unknown"
        pred_range_lower = pred_range.lower()
        if "正常" in pred_range or "normal" in pred_range_lower or "50" in pred_range or "60" in pred_range:
            pred_grade = "normal"
        elif "轻度" in pred_range or "mild" in pred_range_lower or "40" in pred_range:
            pred_grade = "mildly_reduced"
        elif "中度" in pred_range or "moderate" in pred_range_lower or "30" in pred_range:
            pred_grade = "moderately_reduced"
        elif "重度" in pred_range or "severe" in pred_range_lower:
            pred_grade = "severely_reduced"
        
        if pred_grade == gt_grade:
            grade_correct += 1
        
        # 尝试提取预测值
        pred_val = extract_lvef_value(pred_range)
        if pred_val >= 0:
            value_errors.append(abs(pred_val - lvef_val))
        
        total += 1
    
    mae = sum(value_errors) / len(value_errors) if value_errors else -1
    
    return {
        "grade_accuracy": round(grade_correct / total, 4) if total > 0 else 0,
        "mae": round(mae, 2) if mae >= 0 else None,
        "grade_correct": grade_correct,
        "value_samples": len(value_errors),
        "total": total,
    }


# ============================================================================
# Task 3: 诊断评估
# ============================================================================

def evaluate_diagnosis(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    评估诊断任务
    
    指标：
    - 主诊断相似度（文本相似度）
    - 关键诊断关键词覆盖率
    """
    similarities = []
    keyword_recalls = []
    
    # 中英文关键词映射组 (每组包含同义词)
    keyword_groups = [
        {"心肌梗死", "myocardial infarction", "MI"},
        {"STEMI", "ST段抬高", "st-elevation"},
        {"NSTEMI", "非ST段抬高", "non-st-elevation"},
        {"心绞痛", "angina"},
        {"不稳定", "unstable"},
        {"稳定", "stable"},
        {"冠心病", "coronary artery disease", "CAD"},
        {"冠状动脉", "coronary"},
        {"高血压", "hypertension"},
        {"糖尿病", "diabetes"},
        {"心力衰竭", "heart failure"},
        {"心衰", "HF"},
        {"房颤", "atrial fibrillation", "AF"},
        {"支架", "stent"},
        {"PCI", "经皮冠状动脉介入"},
        {"术后", "post-procedure", "after"},
        {"单支", "single-vessel", "single vessel"},
        {"双支", "two-vessel", "double-vessel"},
        {"三支", "three-vessel", "triple-vessel"},
        {"多支", "multi-vessel", "multiple vessel"},
        {"急性", "acute"},
    ]
    
    for pred, gt in zip(predictions, ground_truths):
        # Handle different output structures
        pred_content = pred.get("content", pred)
        
        # Extract diagnosis from different possible formats (FIXED: prioritize raw_text)
        pred_diagnosis = ""
        if "raw_text" in pred_content:  # Priority 1: raw_text (most common)
            pred_diagnosis = pred_content.get("raw_text", "")
        elif "main_diagnosis" in pred_content:  # Priority 2: main_diagnosis
            pred_diagnosis = pred_content.get("main_diagnosis", "")
        elif "primary_diagnosis" in pred_content:  # Priority 3: primary_diagnosis
            pd = pred_content.get("primary_diagnosis", {})
            pred_diagnosis = pd.get("name", "") if isinstance(pd, dict) else str(pd)
        else:  # Fallback: convert entire content to string
            pred_diagnosis = str(pred_content)
        
        gt_diagnosis = gt.get("Discharge_Diagnosis_Western", "") or ""
        
        # 文本相似度
        sim = text_similarity(pred_diagnosis.lower(), gt_diagnosis.lower())
        similarities.append(sim)
        
        # 关键词覆盖率 (使用同义词组)
        gt_lower = gt_diagnosis.lower()
        pred_lower = pred_diagnosis.lower()
        
        # Find which keyword groups appear in GT
        gt_groups = [group for group in keyword_groups if any(kw.lower() in gt_lower for kw in group)]
        
        if gt_groups:
            # Check how many GT groups are covered in prediction
            pred_covered = sum(1 for group in gt_groups if any(kw.lower() in pred_lower for kw in group))
            keyword_recalls.append(pred_covered / len(gt_groups))
    
    return {
        "text_similarity": round(sum(similarities) / len(similarities), 4) if similarities else 0,
        "keyword_recall": round(sum(keyword_recalls) / len(keyword_recalls), 4) if keyword_recalls else 0,
        "total": len(similarities),
    }


# ============================================================================
# Task 4: 用药推荐评估
# ============================================================================

def evaluate_medication(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    评估用药推荐任务
    
    指标：
    - 各类药物Precision/Recall/F1/Accuracy
    - 药物类别覆盖率（推荐了应该推荐的类别）
    - 不合理推荐率（推荐了不应该推荐的药物）
    """
    results = {
        "antiplatelet": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "statin": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "acei_arb": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "beta_blocker": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
    }
    
    category_correct = 0
    total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Handle different output structures
        pred_content = pred.get("content", pred)
        
        # Extract drugs from different possible formats
        pred_text = ""
        if isinstance(pred_content, str):
            pred_text = pred_content
        elif "raw_text" in pred_content:
            pred_text = pred_content.get("raw_text", "")
        elif "recommendations" in pred_content:
            # Format: {"recommendations": [{"drug_class": "...", "drug_name": "..."}]}
            recs = pred_content.get("recommendations", [])
            pred_text = " ".join([
                f"{r.get('drug_class', '')} {r.get('drug_name', '')}"
                for r in recs if isinstance(r, dict)
            ])
        else:
            pred_text = str(pred_content)
        
        pred_text_lower = pred_text.lower()
        
        # 抗血小板 - check for common antiplatelet mentions
        pred_ap = any(kw in pred_text_lower for kw in ["aspirin", "clopidogrel", "ticagrelor", "阿司匹林", "氯吡格雷", "替格瑞洛", "antiplatelet", "dapt"])
        gt_ap = bool(gt.get("Discharge_Antiplatelets", ""))
        _update_binary_confusion(results["antiplatelet"], pred_ap, gt_ap)
        
        # 他汀 - check for statin mentions
        pred_statin = any(kw in pred_text_lower for kw in ["statin", "atorvastatin", "rosuvastatin", "pitavastatin", "他汀", "阿托伐他汀", "瑞舒伐他汀"])
        gt_statin = bool(gt.get("Discharge_Statin", ""))
        _update_binary_confusion(results["statin"], pred_statin, gt_statin)
        
        # ACEI/ARB - check for ACEI/ARB mentions
        pred_acei = any(kw in pred_text_lower for kw in ["acei", "arb", "perindopril", "irbesartan", "valsartan", "普利", "沙坦", "培哚普利", "厄贝沙坦"])
        gt_acei = bool(gt.get("Discharge_ACEI_ARB", ""))
        _update_binary_confusion(results["acei_arb"], pred_acei, gt_acei)
        
        # β受体阻滞剂 - check for beta blocker mentions
        pred_bb = any(kw in pred_text_lower for kw in ["beta-blocker", "beta blocker", "metoprolol", "bisoprolol", "洛尔", "美托洛尔", "比索洛尔"])
        gt_bb = bool(gt.get("Discharge_Beta_Blocker", ""))
        _update_binary_confusion(results["beta_blocker"], pred_bb, gt_bb)
        
        # 类别全部正确
        if (pred_statin == gt_statin) and (pred_acei == gt_acei) and (pred_bb == gt_bb) and (pred_ap == gt_ap):
            category_correct += 1
        
        total += 1
    
    # 计算各类指标
    metrics = {}
    for drug_type, counts in results.items():
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics[drug_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    
    metrics["overall"] = {
        "category_accuracy": round(category_correct / total, 4) if total > 0 else 0,
        "total": total,
    }
    
    return metrics


def _update_confusion(result: Dict, pred_set: Set, gt_set: Set):
    """更新集合形式的混淆矩阵"""
    result["TP"] += len(pred_set & gt_set)
    result["FP"] += len(pred_set - gt_set)
    result["FN"] += len(gt_set - pred_set)


def _update_binary_confusion(result: Dict, pred: bool, gt: bool):
    """更新二元混淆矩阵"""
    if pred and gt:
        result["TP"] += 1
    elif pred and not gt:
        result["FP"] += 1
    elif not pred and gt:
        result["FN"] += 1
    else:
        result["TN"] += 1


# ============================================================================
# Task 5: 报告生成评估
# ============================================================================

def evaluate_report(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    评估报告生成任务
    
    指标：
    - 报告完整性（包含所有必需部分）
    - 报告与诊断一致性
    - 平均置信度
    """
    completeness_scores = []
    consistency_scores = []
    confidences = []
    
    # Report sections to check (as text headers in Markdown)
    required_section_patterns = [
        r"patient\s*summary|患者摘要",
        r"specialist\s*consensus|专家共识",
        r"integrated\s*diagnosis|diagnosis|综合诊断|诊断",
        r"treatment|recommendations|治疗建议|用药",
        r"follow.?up|随访",
        r"conclusion|结论"
    ]
    
    # Cross-language medical term mapping (Chinese -> English equivalents)
    term_translation_map = {
        # Cardiovascular conditions
        "心肌梗死": ["myocardial infarction", "MI", "heart attack"],
        "心绞痛": ["angina", "angina pectoris"],
        "不稳定": ["unstable"],
        "稳定": ["stable"],
        "急性": ["acute"],
        "冠心病": ["coronary artery disease", "CAD", "coronary heart disease"],
        "冠状动脉": ["coronary artery", "coronary"],
        "心力衰竭": ["heart failure", "cardiac failure", "HF"],
        "心衰": ["heart failure", "HF"],
        
        # Risk factors
        "高血压": ["hypertension", "high blood pressure"],
        "糖尿病": ["diabetes", "diabetes mellitus"],
        "血脂": ["lipid", "cholesterol"],
        
        # Cardiac rhythm
        "房颤": ["atrial fibrillation", "AF", "Afib"],
        "房室传导阻滞": ["atrioventricular block", "AV block"],
        
        # Procedures
        "PCI": ["PCI", "percutaneous coronary intervention", "angioplasty"],
        "支架": ["stent", "stenting"],
        "植入": ["implant", "implantation", "placement"],
        
        # Vessels
        "单支": ["single-vessel", "single vessel"],
        "多支": ["multi-vessel", "multiple vessel", "multivessel"],
        "前降支": ["LAD", "left anterior descending"],
        "右冠": ["RCA", "right coronary"],
        "回旋支": ["LCX", "left circumflex"],
        
        # Other conditions
        "反流性食管炎": ["reflux esophagitis", "GERD", "gastroesophageal reflux"],
        "胃炎": ["gastritis"],
        "甲状腺": ["thyroid"],
        "结节": ["nodule", "node"],
    }
    
    for pred, gt in zip(predictions, ground_truths):
        pred_content = pred.get("content", {})
        
        # Handle both string (Markdown) and dict content
        if isinstance(pred_content, str):
            report_text = pred_content.lower()
        elif isinstance(pred_content, dict):
            # Extract from 'content' key if nested
            if 'content' in pred_content:
                report_text = str(pred_content['content']).lower()
            else:
                report_text = json.dumps(pred_content, ensure_ascii=False).lower()
        else:
            report_text = str(pred_content).lower()
        
        # 完整性（包含多少必需部分）
        import re
        present = sum(1 for pattern in required_section_patterns if re.search(pattern, report_text, re.IGNORECASE))
        completeness_scores.append(present / len(required_section_patterns))
        
        # 一致性（报告中是否包含诊断关键术语）
        gt_diagnosis = gt.get("Discharge_Diagnosis_Western", "") or ""
        
        # FIXED: Use number-based splitting instead of comma-based
        # GT format: "1.不稳定型心绞痛2.高血压3.反流性食管炎"
        diagnosis_parts = re.split(r'\d+\.', gt_diagnosis)
        # Extract meaningful terms (length > 3, take first 50 chars to avoid overly long terms)
        diagnosis_terms = [term.strip()[:50].lower() for term in diagnosis_parts if len(term.strip()) > 3][:5]
        
        # Also extract key medical terms using keyword matching
        key_medical_terms = [
            "心肌梗死", "心绞痛", "高血压", "糖尿病", "心衰", "房颤",
            "myocardial infarction", "angina", "hypertension", "diabetes", "heart failure"
        ]
        gt_medical_keywords = [term for term in key_medical_terms if term in gt_diagnosis.lower()]
        
        # Combine both approaches
        all_terms = diagnosis_terms + gt_medical_keywords
        
        if all_terms:
            matches = 0
            for term in all_terms:
                term_lower = term.lower()
                # Direct match (for English terms or exact Chinese match)
                if term_lower in report_text:
                    matches += 1
                # Cross-language match (translate Chinese to English)
                else:
                    # Check if term contains any Chinese characters
                    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in term)
                    if has_chinese:
                        # Try to find Chinese keywords in the term and translate
                        matched = False
                        for cn_keyword, en_equivalents in term_translation_map.items():
                            if cn_keyword in term_lower:
                                # Check if any English equivalent appears in report
                                if any(en_term.lower() in report_text for en_term in en_equivalents):
                                    matched = True
                                    break
                        if matched:
                            matches += 1
            
            consistency = matches / len(all_terms)
        else:
            consistency = 0.5  # Default if no terms found
        
        consistency_scores.append(consistency)
        
        # 置信度
        conf = pred.get("confidence", 0)
        if isinstance(conf, (int, float)):
            confidences.append(conf)
        else:
            confidences.append(0.7)  # Default
    
    return {
        "completeness": round(sum(completeness_scores) / len(completeness_scores), 4) if completeness_scores else 0,
        "diagnosis_consistency": round(sum(consistency_scores) / len(consistency_scores), 4) if consistency_scores else 0,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
        "total": len(predictions),
    }


# ============================================================================
# 主评估函数
# ============================================================================

def run_full_evaluation(results_file: str, dataset_file: str, output_file: str = None) -> Dict:
    """
    运行完整评估
    """
    # 加载数据
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    if isinstance(results_data, dict):
        results = [results_data]
    else:
        results = results_data
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 标准化患者ID（处理浮点数格式如 '191596.0' -> '191596'）
    def normalize_patient_id(id_val):
        try:
            return str(int(float(id_val)))
        except (ValueError, TypeError):
            return str(id_val) if id_val else ""
    
    # 构建患者ID映射（使用标准化的ID）
    gt_map = {normalize_patient_id(p.get("Patient_ID", "")): p for p in dataset}
    
    # 收集匹配的预测和真实值
    coronary_data = []
    cardiac_data = []
    diagnosis_data = []
    medication_data = []
    report_data = []
    
    for result in results:
        patient_id = normalize_patient_id(result.get("patient_id", ""))
        if patient_id not in gt_map:
            continue
        
        gt = gt_map[patient_id]
        outputs = result.get("outputs", {})
        
        if "coronary" in outputs:
            coronary_data.append((outputs["coronary"], gt))
        if "cardiac_function" in outputs:
            cardiac_data.append((outputs["cardiac_function"], gt))
        if "diagnosis" in outputs:
            diagnosis_data.append((outputs["diagnosis"], gt))
        if "medication" in outputs:
            medication_data.append((outputs["medication"], gt))
        if "report" in outputs:
            report_data.append((outputs["report"], gt))
    
    # 评估各任务
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "matched_patients": len(coronary_data),
        "task_metrics": {
            "T1_coronary": evaluate_coronary(*zip(*coronary_data)) if coronary_data else {},
            "T2_cardiac_function": evaluate_cardiac_function(*zip(*cardiac_data)) if cardiac_data else {},
            "T3_diagnosis": evaluate_diagnosis(*zip(*diagnosis_data)) if diagnosis_data else {},
            "T4_medication": evaluate_medication(*zip(*medication_data)) if medication_data else {},
            "T5_report": evaluate_report(*zip(*report_data)) if report_data else {},
        }
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存到: {output_file}")
    
    return evaluation


def print_full_report(evaluation: Dict):
    """打印完整评估报告"""
    print("\n" + "=" * 70)
    print("PCIChain 完整评估报告")
    print("=" * 70)
    
    print(f"\n总结果数: {evaluation['total_results']}")
    print(f"匹配患者数: {evaluation['matched_patients']}")
    
    tm = evaluation["task_metrics"]
    
    # T1: 冠脉识别
    print("\n" + "-" * 70)
    print("【T1: 冠脉识别】")
    t1 = tm.get("T1_coronary", {})
    if t1:
        print(f"  血管识别准确率: {t1.get('accuracy', 0):.4f}")
        print(f"  Precision: {t1.get('precision', 0):.4f}  Recall: {t1.get('recall', 0):.4f}  F1: {t1.get('f1', 0):.4f}")
    
    # T2: 心功能
    print("\n" + "-" * 70)
    print("【T2: 心功能评估】")
    t2 = tm.get("T2_cardiac_function", {})
    if t2:
        print(f"  LVEF分级准确率: {t2.get('grade_accuracy', 0):.4f}")
        if t2.get("mae") is not None:
            print(f"  LVEF数值MAE: {t2.get('mae'):.2f}%")
    
    # T3: 诊断
    print("\n" + "-" * 70)
    print("【T3: 诊断评估】")
    t3 = tm.get("T3_diagnosis", {})
    if t3:
        print(f"  诊断文本相似度: {t3.get('text_similarity', 0):.4f}")
        print(f"  关键词覆盖率: {t3.get('keyword_recall', 0):.4f}")
    
    # T4: 用药
    print("\n" + "-" * 70)
    print("【T4: 用药推荐】")
    t4 = tm.get("T4_medication", {})
    if t4:
        print(f"  {'类别':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
        for drug_type in ["antiplatelet", "statin", "acei_arb", "beta_blocker"]:
            m = t4.get(drug_type, {})
            print(f"  {drug_type:<15} {m.get('precision', 0):<10.4f} {m.get('recall', 0):<10.4f} {m.get('f1', 0):<10.4f} {m.get('accuracy', 0):<10.4f}")
        print(f"  整体类别准确率: {t4.get('overall', {}).get('category_accuracy', 0):.4f}")
    
    # T5: 报告
    print("\n" + "-" * 70)
    print("【T5: 报告生成】")
    t5 = tm.get("T5_report", {})
    if t5:
        print(f"  报告完整性: {t5.get('completeness', 0):.4f}")
        print(f"  诊断一致性: {t5.get('diagnosis_consistency', 0):.4f}")
        print(f"  平均置信度: {t5.get('avg_confidence', 0):.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PCIChain完整评估脚本")
    parser.add_argument("--results", required=True, help="PCIChain结果文件")
    parser.add_argument("--dataset", default="/Users/dsr/Desktop/paper/1213_PCI/患者级别数据集_标准版.json",
                       help="金标准数据集文件")
    parser.add_argument("--output", help="评估结果输出文件")
    
    args = parser.parse_args()
    
    evaluation = run_full_evaluation(args.results, args.dataset, args.output)
    print_full_report(evaluation)
