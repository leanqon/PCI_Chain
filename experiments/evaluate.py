#!/usr/bin/env python3
"""
PCIChain 完整评测脚本
支持多种实验配置和基线对比
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from pci_chain import PCIChain
from pci_chain.chain import ChainResult
from pci_chain.utils import create_llm, MockLLM

# Import task-specific evaluation metrics
from .evaluate_metrics import (
    evaluate_coronary, evaluate_cardiac_function, evaluate_diagnosis,
    evaluate_medication, evaluate_report
)



@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str  # 实验名称
    llm_provider: str = "mock"  # LLM提供商
    llm_model: str = "gpt-4"  # 模型名称
    api_key: Optional[str] = None  # API密钥
    max_corrections: int = 3  # 最大修正次数
    enable_feedback: bool = True  # 启用反馈修正
    enable_llm_contradiction: bool = True  # 启用LLM矛盾检测
    enable_rule_contradiction: bool = True  # 启用规则矛盾检测
    num_samples: int = -1  # 样本数量 (-1表示全部)
    output_dir: str = "results"  # 输出目录
    language: str = "zh"  # 语言


@dataclass
class EvaluationMetrics:
    """评测指标"""
    # 基础统计
    total_patients: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # 矛盾检测
    total_contradictions: int = 0
    total_corrections: int = 0
    correction_success_rate: float = 0.0
    
    # 时间统计
    total_time: float = 0.0
    avg_time_per_patient: float = 0.0
    
    # Agent级别统计
    agent_confidences: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.agent_confidences is None:
            self.agent_confidences = {}


class PCIChainEvaluator:
    """PCIChain评测器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[Dict] = []
        self.metrics = EvaluationMetrics()
        
        # 创建输出目录
        self.output_dir = os.path.join(
            project_root, 
            config.output_dir, 
            config.name,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化PCIChain
        self.chain = self._create_chain()
    
    def _create_chain(self) -> PCIChain:
        """根据配置创建PCIChain"""
        # 创建LLM
        llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=self.config.api_key
        )
        
        return PCIChain(
            llm=llm,
            max_corrections=self.config.max_corrections if self.config.enable_feedback else 0,
            verbose=False,
            language=self.config.language
        )
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载评测数据"""
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # 限制样本数量
        if self.config.num_samples > 0:
            df = df.head(self.config.num_samples)
        
        print(f"Loaded {len(df)} patients from {data_path}")
        return df
    
    def run_evaluation(self, df: pd.DataFrame) -> EvaluationMetrics:
        """运行评测"""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.name}")
        print(f"Patients: {len(df)}")
        print(f"Feedback: {'Enabled' if self.config.enable_feedback else 'Disabled'}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for idx, row in df.iterrows():
            patient_info = row.to_dict()
            # Try to get patient_id from multiple possible fields
            patient_id = str(
                patient_info.get('Patient_ID') or 
                patient_info.get('序号') or 
                idx
            )
            
            try:
                # 运行PCIChain
                result = self.chain.run(patient_info, patient_id=patient_id)
                
                # 记录结果（包含输入数据）
                self._record_result(patient_id, result, success=True, patient_info=patient_info)
                self.metrics.successful_runs += 1
                
                # 打印进度
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} patients...")
                    
            except Exception as e:
                print(f"  Error processing patient {patient_id}: {e}")
                self._record_result(patient_id, None, success=False, error=str(e), patient_info=patient_info)
                self.metrics.failed_runs += 1
        
        # 计算总指标
        self.metrics.total_patients = len(df)
        self.metrics.total_time = time.time() - start_time
        self.metrics.avg_time_per_patient = (
            self.metrics.total_time / self.metrics.total_patients 
            if self.metrics.total_patients > 0 else 0
        )
        
        # 计算修正成功率
        if self.metrics.total_contradictions > 0:
            self.metrics.correction_success_rate = (
                self.metrics.total_corrections / self.metrics.total_contradictions
            )
        
        return self.metrics
    
    def _record_result(self, patient_id: str, result: Optional[ChainResult], 
                       success: bool, error: str = None, patient_info: Dict = None):
        """记录单个患者结果"""
        record = {
            "patient_id": patient_id,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存输入数据（排除敏感信息）
        if patient_info:
            # 敏感字段列表 - 这些不应保存到结果文件
            sensitive_fields = {
                'Patient_Name', 'patient_name', '姓名', '患者姓名',
                'Birth_Date', 'birth_date', '出生日期', 
                'ID_Number', 'id_number', '身份证号', '身份证',
                'Address', 'address', '地址', '住址',
                'Phone', 'phone', '电话', '联系电话',
                'Patient_ID',  # 也可以考虑隐藏
            }
            record["input"] = {
                k: str(v)[:500] if v else None 
                for k, v in patient_info.items() 
                if k not in sensitive_fields
            }
        
        if result and success:
            record["execution_time"] = result.execution_time
            record["contradictions"] = result.contradictions_detected
            record["corrections"] = result.corrections_made
            
            # 更新累计统计
            self.metrics.total_contradictions += result.contradictions_detected
            self.metrics.total_corrections += result.corrections_made
            
            # 记录Agent置信度
            for agent_name, output in result.outputs.items():
                if agent_name not in self.metrics.agent_confidences:
                    self.metrics.agent_confidences[agent_name] = []
                self.metrics.agent_confidences[agent_name].append(output.confidence)
            
            # 记录各Agent输出
            record["outputs"] = {
                name: {
                    "confidence": output.confidence,
                    "content": output.content
                }
                for name, output in result.outputs.items()
            }
        
        self.results.append(record)
        
        # 增量保存结果（每处理一个患者就保存一次）
        self._save_incremental_results()
    
    def _save_incremental_results(self):
        """增量保存结果到文件"""
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
    
    def compute_task_metrics(self, dataset_path: str) -> Dict:
        """计算任务级别的评估指标"""
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 标准化患者ID（处理浮点数格式如 '191596.0' -> '191596'）
        def normalize_patient_id(id_val):
            try:
                return str(int(float(id_val)))
            except (ValueError, TypeError):
                return str(id_val) if id_val else ""
        
        # 构建患者ID映射（使用标准化的ID）
        gt_map = {normalize_patient_id(p.get("Patient_ID", "") or p.get("序号", "")): p for p in dataset}
        
        # 收集匹配的预测和真实值
        coronary_data = []
        cardiac_data = []
        diagnosis_data = []
        medication_data = []
        report_data = []
        
        for result in self.results:
            if not result.get("success"):
                continue
            
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
        task_metrics = {}
        
        if coronary_data:
            task_metrics["T1_coronary"] = evaluate_coronary(*zip(*coronary_data))
        if cardiac_data:
            task_metrics["T2_cardiac_function"] = evaluate_cardiac_function(*zip(*cardiac_data))
        if diagnosis_data:
            task_metrics["T3_diagnosis"] = evaluate_diagnosis(*zip(*diagnosis_data))
        if medication_data:
            task_metrics["T4_medication"] = evaluate_medication(*zip(*medication_data))
        if report_data:
            task_metrics["T5_report"] = evaluate_report(*zip(*report_data))
        
        return task_metrics
    
    def save_results(self, dataset_path: str = None):
        """保存评测结果"""
        # 保存配置
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        # 保存详细结果
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存指标摘要
        metrics_dict = asdict(self.metrics)
        # 计算Agent平均置信度
        metrics_dict["avg_agent_confidences"] = {
            name: sum(confs) / len(confs) if confs else 0
            for name, confs in self.metrics.agent_confidences.items()
        }
        
        # 计算任务级别指标
        if dataset_path:
            metrics_dict["task_metrics"] = self.compute_task_metrics(dataset_path)
        
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return self.output_dir
    
    def print_summary(self, show_task_metrics: bool = True):
        """打印评测摘要"""
        print(f"\n{'='*60}")
        print(f"Evaluation Summary: {self.config.name}")
        print(f"{'='*60}")
        print(f"Total patients: {self.metrics.total_patients}")
        print(f"Successful: {self.metrics.successful_runs}")
        print(f"Failed: {self.metrics.failed_runs}")
        print(f"Total time: {self.metrics.total_time:.2f}s")
        print(f"Avg time/patient: {self.metrics.avg_time_per_patient:.2f}s")
        print(f"\nContradictions detected: {self.metrics.total_contradictions}")
        print(f"Corrections made: {self.metrics.total_corrections}")
        print(f"Correction success rate: {self.metrics.correction_success_rate:.2%}")
        
        print(f"\nAgent Average Confidence:")
        for name, confs in self.metrics.agent_confidences.items():
            avg = sum(confs) / len(confs) if confs else 0
            print(f"  {name}: {avg:.3f}")
        
        # 打印任务级别指标摘要
        if show_task_metrics:
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    full_metrics = json.load(f)
                
                task_metrics = full_metrics.get("task_metrics", {})
                if task_metrics:
                    print(f"\n{'-'*60}")
                    print("Task-Level Metrics:")
                    print(f"{'-'*60}")
                    
                    # T1: 冠脉识别
                    t1 = task_metrics.get("T1_coronary", {})
                    if t1:
                        print(f"\n[T1: Coronary] Accuracy: {t1.get('accuracy', 0):.4f}  " 
                              f"F1: {t1.get('f1', 0):.4f}")
                    
                    # T2: 心功能
                    t2 = task_metrics.get("T2_cardiac_function", {})
                    if t2:
                        print(f"[T2: Cardiac] Grade Acc: {t2.get('grade_accuracy', 0):.4f}")
                    
                    # T3: 诊断
                    t3 = task_metrics.get("T3_diagnosis", {})
                    if t3:
                        print(f"[T3: Diagnosis] Similarity: {t3.get('text_similarity', 0):.4f}")
                    
                    # T4: 用药
                    t4 = task_metrics.get("T4_medication", {})
                    if t4 and "overall" in t4:
                        print(f"[T4: Medication] Category Acc: {t4['overall'].get('category_accuracy', 0):.4f}")
                    
                    # T5: 报告
                    t5 = task_metrics.get("T5_report", {})
                    if t5:
                        print(f"[T5: Report] Completeness: {t5.get('completeness', 0):.4f}")
        
        print(f"{'='*60}\n")



def run_baseline_comparison(data_path: str, num_samples: int = 10):
    """运行基线对比实验"""
    
    experiments = [
        # 基线1：无反馈修正
        ExperimentConfig(
            name="baseline_no_feedback",
            enable_feedback=False,
            num_samples=num_samples
        ),
        # 基线2：仅规则检测
        ExperimentConfig(
            name="pcichain_rules_only",
            enable_feedback=True,
            enable_llm_contradiction=False,
            enable_rule_contradiction=True,
            num_samples=num_samples
        ),
        # 完整PCIChain
        ExperimentConfig(
            name="pcichain_full",
            enable_feedback=True,
            enable_llm_contradiction=True,
            enable_rule_contradiction=True,
            num_samples=num_samples
        ),
    ]
    
    all_results = {}
    
    for config in experiments:
        print(f"\n\n{'#'*60}")
        print(f"Running: {config.name}")
        print(f"{'#'*60}")
        
        evaluator = PCIChainEvaluator(config)
        df = evaluator.load_data(data_path)
        evaluator.run_evaluation(df)
        evaluator.save_results()
        evaluator.print_summary()
        
        all_results[config.name] = {
            "contradictions": evaluator.metrics.total_contradictions,
            "corrections": evaluator.metrics.total_corrections,
            "avg_time": evaluator.metrics.avg_time_per_patient,
            "success_rate": evaluator.metrics.successful_runs / evaluator.metrics.total_patients
        }
    
    # 打印对比摘要
    print(f"\n\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Contradictions':>15} {'Corrections':>12} {'Avg Time':>10}")
    print("-" * 60)
    for name, stats in all_results.items():
        print(f"{name:<25} {stats['contradictions']:>15} {stats['corrections']:>12} {stats['avg_time']:>10.2f}s")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="PCIChain Evaluation")
    parser.add_argument("--data", type=str, 
                        default="/Users/dsr/Desktop/paper/1213_PCI/1213_数据处理/融合数据集_OCR增强版.csv",
                        help="Path to evaluation data")
    parser.add_argument("--mode", choices=["single", "baseline", "full"], default="single",
                        help="Evaluation mode: single/baseline/full")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--provider", default="mock",
                        help="LLM provider: mock/openai/zhipu/deepseek")
    parser.add_argument("--model", default="gpt-4",
                        help="Model name")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for LLM provider")
    parser.add_argument("--language", choices=["en", "zh"], default="zh",
                        help="Output language: en/zh")
    parser.add_argument("--no-feedback", action="store_true",
                        help="Disable feedback correction")
    
    args = parser.parse_args()
    
    if args.mode == "baseline":
        run_baseline_comparison(args.data, args.samples)
    else:
        config = ExperimentConfig(
            name=f"eval_{args.provider}_{args.model}",
            llm_provider=args.provider,
            llm_model=args.model,
            api_key=args.api_key,
            language=args.language,
            enable_feedback=not args.no_feedback,
            num_samples=args.samples
        )
        
        evaluator = PCIChainEvaluator(config)
        df = evaluator.load_data(args.data)
        evaluator.run_evaluation(df)
        evaluator.save_results(dataset_path=args.data)
        evaluator.print_summary()




if __name__ == "__main__":
    main()
