#!/usr/bin/env python3
"""
PCIChain 完整评估脚本 - 中英文双语版
使用 DeepSeek API 运行所有数据的评估
"""

import sys
import os
import json
import warnings
from datetime import datetime

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 添加项目路径
sys.path.insert(0, '/Users/dsr/Desktop/paper/1213_PCI')

# Mock RAG 模块以避免 PDF 解析问题
class MockRetriever:
    def retrieve_for_medication(self, **kwargs):
        return ""
    def retrieve(self, query, top_k=3):
        return []

import pci_chain.rag as rag_module
rag_module.get_retriever = lambda: MockRetriever()

from pci_chain.experiments.evaluate import PCIChainEvaluator, ExperimentConfig

# API 配置
API_KEY = "sk-d13415b66423aa8c14edcfe54a35f0f4603ad768a4c33d9ed73a9c94553e597e"
MODEL = "deepseek/deepseek-v3.2-251201"
BASE_URL = "https://api.qnaigc.com/v1"

# 数据集路径
DATASET_ZH = "/Users/dsr/Desktop/paper/1213_PCI/患者级别数据集_标准版.json"
DATASET_EN = "/Users/dsr/Desktop/paper/1213_PCI/患者级别数据集_标准版_EN.json"

# 输出目录
OUTPUT_DIR = "/Users/dsr/Desktop/paper/1213_PCI/results/full_evaluation"


def run_evaluation(dataset_path: str, language: str, num_samples: int = -1):
    """运行单语言评估"""
    lang_name = "Chinese" if language == "zh" else "English"
    print(f"\n{'='*70}")
    print(f"Running {lang_name} Evaluation")
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Samples: {num_samples if num_samples > 0 else 'ALL'}")
    print(f"{'='*70}")
    
    config = ExperimentConfig(
        name=f"deepseek_{language}",
        llm_provider="deepseek",
        llm_model=MODEL,
        api_key=API_KEY,
        language=language,
        enable_feedback=True,
        enable_llm_contradiction=True,
        enable_rule_contradiction=True,
        max_corrections=2,
        num_samples=num_samples,
        output_dir=OUTPUT_DIR
    )
    
    evaluator = PCIChainEvaluator(config)
    df = evaluator.load_data(dataset_path)
    evaluator.run_evaluation(df)
    evaluator.save_results(dataset_path=dataset_path)
    evaluator.print_summary()
    
    return evaluator.output_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PCIChain Full Evaluation")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples (-1 for all)")
    parser.add_argument("--language", choices=["zh", "en", "both"], default="both", help="Language to evaluate")
    args = parser.parse_args()
    
    print("="*70)
    print("PCIChain Full Evaluation - DeepSeek API")
    print(f"Model: {MODEL}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    if args.language in ["zh", "both"]:
        zh_output = run_evaluation(DATASET_ZH, "zh", args.samples)
        results["zh"] = zh_output
        print(f"\n中文评估结果保存至: {zh_output}")
    
    if args.language in ["en", "both"]:
        en_output = run_evaluation(DATASET_EN, "en", args.samples)
        results["en"] = en_output
        print(f"\nEnglish evaluation results saved to: {en_output}")
    
    # 保存总结
    summary_file = os.path.join(OUTPUT_DIR, f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "samples": args.samples,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估总结保存至: {summary_file}")
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
