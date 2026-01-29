#!/usr/bin/env python3
"""
PCIChain 全流程测试脚本
使用真实患者数据测试5个任务的完整执行
"""

import os
import sys
import json

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pci_chain import PCIChain
from pci_chain.utils import create_llm

# Configuration
DATASET_PATH = "/Users/dsr/Desktop/paper/1213_PCI/患者级别数据集_标准版.json"
DATASET_PATH_EN = "/Users/dsr/Desktop/paper/1213_PCI/患者级别数据集_标准版_EN.json"
OUTPUT_DIR = "/Users/dsr/Desktop/paper/1213_PCI/results"

# Select dataset based on environment variable or default to Chinese
USE_ENGLISH = os.environ.get("USE_ENGLISH", "0") == "1"
ACTIVE_DATASET = DATASET_PATH_EN if USE_ENGLISH else DATASET_PATH

def load_test_patient():
    """加载测试患者（从标准版数据集）"""
    data_path = ACTIVE_DATASET
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 选择第一个患者
    patient = data[0]
    
    # 转换字段名以匹配PCIChain期望的格式
    patient_info = {
        "序号": patient.get("Patient_ID", ""),
        "主诉": patient.get("Chief_Complaint", ""),
        "入院诊断": patient.get("Admission_Diagnosis_Western", ""),
        "出院诊断": patient.get("Discharge_Diagnosis_Western", ""),
        "心超_LVEF": patient.get("Echo_LVEF", ""),
        "PCI_血管部位": patient.get("Culprit_Vessel", ""),
        "PCI_手术记录": patient.get("PCI_Operative_Note", ""),
        "既往史": f"高血压: {patient.get('History_Hypertension', '无')}, "
                  f"糖尿病: {patient.get('History_Diabetes', '无')}, "
                  f"吸烟: {patient.get('History_Smoking', '无')}",
        "出院用药_他汀": patient.get("Discharge_Statin", ""),
        "出院用药_抗血小板": patient.get("Discharge_Antiplatelets", ""),
        "出院用药_ACEI_ARB": patient.get("Discharge_ACEI_ARB", ""),
        "出院用药_Beta阻滞剂": patient.get("Discharge_Beta_Blocker", ""),
    }
    
    return patient_info, patient

def run_full_test():
    """运行完整测试"""
    print("=" * 60)
    print("PCIChain 全流程测试")
    print("=" * 60)
    
    # 1. 加载患者数据
    patient_info, raw_patient = load_test_patient()
    print(f"\n测试患者: {patient_info['序号']}")
    print(f"主诉: {patient_info['主诉']}")
    print(f"出院诊断: {patient_info['出院诊断'][:80]}...")
    print(f"LVEF: {patient_info['心超_LVEF']}")
    print(f"罪犯血管: {patient_info['PCI_血管部位']}")
    
    # 2. 创建LLM (使用Qwen)
    print("\n" + "-" * 40)
    print("正在创建LLM连接 (Qwen qwen-plus)...")
    
    try:
        llm = create_llm(
            provider="qwen",
            model="qwen-plus",
            api_key="sk-36f2de0e8e20483989525bb85eeed3a5",
            temperature=0.3
        )
        print("LLM连接成功")
    except Exception as e:
        print(f"LLM创建失败: {e}")
        return None
    
    # 3. 创建PCIChain
    print("\n" + "-" * 40)
    print("初始化PCIChain框架...")
    
    # Determine language based on dataset
    language = "en" if USE_ENGLISH else "zh"
    print(f"语言/Language: {language}")
    
    chain = PCIChain(
        llm=llm,
        max_corrections=2,
        enable_llm_contradiction=True,
        enable_rule_contradiction=True,
        verbose=True,
        language=language
    )
    
    # 4. 执行任务链
    print("\n" + "-" * 40)
    print("开始执行任务链...")
    print("-" * 40)
    
    try:
        result = chain.run(patient_info, patient_id=str(patient_info['序号']))
    except Exception as e:
        print(f"\n执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 5. 输出结果
    print("\n" + "=" * 60)
    print("执行结果摘要")
    print("=" * 60)
    print(chain.get_summary(result))
    
    # 6. 输出各任务详细结果
    print("\n" + "=" * 60)
    print("各任务输出详情")
    print("=" * 60)
    
    for task_name, output in result.outputs.items():
        print(f"\n【{task_name}】")
        print(f"置信度: {output.confidence}")
        if output.content:
            for key, value in list(output.content.items())[:5]:  # 只显示前5项
                print(f"  {key}: {str(value)[:100]}")
    
    return result


if __name__ == "__main__":
    result = run_full_test()
    
    if result:
        print("\n" + "=" * 60)
        print("测试完成!")
        print(f"总迭代次数: {result.total_iterations}")
        print(f"矛盾检测数: {result.contradictions_detected}")
        print(f"修正次数: {result.corrections_made}")
        print(f"执行时间: {result.execution_time:.2f}秒")
        print("=" * 60)
        
        # 保存结果到文件
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/Users/dsr/Desktop/paper/1213_PCI/results"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"test_result_{timestamp}.json")
        
        result_data = {
            "patient_id": result.patient_id,
            "execution_time": result.execution_time,
            "total_iterations": result.total_iterations,
            "contradictions_detected": result.contradictions_detected,
            "corrections_made": result.corrections_made,
            "outputs": {}
        }
        
        for task_name, output in result.outputs.items():
            result_data["outputs"][task_name] = {
                "confidence": output.confidence,
                "reasoning": output.reasoning[:3000] if output.reasoning else "",  # Limit size
                "content": output.content if output.content else {}
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
