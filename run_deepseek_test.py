#!/usr/bin/env python3
"""
使用 DeepSeek API 运行 PCIChain 完整流程测试
禁用 RAG 以避免 PDF 解析噪音
"""

import sys
import os
import json
import warnings
from datetime import datetime

# 抑制 PDF 解析警告
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 添加项目路径
sys.path.insert(0, '/Users/dsr/Desktop/paper/1213_PCI')

# 在导入之前 mock 掉 RAG
class MockRetriever:
    def retrieve_for_medication(self, **kwargs):
        return ""
    def retrieve(self, query, top_k=3):
        return []

import pci_chain.rag as rag_module
rag_module.get_retriever = lambda: MockRetriever()

from pci_chain.chain import PCIChain
from pci_chain.utils.llm import create_llm

# API 配置
API_KEY = "sk-d13415b66423aa8c14edcfe54a35f0f4603ad768a4c33d9ed73a9c94553e597e"
MODEL = "deepseek/deepseek-v3.2-251201"

# 测试患者数据（英文版，已脱敏）
TEST_PATIENT = {
    "patient_id": "P001",
    "age": 70,
    "gender": "Female",
    "chief_complaint": "Chest tightness for 2 months",
    "history_present_illness": """Patient developed intermittent chest tightness 2 months ago without obvious triggers. 
Symptoms are mild but aggravated by exertion or emotional stress, accompanied by shortness of breath on exercise. 
History of hypertension with maximum BP 180mmHg, controlled with candesartan 8mg qd and amlodipine 5mg qd.""",
    "past_medical_history": "Hypertension for years. Cholecystectomy 20, 8, and 5 years ago.",
    "physical_examination": """T 37.0°C, P 76/min, R 18/min, BP 168/84mmHg
Alert, no distress, cardiopulmonary exam unremarkable, no lower extremity edema.""",
    "echocardiography": """Mildly enlarged LA, LVEF 63%. 
Mild MR. Aortic valve calcification with mild regurgitation. 
LV diastolic dysfunction.""",
    "coronary_angiography": """Single-vessel disease of LAD. 
No significant stenosis at LM ostium. 
Proximal LAD 85% stenosis, mid LAD 50% stenosis. 
LCX and RCA without significant stenosis.""",
    "pci_procedure": """A 3.0×30mm stent was deployed at the LAD lesion. 
Post-procedure angiography showed residual stenosis <10%, TIMI 3 flow.""",
    "入院诊断": "Unstable angina, Hypertension Stage 3",
    "出院诊断": "1. Unstable angina (single-vessel disease, LAD PCI)\n2. Hypertension Stage 3 (very high risk)\n3. Reflux esophagitis\n4. Chronic gastritis",
    "心超_LVEF": "63%",
    "word_出院用药": """Amlodipine 5mg qd
Atorvastatin 20mg qd
Clopidogrel 75mg qd
Irbesartan/HCTZ 150mg qd
Metoprolol XL 47.5mg qd
Pantoprazole 40mg qd
Aspirin 100mg qd"""
}


def main():
    print("=" * 70)
    print("PCIChain 完整流程测试 (DeepSeek API)")
    print(f"模型: {MODEL}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 创建 LLM 实例
    print("\n[1] 创建 DeepSeek LLM 实例...")
    llm = create_llm(
        provider="deepseek",
        model=MODEL,
        api_key=API_KEY,
        temperature=0.1
    )
    
    # 创建 PCIChain 实例
    print("[2] 创建 PCIChain 实例...")
    chain = PCIChain(
        llm=llm,
        max_corrections=2,  # 减少修正次数以加快测试
        enable_llm_contradiction=True,
        enable_rule_contradiction=True,
        verbose=True,
        language="en"  # 英文版
    )
    
    # 运行完整流程
    print("[3] 运行 PCIChain 完整流程（含自修正）...")
    print("-" * 70)
    
    result = chain.run(
        patient_info=TEST_PATIENT,
        patient_id="P001"
    )
    
    # 输出结果摘要
    print("\n" + "=" * 70)
    print("执行结果摘要")
    print("=" * 70)
    
    summary = chain.get_summary(result)
    print(summary)
    
    # 保存详细结果
    output_dir = "/Users/dsr/Desktop/paper/1213_PCI/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 将结果转换为可序列化格式
    result_dict = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "patient_id": result.patient_id,
        "execution_time": result.execution_time,
        "contradictions_detected": result.contradictions_detected,
        "corrections_made": result.corrections_made,
        "total_iterations": result.total_iterations,
        "outputs": {}
    }
    
    for agent_name, output in result.outputs.items():
        result_dict["outputs"][agent_name] = {
            "content": output.content,
            "confidence": output.confidence,
            "reasoning": output.reasoning[:1000] if output.reasoning else None
        }

    
    # 保存修正记录
    result_dict["corrections"] = []
    for correction in result.corrections:
        result_dict["corrections"].append({
            "agent": correction.agent_name,
            "reason": correction.reason,
            "changes": str(correction.changes)[:500]
        })
    
    output_file = os.path.join(
        output_dir,
        f"pcichain_deepseek_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n详细结果已保存至: {output_file}")
    
    # 打印各 Agent 输出
    print("\n" + "=" * 70)
    print("各 Agent 详细输出")
    print("=" * 70)
    
    for agent_name, output in result.outputs.items():
        print(f"\n--- {agent_name} ---")
        print(f"置信度: {output.confidence}")
        print(f"内容: {json.dumps(output.content, ensure_ascii=False, indent=2)}")
    
    return result


if __name__ == "__main__":
    result = main()
