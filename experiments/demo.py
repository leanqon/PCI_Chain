#!/usr/bin/env python3
"""
PCIChain 演示脚本
展示框架的基本用法
"""

import os
import sys
import pandas as pd

# 添加项目路径 - 修复为正确的路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from pci_chain import PCIChain
from pci_chain.utils import create_llm, MockLLM


def load_sample_patient(data_path: str = None, patient_idx: int = 0) -> dict:
    """加载示例患者数据"""
    if data_path is None:
        data_path = "/Users/dsr/Desktop/paper/1213_PCI/1213_数据处理/融合数据集_OCR增强版.csv"
    
    if not os.path.exists(data_path):
        print(f"警告: 数据文件不存在，使用模拟数据")
        return get_mock_patient()
    
    df = pd.read_csv(data_path)
    patient = df.iloc[patient_idx].to_dict()
    return patient


def get_mock_patient() -> dict:
    """模拟患者数据（用于测试）"""
    return {
        "序号": 1,
        "主诉": "胸痛3天",
        "现病史": "患者3天前无明显诱因出现胸痛，位于胸骨后，呈压榨样，持续约30分钟，伴出汗、恶心",
        "既往史": "高血压病史5年，规律服用降压药；2型糖尿病3年，口服二甲双胍",
        "入院诊断": "急性ST段抬高型心肌梗死",
        "出院诊断": "1.冠状动脉粥样硬化性心脏病 急性前壁心肌梗死 2.高血压病3级 3.2型糖尿病",
        "手术及操作名称": "经皮冠状动脉介入治疗(PCI)+支架植入术",
        "PCI_术式": "支架植入",
        "PCI_血管部位": "前降支",
        "心超_LVEF": "45%",
        "出院用药_他汀": "阿托伐他汀20mg qn",
        "出院用药_抗血小板": "阿司匹林100mg qd + 氯吡格雷75mg qd",
    }


def demo_with_mock():
    """使用Mock LLM进行演示（无需API）"""
    print("=" * 60)
    print("PCIChain 演示 (Mock模式)")
    print("=" * 60)
    
    # 配置Mock响应
    mock_responses = {
        "心功能": """### 推理过程
患者诊断为急性前壁心肌梗死，提示前降支病变，可能影响左室功能。
心超显示LVEF 45%，属于轻度降低。

### 结论
- LVEF分级：轻度降低
- 预估LVEF范围：40-49%
- 置信度：0.85
- 主要依据：急性前壁心梗、心超LVEF 45%""",
        
        "冠脉": """### 病变血管
根据手术记录，病变位于前降支(LAD)。

### 病变详情
LAD近中段重度狭窄，已行支架植入。

### 结论
- 受累血管列表：[LAD]
- 主要病变血管：LAD
- 病变范围：单支病变
- 干预方式：支架植入
- 置信度：0.9""",
        
        "用药": """### 用药分析
患者为急性心梗PCI术后，LVEF 45%轻度降低，合并高血压、糖尿病。
需要标准二级预防四联用药。

### 推荐用药方案
1. 抗血小板：阿司匹林100mg qd + 替格瑞洛90mg bid (DAPT 12个月)
2. 他汀类：阿托伐他汀20mg qn
3. ACEI/ARB：培哚普利4mg qd（心梗后心室重构保护）
4. β受体阻滞剂：美托洛尔缓释片47.5mg qd

### 特殊用药
- 奥美拉唑20mg qd（DAPT期间胃保护）
- 二甲双胍0.5g tid（糖尿病）

### 注意事项
- 监测血压、心率
- 定期复查肝肾功能

### 结论
- 推荐药物数量：6种
- 完整四联：是
- 置信度：0.88"""
    }
    
    # 创建Mock LLM
    mock_llm = MockLLM(responses=mock_responses)
    
    # 创建PCIChain
    chain = PCIChain(
        llm=mock_llm,
        max_corrections=2,
        verbose=True
    )
    
    # 加载患者数据
    patient = get_mock_patient()
    print(f"\n患者信息: 序号 {patient.get('序号', 'N/A')}")
    print(f"主诉: {patient.get('主诉', 'N/A')}")
    
    # 执行任务链
    print("\n" + "-" * 40)
    result = chain.run(patient, patient_id=str(patient.get('序号', '')))
    
    # 输出结果
    print("\n" + "=" * 60)
    print("执行结果摘要")
    print("=" * 60)
    print(chain.get_summary(result))
    
    return result


def demo_with_api(api_key: str = None, provider: str = "openai", model: str = "gpt-4"):
    """使用真实API进行演示"""
    print("=" * 60)
    print(f"PCIChain 演示 (使用 {provider} API)")
    print("=" * 60)
    
    # 创建LLM
    llm = create_llm(provider, model=model, api_key=api_key)
    
    # 创建PCIChain
    chain = PCIChain(
        llm=llm,
        max_corrections=3,
        verbose=True
    )
    
    # 加载患者数据
    patient = load_sample_patient()
    print(f"\n患者信息: 序号 {patient.get('序号', 'N/A')}")
    
    # 执行任务链
    result = chain.run(patient, patient_id=str(patient.get('序号', '')))
    
    # 输出结果
    print("\n" + "=" * 60)
    print(chain.get_summary(result))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PCIChain演示")
    parser.add_argument("--mode", choices=["mock", "api"], default="mock",
                        help="运行模式: mock(无需API) 或 api(需要API key)")
    parser.add_argument("--provider", default="openai", 
                        help="API提供商: openai/zhipu")
    parser.add_argument("--model", default="gpt-4",
                        help="模型名称")
    parser.add_argument("--api-key", default=None,
                        help="API密钥(也可通过环境变量设置)")
    
    args = parser.parse_args()
    
    if args.mode == "mock":
        demo_with_mock()
    else:
        demo_with_api(args.api_key, args.provider, args.model)
