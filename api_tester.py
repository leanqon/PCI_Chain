#!/usr/bin/env python3
"""
PCIChain API Tester
测试不同LLM API的PCIChain工作流
"""

import requests
import json
import os
import time
from datetime import datetime

# API Configuration
API_CONFIGS = {
    "deepseek": {
        "url": "https://api.qnaigc.com/v1/chat/completions",
        "models": ["deepseek/deepseek-v3.2-251201"]
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    }
}

def get_api_config(model_name):
    """根据模型名称选择API配置"""
    for provider, config in API_CONFIGS.items():
        if any(m in model_name for m in config["models"]) or provider in model_name.lower():
            return config["url"]
    # Default to qnaigc for unknown models
    return "https://api.qnaigc.com/v1/chat/completions"

def call_api(api_key, model, messages, temperature=0.1, max_tokens=2000):
    """调用LLM API"""
    url = get_api_config(model)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "stream": False,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # Extract content and usage
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        
        return {
            "success": True,
            "content": content,
            "usage": usage,
            "raw_response": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None,
            "usage": {}
        }

def load_test_data(folder, lang="en"):
    """加载测试数据"""
    test_files = [
        f"{lang}_T1_coronary.json",
        f"{lang}_T2_cardiac.json",
        f"{lang}_T3_diagnosis.json",
        f"{lang}_T4_medication.json",
        f"{lang}_T5_report.json",
        f"{lang}_consistency_check.json"
    ]
    
    tests = {}
    for filename in test_files:
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_name = filename.replace(f"{lang}_", "").replace(".json", "")
                tests[test_name] = data["messages"]
    
    return tests

def run_pci_chain_test(api_key, model, test_folder, lang="en", output_folder=None):
    """运行完整的PCIChain测试"""
    
    print(f"\n{'='*60}")
    print(f"PCIChain API Test")
    print(f"Model: {model}")
    print(f"Language: {lang}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Load test data
    tests = load_test_data(test_folder, lang)
    
    if not tests:
        print("Error: No test files found!")
        return None
    
    results = {
        "model": model,
        "language": lang,
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "total_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    
    # Run each test
    for test_name, messages in tests.items():
        print(f"\n--- Testing: {test_name} ---")
        
        start_time = time.time()
        response = call_api(api_key, model, messages)
        elapsed = time.time() - start_time
        
        if response["success"]:
            print(f"✓ Success ({elapsed:.2f}s)")
            print(f"  Tokens: {response['usage']}")
            
            # Accumulate usage
            usage = response["usage"]
            results["total_usage"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
            results["total_usage"]["completion_tokens"] += usage.get("completion_tokens", 0)
            results["total_usage"]["total_tokens"] += usage.get("total_tokens", 0)
            
            # Store result
            results["tests"][test_name] = {
                "success": True,
                "elapsed_seconds": elapsed,
                "usage": usage,
                "output": response["content"]
            }
            
            # Print preview
            preview = response["content"][:200] + "..." if len(response["content"]) > 200 else response["content"]
            print(f"  Preview: {preview}")
            
        else:
            print(f"✗ Failed: {response['error']}")
            results["tests"][test_name] = {
                "success": False,
                "error": response["error"]
            }
    
    # Calculate cost
    input_tokens = results["total_usage"]["prompt_tokens"]
    output_tokens = results["total_usage"]["completion_tokens"]
    cost_rmb = (input_tokens / 1000 * 0.002) + (output_tokens / 1000 * 0.003)
    results["estimated_cost_rmb"] = round(cost_rmb, 4)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total Input Tokens: {input_tokens}")
    print(f"Total Output Tokens: {output_tokens}")
    print(f"Estimated Cost: ¥{cost_rmb:.4f}")
    print(f"Tests Passed: {sum(1 for t in results['tests'].values() if t.get('success'))}/{len(results['tests'])}")
    
    # Save results
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(
            output_folder, 
            f"test_result_{model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results

def evaluate_results(results):
    """评估测试结果"""
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}\n")
    
    evaluations = {}
    
    for test_name, test_result in results.get("tests", {}).items():
        if not test_result.get("success"):
            evaluations[test_name] = {"status": "FAILED", "reason": test_result.get("error")}
            continue
        
        output = test_result.get("output", "")
        
        # Basic checks
        checks = {
            "has_json": "{" in output and "}" in output,
            "has_confidence": "confidence" in output.lower(),
            "has_reasoning": "reasoning" in output.lower() or "推理" in output,
            "reasonable_length": len(output) > 100
        }
        
        # Agent-specific checks
        if "T1" in test_name:
            checks["has_vessel"] = any(v in output for v in ["LAD", "LCX", "RCA", "LM"])
            checks["has_stenosis"] = "%" in output or "stenosis" in output.lower()
        elif "T2" in test_name:
            checks["has_lvef"] = "LVEF" in output or "lvef" in output or "EF" in output
        elif "T3" in test_name:
            checks["has_icd10"] = "I20" in output or "I10" in output or "icd" in output.lower()
        elif "T4" in test_name:
            checks["has_medication"] = any(m in output.lower() for m in ["aspirin", "statin", "阿司匹林", "他汀"])
        elif "T5" in test_name:
            checks["is_markdown"] = "#" in output or "**" in output
        elif "consistency" in test_name:
            checks["has_contradiction_check"] = "contradiction" in output.lower() or "矛盾" in output
        
        passed = sum(checks.values())
        total = len(checks)
        score = passed / total
        
        status = "PASS" if score >= 0.7 else "PARTIAL" if score >= 0.5 else "FAIL"
        
        evaluations[test_name] = {
            "status": status,
            "score": score,
            "checks": checks,
            "passed": f"{passed}/{total}"
        }
        
        print(f"{test_name}: {status} ({passed}/{total})")
        for check, result in checks.items():
            print(f"  {'✓' if result else '✗'} {check}")
    
    return evaluations


if __name__ == "__main__":
    # Configuration
    API_KEY = "sk-d13415b66423aa8c14edcfe54a35f0f4603ad768a4c33d9ed73a9c94553e597e"
    MODEL = "deepseek/deepseek-v3.2-251201"
    TEST_FOLDER = "/Users/dsr/Desktop/paper/1213_PCI/pci_chain/api_test_data"
    OUTPUT_FOLDER = "/Users/dsr/Desktop/paper/1213_PCI/results"
    LANG = "en"
    
    # Run tests
    results = run_pci_chain_test(
        api_key=API_KEY,
        model=MODEL,
        test_folder=TEST_FOLDER,
        lang=LANG,
        output_folder=OUTPUT_FOLDER
    )
    
    # Evaluate
    if results:
        evaluations = evaluate_results(results)
        
        # Save evaluation
        eval_file = os.path.join(
            OUTPUT_FOLDER, 
            f"evaluation_{MODEL.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        print(f"\nEvaluation saved to: {eval_file}")
