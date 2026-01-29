#!/usr/bin/env python3
"""
反馈修正模块
处理矛盾检测后的修正流程
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

from .contradiction import Contradiction


@dataclass
class CorrectionRecord:
    """修正记录"""
    contradiction: Contradiction
    original_output: Dict[str, Any]
    corrected_output: Dict[str, Any]
    correction_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True


class FeedbackCorrector:
    """反馈修正器"""
    
    def __init__(self, agents: Dict, verbose: bool = True):
        self.agents = agents  # agent_name -> agent instance
        self.correction_history: List[CorrectionRecord] = []
        self.verbose = verbose
    
    def correct(self, contradictions: List[Contradiction], 
                all_outputs: Dict, patient_info: Dict) -> Dict:
        """执行反馈修正"""
        corrected_outputs = dict(all_outputs)
        
        for contradiction in contradictions:
            if self.verbose:
                print(f"[FeedbackCorrector] 检测到矛盾: {contradiction.rule_name}")
                print(f"  描述: {contradiction.description}")
                print(f"  涉及上游: {contradiction.upstream_agent}")
            
            # 获取上游Agent
            upstream_name = contradiction.upstream_agent
            if upstream_name not in self.agents:
                if self.verbose:
                    print(f"  跳过: 未找到Agent '{upstream_name}'")
                continue
            
            upstream_agent = self.agents[upstream_name]
            
            # 保存原始输出
            original_output = None
            if upstream_name in corrected_outputs:
                original_output = corrected_outputs[upstream_name].content.copy()
            
            # 请求上游验证
            if self.verbose:
                print(f"  发送质疑: {contradiction.query}")
            
            verification = upstream_agent.verify(contradiction.query, patient_info)
            
            # 记录修正
            if verification.get("corrected", False):
                if self.verbose:
                    print(f"  修正结果: 上游Agent已修正输出")
                    print(f"  修正原因: {verification.get('reason', '未说明')}")
                
                # 更新输出
                corrected_outputs[upstream_name] = upstream_agent.output
                
                # 记录修正历史
                record = CorrectionRecord(
                    contradiction=contradiction,
                    original_output=original_output,
                    corrected_output=verification.get("new_output", {}),
                    correction_reason=verification.get("reason", ""),
                    success=True
                )
                self.correction_history.append(record)
            else:
                if self.verbose:
                    print(f"  修正结果: 上游Agent确认原输出正确")
                
                # 记录未修正
                record = CorrectionRecord(
                    contradiction=contradiction,
                    original_output=original_output,
                    corrected_output=original_output,
                    correction_reason=verification.get("reason", "确认正确"),
                    success=False
                )
                self.correction_history.append(record)
        
        return corrected_outputs
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取修正统计"""
        total = len(self.correction_history)
        successful = sum(1 for r in self.correction_history if r.success)
        
        by_rule = {}
        for record in self.correction_history:
            rule = record.contradiction.rule_name
            if rule not in by_rule:
                by_rule[rule] = {"total": 0, "corrected": 0}
            by_rule[rule]["total"] += 1
            if record.success:
                by_rule[rule]["corrected"] += 1
        
        by_agent = {}
        for record in self.correction_history:
            agent = record.contradiction.upstream_agent
            if agent not in by_agent:
                by_agent[agent] = {"total": 0, "corrected": 0}
            by_agent[agent]["total"] += 1
            if record.success:
                by_agent[agent]["corrected"] += 1
        
        return {
            "total_contradictions": total,
            "successful_corrections": successful,
            "correction_rate": successful / total if total > 0 else 0,
            "by_rule": by_rule,
            "by_agent": by_agent
        }
    
    def get_history(self) -> List[Dict]:
        """获取修正历史"""
        return [
            {
                "rule": r.contradiction.rule_name,
                "description": r.contradiction.description,
                "upstream": r.contradiction.upstream_agent,
                "corrected": r.success,
                "reason": r.correction_reason,
                "timestamp": r.timestamp
            }
            for r in self.correction_history
        ]
