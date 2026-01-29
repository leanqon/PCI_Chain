#!/usr/bin/env python3
"""
PCIChain: 任务链编排器
协调多个Agent按临床逻辑顺序执行
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .agents.base import BaseAgent, AgentOutput
from .agents.cardiac import CardiacFunctionAgent
from .agents.coronary import CoronaryAgent
from .agents.diagnosis import DiagnosisAgent
from .agents.medication import MedicationAgent
from .agents.report import ReportAgent
# Note: RiskAgent removed due to lack of follow-up data
from .utils.contradiction import ContradictionDetector, Contradiction
from .utils.feedback import FeedbackCorrector, CorrectionRecord
from .utils.llm import BaseLLM, create_llm


@dataclass
class ChainResult:
    """任务链执行结果 / Chain Execution Result"""
    outputs: Dict[str, AgentOutput]  # 各Agent输出
    corrections: List[CorrectionRecord]  # 修正记录
    contradictions_detected: int  # 检测到的矛盾数
    corrections_made: int  # 成功修正数
    total_iterations: int  # 总迭代次数
    execution_time: float  # 执行时间(秒)
    patient_id: str = ""


class PCIChain:
    """PCI任务链编排器 / PCI Task Chain Orchestrator"""
    
    def __init__(self, llm: BaseLLM = None, 
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 max_corrections: int = 3,
                 enable_llm_contradiction: bool = True,
                 enable_rule_contradiction: bool = True,
                 verbose: bool = True,
                 language: str = "zh"):
        """
        初始化PCIChain / Initialize PCIChain
        
        Args:
            llm: LLM实例，如果不提供则根据provider创建
            llm_provider: LLM提供商 (openai/zhipu/mock)
            llm_model: 模型名称
            max_corrections: 最大修正迭代次数
            enable_llm_contradiction: 启用LLM辅助矛盾检测
            enable_rule_contradiction: 启用规则矛盾检测
            verbose: 详细输出
            language: 语言选择 ("zh" 中文 / "en" 英文)
        """
        # 创建或使用LLM
        if llm is None:
            try:
                self.llm = create_llm(llm_provider, model=llm_model)
            except Exception as e:
                print(f"Warning: Cannot create LLM ({e}), using MockLLM")
                self.llm = create_llm("mock")
        else:
            self.llm = llm
        
        self.max_corrections = max_corrections
        self.verbose = verbose
        self.language = language  # 语言选择
        
        # 初始化Agents / Initialize all Agents
        self.agents = self._create_agents()
        
        # 初始化矛盾检测器 / Initialize Contradiction Detector
        self.contradiction_detector = ContradictionDetector(
            llm=self.llm if enable_llm_contradiction else None,
            enable_llm=enable_llm_contradiction,
            enable_rules=enable_rule_contradiction
        )
        
        # 初始化反馈修正器 / Initialize Feedback Corrector
        self.feedback_corrector = FeedbackCorrector(
            agents=self.agents,
            verbose=verbose
        )
        
        # 定义任务链 / Define task chain
        # 格式: [(stage_type, agent_names), ...]
        # stage_type: 'parallel' 或 'sequential'
        # 任务链定义 (5个任务，无T5风险分层因无随访数据)
        # Task chain definition (5 tasks, no Risk due to lack of follow-up data)
        self.chain_definition = [
            ("parallel", ["coronary", "cardiac_function"]),  # T1, T2 并行
            ("sequential", ["diagnosis"]),  # T3: 诊断综合
            ("sequential", ["medication"]),  # T4: 用药推荐
            ("sequential", ["report"]),  # T5: MDT报告 (原T6)
        ]
    
    def _create_agents(self) -> Dict[str, BaseAgent]:
        """创建所有Agent / Create all Agents (5 agents, no Risk)"""
        return {
            "coronary": CoronaryAgent(self.llm, language=self.language),           # T1: 冠脉病变识别
            "cardiac_function": CardiacFunctionAgent(self.llm, language=self.language),  # T2: 心功能推理
            "diagnosis": DiagnosisAgent(self.llm, language=self.language),          # T3: 诊断综合评估
            "medication": MedicationAgent(self.llm, language=self.language),        # T4: 用药推荐
            "report": ReportAgent(self.llm, language=self.language),                # T5: MDT报告生成
        }
    
    def run(self, patient_info: Dict, patient_id: str = "") -> ChainResult:
        """
        执行任务链
        
        Args:
            patient_info: 患者病历信息字典
            patient_id: 患者ID（可选，用于记录）
        
        Returns:
            ChainResult: 执行结果
        """
        start_time = datetime.now()
        
        outputs: Dict[str, AgentOutput] = {}
        total_contradictions = 0
        total_corrections = 0
        total_iterations = 0
        
        for stage_type, agent_names in self.chain_definition:
            # 过滤出已实现的Agent
            available_agents = [n for n in agent_names if n in self.agents]
            
            if not available_agents:
                continue
            
            if stage_type == "parallel":
                # 并行执行
                if self.verbose:
                    print(f"\n[PCIChain] 并行执行: {available_agents}")
                
                for agent_name in available_agents:
                    agent = self.agents[agent_name]
                    if self.verbose:
                        print(f"  执行 {agent_name}...")
                    output = agent.process(patient_info, outputs)
                    outputs[agent_name] = output
                    if self.verbose:
                        print(f"  {agent_name} 置信度: {output.confidence:.2f}")
            
            else:  # sequential
                for agent_name in available_agents:
                    agent = self.agents[agent_name]
                    if self.verbose:
                        print(f"\n[PCIChain] 串行执行: {agent_name}")
                    
                    # 执行当前Agent
                    output = agent.process(patient_info, outputs)
                    outputs[agent_name] = output
                    
                    if self.verbose:
                        print(f"  {agent_name} 置信度: {output.confidence:.2f}")
                    
                    # 矛盾检测
                    contradictions = self.contradiction_detector.detect(
                        agent_name, output, outputs
                    )
                    
                    if contradictions:
                        total_contradictions += len(contradictions)
                        if self.verbose:
                            print(f"  检测到 {len(contradictions)} 个矛盾")
                        
                        # 反馈修正循环
                        correction_count = 0
                        while contradictions and correction_count < self.max_corrections:
                            # 执行修正
                            outputs = self.feedback_corrector.correct(
                                contradictions, outputs, patient_info
                            )
                            
                            # 重新执行当前Agent
                            output = agent.process(patient_info, outputs)
                            outputs[agent_name] = output
                            
                            # 重新检测矛盾
                            contradictions = self.contradiction_detector.detect(
                                agent_name, output, outputs
                            )
                            
                            correction_count += 1
                            total_iterations += 1
                        
                        total_corrections += correction_count
        
        # 计算执行时间
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 构建结果
        result = ChainResult(
            outputs=outputs,
            corrections=self.feedback_corrector.correction_history.copy(),
            contradictions_detected=total_contradictions,
            corrections_made=total_corrections,
            total_iterations=total_iterations,
            execution_time=execution_time,
            patient_id=patient_id
        )
        
        if self.verbose:
            print(f"\n[PCIChain] 执行完成")
            print(f"  检测到矛盾: {total_contradictions}")
            print(f"  执行修正: {total_corrections}")
            print(f"  执行时间: {execution_time:.2f}秒")
        
        return result
    
    def run_baseline(self, patient_info: Dict, patient_id: str = "") -> ChainResult:
        """
        运行基线版本（无反馈修正）
        用于对比实验
        """
        # 临时禁用反馈修正
        original_max = self.max_corrections
        self.max_corrections = 0
        
        result = self.run(patient_info, patient_id)
        
        # 恢复设置
        self.max_corrections = original_max
        
        return result
    
    def get_summary(self, result: ChainResult) -> str:
        """生成结果摘要"""
        lines = [
            f"=== PCIChain 执行摘要 ===",
            f"患者ID: {result.patient_id}",
            f"执行时间: {result.execution_time:.2f}秒",
            f"检测矛盾: {result.contradictions_detected}",
            f"成功修正: {result.corrections_made}",
            "",
            "--- Agent输出 ---"
        ]
        
        for name, output in result.outputs.items():
            lines.append(f"\n[{name}] 置信度: {output.confidence:.2f}")
            for k, v in output.content.items():
                lines.append(f"  {k}: {v}")
        
        if result.corrections:
            lines.append("\n--- 修正记录 ---")
            for c in result.corrections:
                lines.append(f"  {c.contradiction.rule_name}: {c.correction_reason}")
        
        return "\n".join(lines)
