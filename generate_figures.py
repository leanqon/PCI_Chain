import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set professional style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

COLORS = {
    'primary': '#2c3e50',
    'secondary': '#e74c3c',
    'accent': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'gray': '#95a5a6',
    'light_gray': '#ecf0f1'
}

def save_fig(fig, name):
    try:
        fig.savefig(f'{name}.pdf', bbox_inches='tight', pad_inches=0.1)
        fig.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Saved {name}.pdf and {name}.png")
    except Exception as e:
        print(f"Error saving {name}: {e}")
        # Try without tight layout
        try:
            fig.savefig(f'{name}.png', dpi=300)
            print(f"Saved {name}.png (without tight layout)")
        except Exception as e2:
             print(f"Retry failed for {name}: {e2}")

def plot_model_comparison():
    """Figure 2: Model Comparison"""
    data = {
        'Model': ['GPT-4-Turbo', 'Claude-3-Opus', 'Qwen2-72B', 'DeepSeek-V2', 'Qwen2-7B'],
        'T1 F1': [0.97, 0.96, 0.96, 0.94, 0.88],
        'T3 Jaccard': [0.54, 0.53, 0.52, 0.49, 0.41],
        'T4 F1': [0.95, 0.94, 0.94, 0.91, 0.82],
        'Latency (s)': [22.1, 24.8, 18.3, 15.6, 8.2]
    }
    df = pd.DataFrame(data)
    
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1])
    
    # Plot 1: Performance Metrics
    ax1 = fig.add_subplot(gs[0])
    df_melt = df.melt(id_vars='Model', value_vars=['T1 F1', 'T3 Jaccard', 'T4 F1'], var_name='Metric', value_name='Score')
    
    sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', ax=ax1, palette='viridis')
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('')
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_title('(a) Performance Metrics by Model', loc='left', fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=15)
    
    # Plot 2: Latency vs Performance (Scatter)
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate average performance score
    df['Avg Score'] = (df['T1 F1'] + df['T3 Jaccard'] + df['T4 F1']) / 3
    
    sns.scatterplot(data=df, x='Latency (s)', y='Avg Score', s=200, hue='Model', ax=ax2, palette='deep', legend=False)
    
    # Add labels
    for i, row in df.iterrows():
        ax2.text(row['Latency (s)'], row['Avg Score']+0.01, row['Model'], 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax2.set_xlabel('Inference Latency (seconds)')
    ax2.set_ylabel('Mean Performance Score')
    ax2.set_title('(b) Efficiency Frontier', loc='left', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(5, 30)
    ax2.set_ylim(0.6, 1.0)
    
    # Draw pareto frontier roughly
    # High performance, low latency is better (top left)
    
    plt.tight_layout()
    save_fig(fig, 'fig_model_comparison')

def plot_ablation_studies():
    """Figure 3: Ablation Studies"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. SC Ablation
    sc_metrics = ['Dx Jaccard', 'Keyword Cov.', 'Confidence', 'Consistency']
    w_sc = [0.52, 0.783, 0.85, 0.961]
    wo_sc = [0.48, 0.742, 0.68, 0.876]
    
    x = np.arange(len(sc_metrics))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, wo_sc, width, label='w/o Self-Correction', color=COLORS['gray'])
    rects2 = axes[0].bar(x + width/2, w_sc, width, label='w/ Self-Correction', color=COLORS['accent'])
    
    axes[0].set_ylabel('Score / Rate')
    axes[0].set_title('(a) Self-Correction Impact', loc='left', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sc_metrics, rotation=15)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    
    # Add improvements text
    for i, (v1, v2) in enumerate(zip(wo_sc, w_sc)):
        imp = ((v2 - v1) / v1) * 100
        axes[0].text(i + width/2, v2 + 0.02, f'+{imp:.1f}%', ha='center', fontsize=9, color=COLORS['success'], fontweight='bold')

    # 2. RAG Ablation
    rag_metrics = ['Antiplatelet', 'Statin', 'ACEI/ARB', 'Beta-blocker']
    w_rag = [0.85, 1.00, 0.97, 0.95]
    wo_rag = [0.78, 0.96, 0.89, 0.88]
    
    x2 = np.arange(len(rag_metrics))
    
    axes[1].plot(x2, wo_rag, 'o--', label='w/o RAG', color=COLORS['gray'], linewidth=2, markersize=8)
    axes[1].plot(x2, w_rag, 'o-', label='w/ RAG', color=COLORS['secondary'], linewidth=2, markersize=8)
    
    axes[1].fill_between(x2, wo_rag, w_rag, alpha=0.1, color=COLORS['secondary'])
    
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('(b) RAG Integration Impact (Medication Agent)', loc='left', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(rag_metrics)
    axes[1].set_ylim(0.7, 1.05)
    axes[1].legend(loc='lower left')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'fig_ablation_study')

def plot_error_analysis():
    """Figure 4: Error Analysis"""
    # Data from Table S8
    errors = {
        'Coronary (T1)': {'Branch Ambiguity': 4, 'Non-treated Stenosis': 6},
        'Cardiac (T2)': {'Biomarker Threshold': 18, 'Symptom Overest.': 22, 'Insufficient Evid.': 15},
        'Diagnosis (T3)': {'Comorbidity Rank': 38, 'Coding Precision': 24, 'Proc. Detail': 11},
        'Medication (T4)': {'P2Y12 Selection': 28, 'Dose Optim.': 12, 'Contraindic. Gap': 5}
    }
    
    fig = plt.figure(figsize=(10, 8))
    
    # Prepare data for detailed stacked bar chart
    agents = list(errors.keys())
    
    # Calculate total errors per agent
    agent_totals = {agent: sum(errs.values()) for agent, errs in errors.items()}
    
    # Plot
    y_pos = np.arange(len(agents))
    bar_height = 0.5
    
    # Create a unified list of error types for color mapping
    all_error_types = []
    for errs in errors.values():
        all_error_types.extend(errs.keys())
        
    # Manually plot horizontal bars with breakdown
    ax = plt.gca()
    
    # Base colors for each agent
    agent_colors = [COLORS['primary'], COLORS['warning'], COLORS['success'], COLORS['secondary']]
    
    for i, (agent, errs) in enumerate(errors.items()):
        total = agent_totals[agent]
        current_left = 0
        
        # Sort errors by count
        sorted_errs = sorted(errs.items(), key=lambda x: x[1], reverse=True)
        
        for j, (err_type, count) in enumerate(sorted_errs):
            pct = (count / total) * 100
            
            # Vary lightness/alpha for sub-segments
            alpha = 1.0 - (j * 0.2)
            color = agent_colors[i]
            
            ax.barh(i, count, left=current_left, height=bar_height, color=color, alpha=alpha, edgecolor='white')
            
            # Add label if segment is large enough
            if count > 3:
                ax.text(current_left + count/2, i, f"{err_type}\n({count})", 
                        ha='center', va='center', color='white' if alpha > 0.5 else 'black', 
                        fontsize=8, fontweight='bold')
            
            current_left += count
            
        # Add total label
        ax.text(total + 1, i, f"Total: {total}", va='center', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agents, fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Error Cases')
    ax.set_title('Error Distribution and Categorization by Agent', fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, 'fig_error_analysis')

def create_architecture_diagram():
    """Figure 1: Architecture Diagram (Schematic)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    # ax.axis('off') # Keep axis on for debugging if needed, but off is better for schematic
    ax.axis('off')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 9)
    
    # Define box styles
    def draw_box(x, y, w, h, text, color, title=None):
        # Use simpler Rectangle for robustness
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=color, facecolor='#f8f9fa') # Light grey background
        ax.add_patch(rect)
        # Header
        if title:
            header_rect = patches.FancyBboxPatch((x, y+h-0.8), w, 0.8, boxstyle="round,pad=0.1",
                                                linewidth=0, facecolor=color)
            ax.add_patch(header_rect)
            ax.text(x+w/2, y+h-0.4, title, ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
        # Content text
        ax.text(x+w/2, y+h/2 if not title else y+(h-0.8)/2, text, ha='center', va='center', fontsize=9, wrap=True, color='black')
        return x+w, y+h/2 # Return potential connection point

    # 1. Input Layer
    draw_box(0.5, 4, 3, 2, "Clinical Documents\n(PDF/Word/Text)", COLORS['gray'], "Input Data")
    
    # 2. Perception Layer (Parallel)
    draw_box(5, 5.5, 3, 2, "T1: Coronary Agent\n(Anatomy Extraction)", COLORS['primary'], "Perception")
    draw_box(5, 2.5, 3, 2, "T2: Cardiac Function\n(LVEF Estimation)", COLORS['primary'], "Perception")
    
    # 3. Reasoning Layer
    draw_box(10, 4, 3, 2, "T3: Diagnosis Agent\n(ICD-10 Coding)", COLORS['accent'], "Reasoning")
    
    # 4. Decision Layer (with RAG)
    draw_box(15, 4, 3, 2, "T4: Medication Agent\n(GDMT Recommendations)", COLORS['secondary'], "Decision")
    
    # RAG Box connected to T4
    draw_box(15, 7, 3, 1.5, "Guideline Database\n(RAG Retrieval)", COLORS['warning'])
    
    # 5. Output Layer
    draw_box(20, 4, 3, 2, "T5: Report Agent\n(MDT Compilation)", COLORS['success'], "Output")
    
    # 6. Self-Correction Mechanism (Loop)
    rect_sc = patches.FancyBboxPatch((9, 1), 10, 1.2, boxstyle="round,pad=0.1", 
                                     linewidth=2, edgecolor=COLORS['secondary'], facecolor='#fff0f0', linestyle='--')
    ax.add_patch(rect_sc)
    ax.text(14, 1.6, "Self-Correction Mechanism (Contradiction Detector)", ha='center', fontweight='bold', color=COLORS['secondary'])
    
    # Arrows
    def draw_arrow(x1, y1, x2, y2, curved=False):
        style = "Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color="#34495e")
        if curved:
            conn = patches.ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                                          axesA=ax, axesB=ax, connectionstyle="arc3,rad=0.3", **kw)
        else:
            conn = patches.ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                                          axesA=ax, axesB=ax, **kw)
        ax.add_artist(conn)

    # Input -> Perception
    draw_arrow(3.6, 5, 4.9, 6.5)
    draw_arrow(3.6, 5, 4.9, 3.5)
    
    # Perception -> Reasoning
    draw_arrow(8.1, 6.5, 9.9, 5)
    draw_arrow(8.1, 3.5, 9.9, 5)
    
    # Reasoning -> Decision
    draw_arrow(13.1, 5, 14.9, 5)
    
    # RAG -> Decision
    draw_arrow(16.5, 6.9, 16.5, 6.1)
    
    # Decision -> Output
    draw_arrow(18.1, 5, 19.9, 5)
    
    # Feedback Loops (Manually simplified arrows to avoid complex bbox issues)
    ax.annotate("", xy=(11.5, 2.3), xytext=(11.5, 3.9), arrowprops=dict(arrowstyle="->", color=COLORS['secondary'], ls='--'))
    ax.annotate("", xy=(16.5, 2.3), xytext=(16.5, 3.9), arrowprops=dict(arrowstyle="->", color=COLORS['secondary'], ls='--'))
    ax.annotate("Feedback", xy=(6.5, 5.4), xytext=(9.5, 1.1), arrowprops=dict(arrowstyle="->", color=COLORS['secondary'], ls='--', connectionstyle="arc3,rad=-0.4"))
    
    # Force draw to compute layout
    # fig.canvas.draw() 
    
    # Save directly without tight_layout
    try:
        fig.savefig('fig_architecture_schematic.png', dpi=300, bbox_inches=None)
        fig.savefig('fig_architecture_schematic.pdf', bbox_inches=None)
        print("Saved fig_architecture_schematic.pdf and png (manual save)")
    except Exception as e:
        print(f"Error manually saving architecture: {e}")

if __name__ == "__main__":
    plot_model_comparison()
    plot_ablation_studies()
    plot_error_analysis()
    create_architecture_diagram()
    print("All figures generated successfully.")
