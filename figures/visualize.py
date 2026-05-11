import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

df_raw = pd.read_csv('main_summary_144.csv')
df_mean = pd.read_csv('main_summary_144_mean_std.csv')
df_best = pd.read_csv('main_summary_144_best_fixed.csv')

os.makedirs('figures', exist_ok=True)

METHOD_COLORS = {
    'lora': '#2196F3', 'qlora': '#FF9800', 'dora': '#4CAF50', 'ia3': '#E91E63',
}
METHOD_MARKERS = {'lora': 'o', 'qlora': 's', 'dora': 'D', 'ia3': '^'}
MODEL_NAMES = {'gemma': 'Gemma-2-2B-it', 'qwen': 'Qwen2.5-7B-Instruct'}
TASK_METRICS = {'gsm8k': 'EM', 'squad_v2': 'F1', 'dialogsum': 'ROUGE-L'}
TASK_NAMES = {'gsm8k': 'GSM8K', 'squad_v2': 'SQuAD-v2', 'dialogsum': 'DialogSum'}
BUDGETS = [128, 512, 2048]
METHODS = ['lora', 'qlora', 'dora', 'ia3']


def grouped_bar(ax, data, methods, budgets, value_col, err_col=None, ylabel=''):
    """Draw a grouped bar chart with methods as groups within each budget."""
    x = np.arange(len(budgets))
    width = 0.18
    for i, method in enumerate(methods):
        m_data = data[data['method'] == method].sort_values('budget')
        heights = m_data[value_col].values
        yerr = m_data[err_col].values if err_col and err_col in m_data.columns else None
        ax.bar(x + i * width - 1.5 * width, heights, width,
               yerr=yerr, label=method.upper(),
               color=METHOD_COLORS[method], capsize=3, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('Data Budget')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)


# ============================================================
# FIGURE 1: Main Results - Grouped Bar Charts
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('PEFT Methods Performance Comparison Across Tasks and Models',
             fontsize=16, fontweight='bold', y=1.02)

for row_idx, model in enumerate(['gemma', 'qwen']):
    for col_idx, task in enumerate(['gsm8k', 'squad_v2', 'dialogsum']):
        ax = axes[row_idx, col_idx]
        subset = df_mean[(df_mean['model'] == model) & (df_mean['task'] == task)]
        grouped_bar(ax, subset, METHODS, BUDGETS, 'mean', 'std', TASK_METRICS[task])
        ax.set_title(f'{MODEL_NAMES[model]} - {TASK_NAMES[task]}')
        if col_idx == 0:
            ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig1_main_results_bar.png')
plt.close()
print('[OK] fig1_main_results_bar.png')


# ============================================================
# FIGURE 2: Line Plots - Performance vs Data Budget
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Performance Scaling with Data Budget', fontsize=16, fontweight='bold', y=1.02)

for col_idx, task in enumerate(['gsm8k', 'squad_v2', 'dialogsum']):
    ax = axes[col_idx]
    for model in ['gemma', 'qwen']:
        subset = df_mean[(df_mean['model'] == model) & (df_mean['task'] == task)]
        for method in METHODS:
            m_data = subset[subset['method'] == method].sort_values('budget')
            if len(m_data) == 0:
                continue
            ls = '-' if model == 'gemma' else '--'
            label = f'{model.upper()}+{method.upper()}' if col_idx == 2 else None
            ax.errorbar(m_data['budget'], m_data['mean'], yerr=m_data['std'],
                        marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                        linestyle=ls, linewidth=1.5, markersize=7, capsize=3, label=label)

    ax.set_xlabel('Data Budget')
    ax.set_ylabel(TASK_METRICS[task])
    ax.set_title(TASK_NAMES[task])
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels(['128', '512', '2048'])
    ax.grid(alpha=0.3)

handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig2_performance_vs_budget.png')
plt.close()
print('[OK] fig2_performance_vs_budget.png')


# ============================================================
# FIGURE 3: VRAM Usage Comparison
# ============================================================
vram_stats = df_raw.groupby(['model', 'method', 'budget'])['peak_vram_gb'].agg(['mean', 'std']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Peak VRAM Usage by PEFT Method and Model', fontsize=16, fontweight='bold', y=1.02)

for col_idx, model in enumerate(['gemma', 'qwen']):
    ax = axes[col_idx]
    subset = vram_stats[vram_stats['model'] == model]
    subset = subset.rename(columns={'mean': 'mean_val', 'std': 'std_val'})
    grouped_bar(ax, subset, METHODS, BUDGETS, 'mean_val', 'std_val', 'Peak VRAM (GB)')
    ax.set_title(MODEL_NAMES[model])
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig3_vram_usage.png')
plt.close()
print('[OK] fig3_vram_usage.png')


# ============================================================
# FIGURE 4: Training Time Comparison
# ============================================================
time_stats = df_raw.groupby(['model', 'method', 'budget'])['train_time_min'].agg(['mean', 'std']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Time by PEFT Method and Model', fontsize=16, fontweight='bold', y=1.02)

for col_idx, model in enumerate(['gemma', 'qwen']):
    ax = axes[col_idx]
    subset = time_stats[time_stats['model'] == model]
    subset = subset.rename(columns={'mean': 'mean_val', 'std': 'std_val'})
    grouped_bar(ax, subset, METHODS, BUDGETS, 'mean_val', 'std_val', 'Training Time (min)')
    ax.set_title(MODEL_NAMES[model])
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig4_training_time.png')
plt.close()
print('[OK] fig4_training_time.png')


# ============================================================
# FIGURE 5: Best Method per (Model, Task, Budget)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Best PEFT Method by Task and Budget (with score)', fontsize=16, fontweight='bold', y=1.02)

for col_idx, model in enumerate(['gemma', 'qwen']):
    ax = axes[col_idx]
    subset = df_best[df_best['model'] == model]

    for task_idx, task in enumerate(['gsm8k', 'squad_v2', 'dialogsum']):
        task_data = subset[subset['task'] == task]
        for budget_idx, budget in enumerate(BUDGETS):
            row = task_data[task_data['budget'] == budget]
            if len(row) == 0:
                continue
            best = row.iloc[0]
            method_name = best['best_method'].upper()
            score = best['best_score']

            rect = plt.Rectangle((budget_idx - 0.4, task_idx - 0.4), 0.8, 0.8,
                                  facecolor=METHOD_COLORS.get(best['best_method'], 'gray'),
                                  alpha=0.3, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(budget_idx, task_idx + 0.12, method_name,
                    ha='center', va='center', fontsize=9, fontweight='bold')
            ax.text(budget_idx, task_idx - 0.18, f'{score:.4f}',
                    ha='center', va='center', fontsize=8, color='#555')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['128', '512', '2048'])
    ax.set_xlabel('Data Budget')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['GSM8K', 'SQuAD-v2', 'DialogSum'])
    ax.set_title(MODEL_NAMES[model])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

plt.tight_layout()
plt.savefig('figures/fig5_best_method_heatmap.png')
plt.close()
print('[OK] fig5_best_method_heatmap.png')


# ============================================================
# FIGURE 6: Efficiency Scatter - Score vs VRAM & Score vs Time
# ============================================================
efficiency_data = []
for model in ['gemma', 'qwen']:
    for task in ['gsm8k', 'squad_v2', 'dialogsum']:
        for budget in BUDGETS:
            for method in METHODS:
                rows = df_raw[(df_raw['model'] == model) & (df_raw['task'] == task) &
                               (df_raw['budget'] == budget) & (df_raw['method'] == method)]
                if len(rows) == 0:
                    continue
                metric_col = {'gsm8k': 'em', 'squad_v2': 'f1', 'dialogsum': 'rougeL'}[task]
                efficiency_data.append({
                    'model': model, 'task': task, 'method': method, 'budget': budget,
                    'score': rows[metric_col].mean(),
                    'vram': rows['peak_vram_gb'].mean(),
                    'time': rows['train_time_min'].mean(),
                })

eff_df = pd.DataFrame(efficiency_data)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Efficiency Analysis: Performance vs Resource Usage', fontsize=16, fontweight='bold', y=1.02)

for col_idx, task in enumerate(['gsm8k', 'squad_v2', 'dialogsum']):
    task_eff = eff_df[eff_df['task'] == task]

    # Score vs VRAM
    ax = axes[0, col_idx]
    for model in ['gemma', 'qwen']:
        for method in METHODS:
            d = task_eff[(task_eff['model'] == model) & (task_eff['method'] == method)]
            ls = '-' if model == 'gemma' else '--'
            ax.plot(d['vram'], d['score'], marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method], linestyle=ls, linewidth=1.2, markersize=6)
            for _, r in d.iterrows():
                ax.annotate(str(int(r['budget'])), (r['vram'], r['score']),
                            textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)
    ax.set_xlabel('Peak VRAM (GB)')
    ax.set_ylabel(TASK_METRICS[task])
    ax.set_title(f'{TASK_NAMES[task]} - Score vs VRAM')
    ax.grid(alpha=0.3)

    # Score vs Training Time
    ax = axes[1, col_idx]
    for model in ['gemma', 'qwen']:
        for method in METHODS:
            d = task_eff[(task_eff['model'] == model) & (task_eff['method'] == method)]
            ls = '-' if model == 'gemma' else '--'
            ax.plot(d['time'], d['score'], marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method], linestyle=ls, linewidth=1.2, markersize=6)
            for _, r in d.iterrows():
                ax.annotate(str(int(r['budget'])), (r['time'], r['score']),
                            textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)
    ax.set_xlabel('Training Time (min)')
    ax.set_ylabel(TASK_METRICS[task])
    ax.set_title(f'{TASK_NAMES[task]} - Score vs Time')
    ax.grid(alpha=0.3)

legend_elements = []
for method in METHODS:
    legend_elements.append(Line2D([0], [0], marker=METHOD_MARKERS[method], color='gray',
                                   markerfacecolor=METHOD_COLORS[method], markersize=8,
                                   linestyle='-', label=method.upper(), linewidth=1.2))
legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', label='Gemma', linewidth=1.2))
legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', label='Qwen', linewidth=1.2))
fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig6_efficiency_scatter.png')
plt.close()
print('[OK] fig6_efficiency_scatter.png')


# ============================================================
# FIGURE 7: Trainable Parameter Ratio
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Trainable Parameter Ratio by PEFT Method', fontsize=16, fontweight='bold', y=1.02)

for col_idx, model in enumerate(['gemma', 'qwen']):
    ax = axes[col_idx]
    subset = df_raw[df_raw['model'] == model][['method', 'trainable_ratio']].drop_duplicates()
    subset = subset.set_index('method').reindex(METHODS).reset_index()

    bars = ax.bar(subset['method'].str.upper(), subset['trainable_ratio'] * 100,
                  color=[METHOD_COLORS[m] for m in subset['method']], edgecolor='black', linewidth=0.5)
    for bar, ratio in zip(bars, subset['trainable_ratio']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ratio*100:.4f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Trainable Parameters (%)')
    ax.set_title(MODEL_NAMES[model])
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig7_trainable_params.png')
plt.close()
print('[OK] fig7_trainable_params.png')


# ============================================================
# FIGURE 8: Method Rankings (Bar Chart - simpler than radar)
# ============================================================
ranking_data = []
for model in ['gemma', 'qwen']:
    for task in ['gsm8k', 'squad_v2', 'dialogsum']:
        for budget in BUDGETS:
            sub = df_mean[(df_mean['model'] == model) & (df_mean['task'] == task) & (df_mean['budget'] == budget)]
            sub = sub.sort_values('mean', ascending=False).reset_index(drop=True)
            sub['rank'] = range(1, len(sub) + 1)
            ranking_data.append(sub[['model', 'task', 'budget', 'method', 'mean', 'rank']])

rank_df = pd.concat(ranking_data, ignore_index=True)
avg_rank = rank_df.groupby(['model', 'method'])['rank'].mean().reset_index()
avg_rank.columns = ['model', 'method', 'avg_rank']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Average Method Ranking (lower = better)', fontsize=16, fontweight='bold', y=1.02)

# Overall average
ax = axes[0]
overall = avg_rank.groupby('method')['avg_rank'].mean().reset_index()
overall = overall.set_index('method').reindex(METHODS).reset_index()
bars = ax.barh(overall['method'].str.upper(), overall['avg_rank'],
               color=[METHOD_COLORS[m] for m in overall['method']], edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, overall['avg_rank']):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', fontsize=10)
ax.set_xlabel('Average Rank')
ax.set_title('Overall (Both Models)')
ax.set_xlim(0, 4.5)
ax.grid(axis='x', alpha=0.3)

for col_idx, model in enumerate(['gemma', 'qwen']):
    ax = axes[col_idx + 1]
    sub = avg_rank[avg_rank['model'] == model].set_index('method').reindex(METHODS).reset_index()
    bars = ax.barh(sub['method'].str.upper(), sub['avg_rank'],
                   color=[METHOD_COLORS[m] for m in sub['method']], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, sub['avg_rank']):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', fontsize=10)
    ax.set_xlabel('Average Rank')
    ax.set_title(MODEL_NAMES[model])
    ax.set_xlim(0, 4.5)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig8_method_ranking.png')
plt.close()
print('[OK] fig8_method_ranking.png')


# ============================================================
# FIGURE 9: Score Heatmap
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Performance Score Heatmap (Mean)', fontsize=16, fontweight='bold', y=1.02)

for row_idx, model in enumerate(['gemma', 'qwen']):
    for col_idx, task in enumerate(['gsm8k', 'squad_v2', 'dialogsum']):
        ax = axes[row_idx, col_idx]
        sub = df_mean[(df_mean['model'] == model) & (df_mean['task'] == task)].copy()
        pivot = sub.pivot_table(index='method', columns='budget', values='mean')
        pivot = pivot.reindex(METHODS)
        pivot.columns = ['128', '512', '2048']

        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{MODEL_NAMES[model]} - {TASK_NAMES[task]}')
        ax.set_xlabel('Data Budget')
        ax.set_ylabel('Method')

plt.tight_layout()
plt.savefig('figures/fig9_score_heatmap.png')
plt.close()
print('[OK] fig9_score_heatmap.png')


# ============================================================
# FIGURE 10: Combined VRAM+Time+Score overview
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Training Cost Summary (VRAM and Time per Method-Model-Budget)',
             fontsize=16, fontweight='bold', y=1.02)

for row_idx, metric_name in enumerate(['peak_vram_gb', 'train_time_min']):
    ylabel = 'Peak VRAM (GB)' if metric_name == 'peak_vram_gb' else 'Training Time (min)'
    stats = df_raw.groupby(['model', 'method', 'budget'])[metric_name].agg(['mean', 'std']).reset_index()
    for col_idx, model in enumerate(['gemma', 'qwen']):
        ax = axes[row_idx, col_idx]
        sub = stats[stats['model'] == model].rename(columns={'mean': 'mean_val', 'std': 'std_val'})
        grouped_bar(ax, sub, METHODS, BUDGETS, 'mean_val', 'std_val', ylabel)
        ax.set_title(f'{MODEL_NAMES[model]}')
        ax.legend(loc='best', fontsize=8)

axes[0, 0].set_ylabel('Peak VRAM (GB)', fontweight='bold')
axes[1, 0].set_ylabel('Training Time (min)', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig10_cost_summary.png')
plt.close()
print('[OK] fig10_cost_summary.png')


print('\n=== All 10 figures saved to ./figures/ ===')
