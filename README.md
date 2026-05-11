# DSAA PEFT Handoff

这是 DSAA PEFT 项目的清理交接版仓库，用于代码审阅、结果复核、图表重现，以及在相同数据布局下重新运行实验。项目比较四种参数高效微调方法在三个指令微调任务上的表现与资源成本，并以 `Final Report/main0509.tex` 作为完整书面报告。

研究设置：

- 模型：`Qwen/Qwen2.5-3B-Instruct`、`google/gemma-2-2b-it`
- 方法：LoRA、QLoRA、DoRA、IA3
- 任务：GSM8K（EM）、SQuAD v2（F1 / EM）、DialogSum（ROUGE-L）
- 数据预算：`128`、`512`、`2048` 条训练样本
- 主实验种子：`42`、`43`
- 主实验矩阵：`4 methods × 2 models × 3 tasks × 3 budgets × 2 seeds = 144` 个训练 run，外加 6 个无微调的 base model 基线
- 每个评估文件默认记录前 `200` 个测试/验证样本的预测结果

## Repository Layout

```text
.
├── data/
│   ├── dialogsum/                # train_*.jsonl, validation.jsonl, test.jsonl
│   ├── gsm8k/                    # train_*.jsonl, test.jsonl
│   └── squad_v2/                 # train_*.jsonl, validation.jsonl
├── figures/                      # 由 visualize.py 生成的图表 + 图表说明
├── Final Report/                 # 书面报告 LaTeX 源码 + 报告引用的图
│   ├── main0509.tex
│   ├── references.bib
│   └── figures/                  # Workflow.png + fig1..fig12 的报告版本
├── manifests/                    # 批量实验 manifest
│   └── main_debug_72.csv
├── results/
│   ├── base_model/               # base model 无微调评估输出与对比表
│   ├── main/                     # 144 个主实验的 eval/meta JSON
│   ├── main_summary_144.csv
│   ├── main_summary_144_mean_std.csv
│   └── main_summary_144_best_fixed.csv
└── scripts/
    ├── eval/                     # GSM8K / DialogSum / SQuAD v2 评估脚本
    ├── preprocess/               # 从 Hugging Face datasets 构建预算数据集
    ├── train/                    # SLURM 训练 sbatch + manifest 提交脚本
    ├── train_peft.py             # PEFT 训练核心入口（4 method 共用）
    ├── aggregate_main_results.py # 把 results/main/*_meta.json 汇总为 CSV
    ├── check_env.py              # 环境冒烟测试
    ├── gen_manifest_72.py        # 重新生成 72-run debug manifest
    └── min_forward.py            # base model 单次前向冒烟测试
```

## Final Report

`Final Report/` 是书面报告：

- `main0509.tex` —— 全文 LaTeX 源码，使用标准的 `article` class，`booktabs`、`subcaption`、`hyperref`、`amsmath` 等常见 package。
- `references.bib` —— BibTeX 引用，覆盖 LoRA、QLoRA、DoRA、IA3、Prefix/Prompt Tuning、GSM8K、DialogSum、SQuAD v2，以及 Gemma-2、Qwen2.5、Transformers、PEFT library、ROUGE。
- `Final Report/figures/`
  - `Workflow.png` —— 报告 Methodology 章节开头的总流程图，覆盖 data → backbones → PEFT methods → training config → execution → evaluation → analysis 七个阶段。
  - `fig1_main_results_bar.png` … `fig10_cost_summary.png` —— 144-run 主实验和效率分析图。
  - `fig11_base_vs_peft_delta.png`、`fig12_base_vs_peft_absolute.png` —— base model 对比图（仅在报告里使用，不在仓库根 `figures/` 中）。

编译报告：

```bash
cd "Final Report"
pdflatex main0509.tex
bibtex main0509
pdflatex main0509.tex
pdflatex main0509.tex
```

> 注意：报告中 Methodology 章节里 `Table~\ref{tab:hyperparams}` 给出的训练超参数（LoRA `r=16, α=32, dropout=0.05`；所有 7 个 linear 模块作为 LoRA / QLoRA / DoRA target；IA3 = `W_k, W_v, W_down` 等）和 `scripts/train_peft.py` 的真实行为已经显式对齐。`Workflow.png` 内嵌的数字属于示意，以表为准（caption 已经做了相应提示）。

## Data

`data/` 中已经包含本次交接所需的 JSONL 数据文件：

- `data/gsm8k/`
  - `train_128.jsonl`、`train_512.jsonl`、`train_1024.jsonl`、`train_2048.jsonl`
  - `test.jsonl`
- `data/dialogsum/`
  - `train_128.jsonl`、`train_512.jsonl`、`train_1024.jsonl`、`train_2048.jsonl`
  - `validation.jsonl`、`test.jsonl`
- `data/squad_v2/`
  - `train_128.jsonl`、`train_512.jsonl`、`train_1024.jsonl`、`train_2048.jsonl`
  - `validation.jsonl`

每条训练样本都包含一个 `text` 字段，使用统一的 instruction-response prompt 模板。评估脚本会读取任务相关的原始字段，例如 `question`、`answer`、`dialogue`、`summary`、`context`。

如果需要从原始 Hugging Face 数据集重新生成数据：

```bash
python scripts/preprocess/build_gsm8k_budget.py
python scripts/preprocess/build_dialogsum_budget.py
python scripts/preprocess/build_squadv2_budget.py
```

这些脚本中的默认输出路径是 `~/projects/dsaa_peft/data/...`。如果在其他机器上运行，请先按需修改各脚本里的 `OUT_DIR`。

## Results

主要结果文件位于 `results/`：

- `results/main/`
  - 主实验的 per-run 文件。
  - 每个配置通常包含一个 `*_eval200.json` 和一个 `*_meta.json`。
  - `*_eval200.json` 保存指标和逐样本预测。
  - `*_meta.json` 合并了模型、任务、方法、预算、种子、学习率、训练耗时、峰值显存、可训练参数比例和评估指标。
- `results/main_summary_144.csv`
  - 144 个主实验单次运行结果明细。
- `results/main_summary_144_mean_std.csv`
  - 按 `model × task × method × budget` 聚合两个随机种子的均值和标准差。
- `results/main_summary_144_best_fixed.csv`
  - 每个 `model × task × budget` 下的最佳方法。
- `results/base_model/base_model_summary.csv`
  - base model 在三个任务上的无微调基线结果。
- `results/base_model/base_vs_best_peft.csv`
  - base model 与最佳 PEFT 方法的逐 task 对比。

当前最佳 PEFT 相比 base model 的汇总（来自 `results/base_model/base_vs_best_peft.csv`）：

- Gemma + DialogSum：base ROUGE-L `0.2506`，最佳 QLoRA `0.3243`，提升 `+0.0737`
- Gemma + GSM8K：base EM `0.0100`，最佳 DoRA `0.0750`，提升 `+0.0650`
- Gemma + SQuAD v2：base F1 `0.5418`，最佳 LoRA `0.5904`，提升 `+0.0486`
- Qwen + DialogSum：base ROUGE-L `0.1875`，最佳 DoRA `0.2297`，提升 `+0.0421`
- Qwen + GSM8K：base EM `0.0150`，最佳 QLoRA `0.0350`，提升 `+0.0200`
- Qwen + SQuAD v2：base F1 `0.1099`，最佳 LoRA `0.1262`，提升 `+0.0163`

## Figures

仓库里有两份图表，互为镜像但用途不同：

- 仓库根 `figures/`：由 `figures/visualize.py` 在本地生成，附带 `figures_explanation.md` 的中文说明，命名为 `fig1_main_results_bar.png` … `fig10_cost_summary.png`。
- `Final Report/figures/`：实际嵌入论文的版本，多包含 `Workflow.png`、`fig11_base_vs_peft_delta.png`、`fig12_base_vs_peft_absolute.png`。

主要图表清单：

- `fig1_main_results_bar.png`：主实验分组柱状图
- `fig2_performance_vs_budget.png`：性能随数据预算变化
- `fig3_vram_usage.png`：峰值显存对比
- `fig4_training_time.png`：训练时间对比
- `fig5_best_method_heatmap.png`：每个配置下的最佳方法
- `fig6_efficiency_scatter.png`：性能与资源成本散点图
- `fig7_trainable_params.png`：可训练参数比例
- `fig8_method_ranking.png`：方法平均排名
- `fig9_score_heatmap.png`：所有配置得分热力图
- `fig10_cost_summary.png`：成本汇总
- `fig11_base_vs_peft_delta.png`、`fig12_base_vs_peft_absolute.png`：base vs PEFT 对比（仅在 `Final Report/figures/`）
- `Workflow.png`：报告 Methodology 章节的总流程图（仅在 `Final Report/figures/`）

`figures/visualize.py` 默认从当前工作目录读取 `main_summary_144.csv`、`main_summary_144_mean_std.csv` 和 `main_summary_144_best_fixed.csv`。如果从仓库根运行，可以先把 CSV 复制到 `figures/`，或调整脚本中的 CSV 路径。

## Scripts

### Environment Checks

- `scripts/check_env.py` 检查 PyTorch / CUDA、Tokenizer、Datasets 和 PEFT 是否可用。
- `scripts/min_forward.py` 加载 Qwen base model 并做一次最小生成测试。

```bash
python scripts/check_env.py
python scripts/min_forward.py
```

### Training

核心训练入口是 `scripts/train_peft.py`（注意：在当前 handoff 仓库里它位于 `scripts/` 根下，不是 `scripts/train/` 中）。它支持：

- `--method lora`
- `--method qlora`
- `--method dora`
- `--method ia3`

示例单次训练：

```bash
python scripts/train_peft.py \
  --method lora \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --train_file data/gsm8k/train_128.jsonl \
  --output_dir outputs/qwen_gsm8k_lora_b128_s42 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --seed 42
```

训练完成后，脚本会在 `output_dir` 中保存 adapter、tokenizer 和 `train_summary.json`。该 summary 后续会被 meta 汇总脚本读取。脚本默认配置和论文表 `tab:hyperparams` 已经对齐：LoRA / QLoRA / DoRA 的 target modules 是 `q/k/v/o/gate/up/down` 七个 linear；IA3 的 target 是 `k_proj, v_proj, down_proj` 并把 `down_proj` 设为 feedforward module；优化器是默认 AdamW（无 warmup、无 weight decay、linear 衰减）。

### Evaluation

任务评估脚本位于 `scripts/eval/`：

- `eval_gsm8k.py`：提取预测和答案中的最后一个数字，计算 EM。
- `eval_dialogsum.py`：生成摘要，计算 ROUGE-L。
- `eval_squadv2.py`：生成答案 span 或 `unanswerable`，计算 EM / F1（SQuAD 官方 normalize 规则）。

示例：

```bash
python scripts/eval/eval_gsm8k.py \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --adapter_path outputs/qwen_gsm8k_lora_b128_s42 \
  --test_file data/gsm8k/test.jsonl \
  --output_file results/main/qwen_gsm8k_lora_b128_s42_eval200.json \
  --max_examples 200 \
  --batch_size 4 \
  --max_new_tokens 32
```

`--adapter_path none` 或不传 adapter 时，可用于评估 base model。所有评估都使用 greedy decoding、左 padding，固定 200 例评估子集。

### SLURM Jobs

仓库包含若干面向 HPC / SLURM 的脚本：

- `scripts/train/make_8_main_jobs.sh`
  - 生成 `scripts/train/main_jobs/` 下的 8 个主实验 sbatch 文件。
  - 覆盖 `2 models × 4 PEFT methods`。
- `scripts/train/main_jobs/*.sbatch`
  - 每个文件负责固定的 `model × method`，循环运行所有 `task × budget × seed`。
- `scripts/train/one_train_debug72.sbatch`
  - 用于 72-run debug manifest 的单行训练。
- `scripts/eval/one_eval_debug72.sbatch`
  - 用于 72-run debug manifest 的单行评估。
- `scripts/train/submit_train_batch_72.sh`
  - 提交 manifest 的一个闭区间切片作为训练任务。
- `scripts/eval/submit_eval_batch_72.sh`
  - 提交 manifest 的一个闭区间切片作为评估任务。

```bash
bash scripts/train/submit_train_batch_72.sh 0 71
bash scripts/eval/submit_eval_batch_72.sh 0 71
```

注意事项：

- 这些 SLURM 脚本里写死了集群路径 `/hpc2hdd/home/mliu954/projects/dsaa_peft`、conda 环境 `peft_llm`、CUDA module 和分区名，迁移到其他机器时需要先统一替换。
- sbatch 文件内部仍按原 HPC 项目布局调用 `${ROOT}/scripts/train/train_peft.py`；当前 handoff 仓库中训练脚本已经移到 `scripts/train_peft.py`。如果想直接用本仓库重跑 SLURM 任务，请在 sbatch 模板里把这一行改成 `${ROOT}/scripts/train_peft.py`，或者把训练脚本复制/软链回 `scripts/train/`。

## Manifest

`manifests/main_debug_72.csv` 是 72-run debug sweep 的配置表，字段：

- `run_id`、`model_tag`、`model_name`、`method`、`task`
- `budget`、`seed`、`lr`、`epochs`、`max_length`

`scripts/gen_manifest_72.py` 可重新生成该文件。当前 debug manifest 使用 `128 / 512 / 1024` 三个预算，和主实验的 `128 / 512 / 2048` 不完全相同，因为 debug 跑在受限的 GPU 时间内。

## Aggregation

`scripts/aggregate_main_results.py` 会读取 `results/main/*_meta.json` 并输出扁平 CSV。脚本里默认路径是：

- 输入：`/hpc2hdd/home/mliu954/projects/dsaa_peft/results/main`
- 输出：`/hpc2hdd/home/mliu954/projects/dsaa_peft/results/main_summary.csv`

如果在本机运行，请先把 `BASE` 和 `OUT_CSV` 改成相对路径或当前机器的绝对路径。

## Expected Environment

主要 Python 依赖包括：

- `torch`
- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`
- `accelerate`
- `tqdm`
- `rouge-score`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

训练 QLoRA 需要可用的 CUDA GPU 和 bitsandbytes。SLURM 脚本假设集群上存在 `anaconda3` module、`cuda/12.8` module，以及名为 `peft_llm` 的 conda 环境。报告编译需要本机有 `pdflatex` 和 `bibtex`。

## Not Included

为了控制仓库体积，以下内容被有意排除：

- `outputs/`
- `outputs_72/`
- `logs/`
- `results_backup/`
- checkpoint 目录、optimizer / scheduler state
- adapter `.safetensors`
- Hugging Face 本地缓存

`.gitignore` 已配置上述规则。`results/main/` 和 `results/base_model/` 中保留的是评估 JSON、meta JSON 和汇总 CSV，不包含大模型权重或 adapter 权重。

## Handoff Notes

- 这是一个结果复核与复现实验用仓库，不是打包好的 Python library。
- 多个脚本中保留了原 HPC 项目的绝对路径；迁移运行前请统一替换。
- `scripts/train_peft.py` 在本次交接中被搬到了 `scripts/` 根下；所有 sbatch 模板仍按旧路径 `scripts/train/train_peft.py` 调用，需要时请同步修改。
- `data/`、`results/` 和 `Final Report/` 都已经随仓库保留，足够用于检查结果、重新画图、重新编译报告和做轻量分析。
- 完整重跑训练需要重新下载 base model 权重，并在本机或集群上准备 GPU 环境。
