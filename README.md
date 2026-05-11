# DSAA PEFT Handoff

这是 DSAA PEFT 项目的清理交接版仓库，用于代码审阅、结果复核、图表重现，以及在相同数据布局下重新运行实验。项目比较四种参数高效微调方法在三个指令微调任务上的表现与资源成本。

研究设置：

- 模型：`Qwen/Qwen2.5-3B-Instruct`、`google/gemma-2-2b-it`
- 方法：LoRA、QLoRA、DoRA、IA3
- 任务：GSM8K（EM）、SQuAD v2（F1 / EM）、DialogSum（ROUGE-L）
- 数据预算：`128`、`512`、`2048` 条训练样本
- 主实验种子：`42`、`43`
- 每个评估文件默认记录前 `200` 个测试/验证样本的预测结果

## Repository Layout

```text
.
├── data/
│   ├── dialogsum/          # train_*.jsonl, validation.jsonl, test.jsonl
│   ├── gsm8k/              # train_*.jsonl, test.jsonl
│   └── squad_v2/           # train_*.jsonl, validation.jsonl
├── figures/                # 论文/报告用图表、图表说明和可视化脚本
├── manifests/              # 批量实验 manifest
├── results/
│   ├── base_model/         # base model zero/few-shot style evaluation outputs
│   ├── main/               # 144 个主实验的 eval/meta JSON
│   └── main_summary_*.csv  # 主实验汇总表
└── scripts/
    ├── eval/               # GSM8K / DialogSum / SQuAD v2 评估脚本
    ├── preprocess/         # 从 Hugging Face datasets 构建预算数据集
    ├── train/              # PEFT 训练脚本与 SLURM 提交脚本
    ├── aggregate_main_results.py
    ├── check_env.py
    ├── gen_manifest_72.py
    └── min_forward.py
```

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

每条训练样本都包含一个 `text` 字段，格式为统一的 instruction-response prompt。评估脚本会读取任务相关的原始字段，例如 `question`、`answer`、`dialogue`、`summary`、`context`。

如果需要从原始 Hugging Face 数据集重新生成数据：

```bash
python scripts/preprocess/build_gsm8k_budget.py
python scripts/preprocess/build_dialogsum_budget.py
python scripts/preprocess/build_squadv2_budget.py
```

这些脚本中的默认输出路径是 `~/projects/dsaa_peft/data/...`。如果在其他机器上运行，请先按需修改 `OUT_DIR`。

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
  - base model 在三个任务上的基线结果。
- `results/base_model/base_vs_best_peft.csv`
  - base model 与最佳 PEFT 方法的对比。

当前最佳 PEFT 相比 base model 的汇总结果在 `results/base_model/base_vs_best_peft.csv` 中：

- Gemma + DialogSum：base ROUGE-L `0.2506`，最佳 QLoRA `0.3243`，提升 `+0.0737`
- Gemma + GSM8K：base EM `0.0100`，最佳 DoRA `0.0750`，提升 `+0.0650`
- Gemma + SQuAD v2：base F1 `0.5418`，最佳 LoRA `0.5904`，提升 `+0.0486`
- Qwen + DialogSum：base ROUGE-L `0.1875`，最佳 DoRA `0.2297`，提升 `+0.0421`
- Qwen + GSM8K：base EM `0.0150`，最佳 QLoRA `0.0350`，提升 `+0.0200`
- Qwen + SQuAD v2：base F1 `0.1099`，最佳 LoRA `0.1262`，提升 `+0.0163`

## Figures

`figures/` 中包含最终图表和说明：

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
- `figures_explanation.md`：每张图的布局、用途和主要发现
- `visualize.py`：生成图表的脚本

`figures/visualize.py` 默认从当前工作目录读取 `main_summary_144.csv`、`main_summary_144_mean_std.csv` 和 `main_summary_144_best_fixed.csv`。如果从仓库根目录运行，可以先复制或调整脚本中的 CSV 路径。

## Scripts

### Environment Checks

- `scripts/check_env.py`
  - 检查 PyTorch / CUDA、Tokenizer、Datasets 和 PEFT 是否可用。
- `scripts/min_forward.py`
  - 加载 Qwen base model 并做一次最小生成测试。

示例：

```bash
python scripts/check_env.py
python scripts/min_forward.py
```

### Training

核心训练入口是 `scripts/train/train_peft.py`。它支持：

- `--method lora`
- `--method qlora`
- `--method dora`
- `--method ia3`

示例单次训练：

```bash
python scripts/train/train_peft.py \
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

训练完成后，脚本会在 `output_dir` 中保存 adapter、tokenizer 和 `train_summary.json`。该 summary 后续会被 meta 汇总脚本读取。

### Evaluation

任务评估脚本位于 `scripts/eval/`：

- `eval_gsm8k.py`：提取预测与答案中的最后一个数字，计算 EM。
- `eval_dialogsum.py`：生成摘要，计算 ROUGE-L。
- `eval_squadv2.py`：生成答案 span 或 `unanswerable`，计算 EM / F1。

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

`--adapter_path none` 或不传 adapter 时，可用于评估 base model。

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

示例：

```bash
bash scripts/train/submit_train_batch_72.sh 0 71
bash scripts/eval/submit_eval_batch_72.sh 0 71
```

注意：这些 SLURM 脚本大多写死了集群路径 `/hpc2hdd/home/mliu954/projects/dsaa_peft`、conda 环境 `peft_llm`、CUDA module 和分区名。迁移到其他机器时需要先修改这些路径和调度配置。

## Manifest

`manifests/main_debug_72.csv` 是 72-run debug sweep 的配置表，字段包括：

- `run_id`
- `model_tag`
- `model_name`
- `method`
- `task`
- `budget`
- `seed`
- `lr`
- `epochs`
- `max_length`

`scripts/gen_manifest_72.py` 可重新生成该文件。当前 debug manifest 使用 `128 / 512 / 1024` 三个预算，和主实验的 `128 / 512 / 2048` 不完全相同。

## Aggregation

`scripts/aggregate_main_results.py` 会读取 `results/main/*_meta.json` 并输出扁平 CSV。脚本中默认路径是：

- 输入：`/hpc2hdd/home/mliu954/projects/dsaa_peft/results/main`
- 输出：`/hpc2hdd/home/mliu954/projects/dsaa_peft/results/main_summary.csv`

如果在当前 handoff 仓库中本地运行，请先把 `BASE` 和 `OUT_CSV` 改成相对路径或当前机器的绝对路径。

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

训练 QLoRA 需要可用的 CUDA GPU 和 bitsandbytes。SLURM 脚本假设集群上存在 `anaconda3` module、`cuda/12.8` module，以及名为 `peft_llm` 的 conda 环境。

## Not Included

为了控制仓库体积，以下内容被有意排除：

- `outputs/`
- `outputs_72/`
- `logs/`
- `results_backup/`
- checkpoint 目录
- optimizer / scheduler states
- adapter `.safetensors`
- Hugging Face 本地缓存

`.gitignore` 已配置上述规则。`results/main/` 和 `results/base_model/` 中保留的是评估 JSON、meta JSON 和汇总 CSV，不包含大模型权重或 adapter 权重。

## Handoff Notes

- 这是一个结果复核与复现实验用仓库，不是打包好的 Python library。
- 多个脚本中的路径仍保留原 HPC 项目路径；迁移运行前请先统一修改路径。
- `data/` 和 `results/` 已随仓库保留，足够用于检查结果、重新画图和做轻量分析。
- 若要完整重跑训练，需要重新下载 base model 权重，并在本机或集群上准备 GPU 环境。
