# LLM Classification Finetuning

一个面向学习的 Kaggle 项目仓库，用来完成 **LLM Classification Finetuning** 的端到端练习：数据理解、TF-IDF baseline、验证、Transformer 微调和 `submission.csv` 生成。

## 项目目标

- 用两周节奏跑完整个 Kaggle 学习闭环。
- 本地维护代码、文档和实验记录。
- 在 Kaggle Notebook 中访问比赛数据并使用 GPU 训练。
- 不提交 Kaggle 原始数据、账号 token、模型权重或大型输出文件。

## 目录结构

```text
.
├── configs/              # YAML 实验配置
├── data/
│   ├── raw/              # Kaggle 原始数据占位，默认不提交内容
│   ├── processed/        # 中间数据占位，默认不提交内容
│   └── sample/           # 合成小样本，用于本地冒烟测试
├── docs/                 # 学习路线、实验记录、Kaggle Notebook 说明
├── notebooks/            # EDA、baseline、finetune notebooks
├── outputs/              # 指标、预测等生成物，默认不提交内容
├── src/                  # 可复用训练/预测代码
├── submissions/          # Kaggle 提交文件，默认不提交内容
└── tests/                # 标准库 unittest 测试
```

## 快速开始

安装核心依赖：

```bash
python3 -m pip install -r requirements.txt
```

用合成小样本跑通本地训练与预测：

```bash
python3 -m src.train --config configs/local_sample.yaml
python3 -m src.predict --config configs/local_sample.yaml --output outputs/submission_sample.csv
```

在 Kaggle Notebook 中使用比赛数据时，默认配置读取：

```text
/kaggle/input/llm-classification-finetuning
```

训练 baseline：

```bash
python -m src.train --config configs/baseline.yaml
python -m src.predict --config configs/baseline.yaml --output submissions/submission.csv
```

在 Kaggle Notebook 中也可以用一条命令跑完测试、训练、预测和提交文件校验：

```bash
python scripts/run_kaggle_baseline.py
```

比赛提交版本必须关闭 Internet。最终提交时优先使用 `docs/no_internet_kaggle_templates.md` 中的单 cell 模板，并确保输出：

```text
/kaggle/working/submission.csv
```

在 Kaggle GPU 上完成 Transformer 微调后生成提交：

```bash
python -m src.finetune --config configs/finetune.yaml
python -m src.predict_finetune --config configs/finetune.yaml --output submissions/submission_finetune.csv
```

运行测试：

```bash
python3 -m unittest discover -s tests
```

## 学习路线

1. 阅读 `docs/learning_roadmap.md`，明确两周安排。
2. 运行 `notebooks/01_eda.ipynb`，理解数据字段、标签分布和文本长度。
3. 运行 `notebooks/02_baseline.ipynb`，完成 TF-IDF + Logistic Regression baseline。
4. 按 `docs/kaggle_baseline_runbook.md` 在 Kaggle 上提交第一版 baseline。
5. 运行 `notebooks/03_finetune_kaggle.ipynb`，在 Kaggle GPU 上做小规模 Transformer 微调。
6. 将每次实验结论写入 `docs/experiment_log.md`。

提分路线见 `docs/score_improvement_plan.md`。

## 安全约束

- 不提交 `data/raw/`、`data/processed/`、`outputs/`、`submissions/`、`models/` 中的生成内容。
- 不提交 `kaggle.json`、`.env` 或任何 API token。
- 大模型权重、训练 checkpoint 和 Kaggle submission 默认只保留本地或 Kaggle 环境。
- 提交前运行 `python3 -m unittest discover -s tests`，其中 notebook hygiene 测试会拒绝已执行输出，避免意外提交原始比赛文本。
