# 三模型回答质量与性价比横评

> 一套可在本地复跑的 LLM 质量与成本横评工具。

Contact: **Jacksun** · [qinji@jack-sun.com](mailto:qinji@jack-sun.com)

![model-quality-benchmark project visual](docs/assets/model-quality-benchmark-hero.png)

它提供通用创意、项目策划、沟通、运维与提示词设计任务，用于比较模型回答质量、延迟与预估成本。

本目录是一套本地可复跑的 LLM 横评工具，用来对比：

- DeepSeek V4 Flash
- DeepSeek V4 Pro
- GPT-5.5

评测重点不是“谁最强”，而是：**Jack 的真实工作流里，哪个模型最划算。**

## 1. 准备 Key

```bash
cd model-quality-benchmark
cp .env.local.example .env.local
```

然后把真实 Key 填进 `.env.local`。不要把 `.env.local` 发给别人，也不要提交。

## 2. 如有必要，修正模型 ID / 价格

编辑：

```bash
config/models.json
```

重点确认：

- `model` 是否是供应商真实模型 ID
- `price_input_per_1m` 每百万输入 token 价格
- `price_output_per_1m` 每百万输出 token 价格

价格不填也能跑，但报告里的成本/性价比只会显示 0 或无法计算。

## 3. 跑评测

```bash
python3 scripts/run_benchmark.py run \
  --models config/models.json \
  --tasks tasks/jack_workflows.jsonl \
  --out outputs/run_001
```

只跑某一个模型测试连通性：

```bash
python3 scripts/run_benchmark.py run --only deepseek-v4-flash --limit 1 --out outputs/smoke
```

## 4. 人工评分

跑完后会生成：

```bash
outputs/run_001/manual_scores.csv
```

按 1–5 分填写：

- usefulness
- accuracy
- expression
- jack_fit
- actionability

然后生成报告：

```bash
python3 scripts/run_benchmark.py report --run outputs/run_001
```

## 5. 可选：自动裁判评分

如果 `.env.local` 里配置了 `JUDGE_*`，可以：

```bash
python3 scripts/run_benchmark.py judge --run outputs/run_001
python3 scripts/run_benchmark.py report --run outputs/run_001
```

自动裁判适合初筛，但最终建议 Jack 人工抽查 Top / Bottom 样例。

## 输出文件

- `results.jsonl`：完整原始结果
- `results.csv`：结构化结果
- `outputs_by_model.md`：所有模型回答，方便人工盲评/复盘
- `manual_scores.csv`：人工评分表
- `judge_scores.csv`：自动裁判评分表，可选
- `report.md`：最终横评报告
