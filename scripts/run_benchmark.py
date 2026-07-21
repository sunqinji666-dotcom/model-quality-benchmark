#!/usr/bin/env python3
"""Jack 三模型回答质量与性价比横评工具。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request, error

ROOT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = ["usefulness", "accuracy", "expression", "jack_fit", "actionability"]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSONL 解析失败 {path}:{idx}: {exc}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def approx_tokens(text: str) -> int:
    # 粗略估算：中文字符 token 密度高，英文按 4 chars/token 近似。
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(0, len(text) - cjk)
    return max(1, int(cjk * 0.9 + other / 4))


def cost_yuan(input_tokens: int, output_tokens: int, model_cfg: Dict[str, Any]) -> float:
    in_price = float(model_cfg.get("price_input_per_1m") or 0)
    out_price = float(model_cfg.get("price_output_per_1m") or 0)
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int, extra_params: Optional[Dict[str, Any]] = None,
    stream: bool = True,
) -> Tuple[str, Dict[str, Any], Optional[float], float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    # 传额外参数（如 thinking=False）
    for k, v in (extra_params or {}).items():
        if v is not None:
            payload[k] = v

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    usage: Dict[str, Any] = {}
    first_token_at: Optional[float] = None
    chunks: List[str] = []

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            if not stream:
                body = json.loads(resp.read().decode("utf-8"))
                total = time.perf_counter() - started
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content, body.get("usage") or {}, total, total

            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - started
                    chunks.append(piece)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"网络错误: {exc}") from exc

    total = time.perf_counter() - started
    return "".join(chunks), usage, first_token_at, total


def system_prompt() -> str:
    return (
        "你是 Jack（孙秦吉）的长期 AI 工作伙伴。Jack 是桂林商业摄影师、导演、剪辑、光感影视工作室主理人，"
        "正在从商业客户项目转向个人内容创作。回答要用清晰中文，直接、实用、有行业判断，"
        "风格真实克制，避免空话、鸡血、PPT腔和过度宏大叙事。优先给具体可执行方法。"
    )


def run(args: argparse.Namespace) -> None:
    load_env_file(ROOT / ".env.local")
    cfg = read_json(Path(args.models))
    tasks = read_jsonl(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]

    models = cfg["models"]
    if args.only:
        only = set(args.only.split(","))
        models = [m for m in models if m["id"] in only]
    if not models:
        raise SystemExit("没有可运行的模型，请检查 --only 或 config/models.json")

    defaults = cfg.get("default_params", {})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for task in tasks:
        for m in models:
            key = os.environ.get(m["api_key_env"], "")
            row: Dict[str, Any] = {
                "task_id": task["id"],
                "category": task.get("category", ""),
                "title": task.get("title", ""),
                "model_id": m["id"],
                "model_label": m.get("label", m["id"]),
                "provider_model": m["model"],
                "prompt": task["prompt"],
                "ok": False,
                "error": "",
                "answer": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_yuan": 0.0,
                "ttft_seconds": None,
                "total_seconds": None,
            }
            print(f"▶ {task['id']} / {m['id']}", flush=True)
            if not key:
                row["error"] = f"缺少环境变量 {m['api_key_env']}"
                results.append(row)
                continue
            try:
                messages = [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": task["prompt"]},
                ]
                answer, usage, ttft, total = chat_completion(
                    base_url=m["base_url"],
                    api_key=key,
                    model=m["model"],
                    messages=messages,
                    temperature=float(defaults.get("temperature", 0.4)),
                    max_tokens=int(defaults.get("max_tokens", 1600)),
                    timeout_seconds=int(defaults.get("timeout_seconds", 180)),
                    stream=not args.no_stream,
                    extra_params={k: v for k, v in m.items() if k not in ("id","label","base_url","api_key_env","model","price_input_per_1m","price_output_per_1m","currency","notes")},
                )
                in_tok = int(usage.get("prompt_tokens") or approx_tokens(system_prompt() + task["prompt"]))
                out_tok = int(usage.get("completion_tokens") or approx_tokens(answer))
                row.update(
                    ok=True,
                    answer=answer,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_yuan=round(cost_yuan(in_tok, out_tok, m), 8),
                    ttft_seconds=round(ttft, 3) if ttft is not None else None,
                    total_seconds=round(total, 3),
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
            results.append(row)

    save_run_outputs(out_dir, results)
    print(f"\n完成：{out_dir}")
    print(f"- {out_dir / 'results.jsonl'}")
    print(f"- {out_dir / 'results.csv'}")
    print(f"- {out_dir / 'outputs_by_model.md'}")
    print(f"- {out_dir / 'manual_scores.csv'}")


def save_run_outputs(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(out_dir / "results.jsonl", rows)
    fieldnames = [
        "task_id", "category", "title", "model_id", "model_label", "provider_model", "ok", "error",
        "input_tokens", "output_tokens", "cost_yuan", "ttft_seconds", "total_seconds", "prompt", "answer",
    ]
    with (out_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    with (out_dir / "outputs_by_model.md").open("w", encoding="utf-8") as f:
        f.write("# 模型原始回答\n\n")
        for r in rows:
            f.write(f"## {r['task_id']}｜{r['category']}｜{r['model_label']}\n\n")
            f.write(f"**题目**：{r['title']}\n\n")
            f.write(f"**状态**：{'OK' if r['ok'] else 'ERROR'}\n\n")
            if r.get("error"):
                f.write(f"**错误**：`{r['error']}`\n\n")
            f.write("### Prompt\n\n")
            f.write(r.get("prompt", "") + "\n\n")
            f.write("### Answer\n\n")
            f.write((r.get("answer") or "") + "\n\n---\n\n")

    score_fields = ["task_id", "category", "title", "model_id", "model_label", *SCORE_FIELDS, "notes"]
    with (out_dir / "manual_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        for r in rows:
            if r.get("ok"):
                writer.writerow({
                    "task_id": r["task_id"], "category": r["category"], "title": r["title"],
                    "model_id": r["model_id"], "model_label": r["model_label"],
                    **{field: "" for field in SCORE_FIELDS}, "notes": "",
                })


def load_results(run_dir: Path) -> List[Dict[str, Any]]:
    return read_jsonl(run_dir / "results.jsonl")


def load_scores(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task_id"], row["model_id"])
            parsed: Dict[str, Any] = {"notes": row.get("notes", "")}
            complete = True
            for field in SCORE_FIELDS:
                val = str(row.get(field, "")).strip()
                if not val:
                    complete = False
                    break
                parsed[field] = float(val)
            if complete:
                parsed["quality_total"] = sum(parsed[f] for f in SCORE_FIELDS)
                scores[key] = parsed
    return scores


def choose_scores(run_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], str]:
    judge = load_scores(run_dir / "judge_scores.csv")
    if judge:
        return judge, "judge_scores.csv"
    manual = load_scores(run_dir / "manual_scores.csv")
    return manual, "manual_scores.csv"


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    rows = load_results(run_dir)
    scores, score_source = choose_scores(run_dir)
    enriched = []
    for r in rows:
        s = scores.get((r["task_id"], r["model_id"]), {})
        total = s.get("quality_total")
        cost = float(r.get("cost_yuan") or 0)
        seconds = float(r.get("total_seconds") or 0)
        enriched.append({
            **r,
            **s,
            "quality_total": total,
            "quality_per_yuan": (total / cost) if total is not None and cost > 0 else None,
            "quality_per_second": (total / seconds) if total is not None and seconds > 0 else None,
        })

    md = build_report(enriched, score_source)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(run_dir / "report.md")


def avg(nums: List[float]) -> Optional[float]:
    nums = [n for n in nums if n is not None]
    return sum(nums) / len(nums) if nums else None


def fmt(v: Any, digits: int = 2) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build_report(rows: List[Dict[str, Any]], score_source: str) -> str:
    ok_rows = [r for r in rows if r.get("ok")]
    scored = [r for r in ok_rows if r.get("quality_total") is not None]
    lines = ["# 三模型回答质量与性价比横评报告", ""]
    lines += [f"评分来源：`{score_source}`", ""]

    if not scored:
        lines += [
            "## 当前状态",
            "",
            "评测已完成调用，但还没有有效评分。请填写 `manual_scores.csv` 后运行：",
            "",
            "```bash",
            "python3 scripts/run_benchmark.py report --run outputs/run_001",
            "```",
            "",
        ]
        lines += failure_section(rows)
        return "\n".join(lines)

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in scored:
        by_model.setdefault(r["model_id"], []).append(r)

    lines += ["## 总榜", "", "| 模型 | 平均质量/25 | 平均耗时s | 平均成本¥ | 每元质量分 | 单位时间质量分 |", "|---|---:|---:|---:|---:|---:|"]
    model_summary = []
    for model_id, items in sorted(by_model.items()):
        q = avg([i["quality_total"] for i in items])
        sec = avg([float(i.get("total_seconds") or 0) for i in items])
        cost = avg([float(i.get("cost_yuan") or 0) for i in items])
        qpy = avg([i.get("quality_per_yuan") for i in items if i.get("quality_per_yuan") is not None])
        qps = avg([i.get("quality_per_second") for i in items if i.get("quality_per_second") is not None])
        label = items[0].get("model_label", model_id)
        model_summary.append((model_id, label, q or 0, cost or 0))
        lines.append(f"| {label} | {fmt(q)} | {fmt(sec)} | {fmt(cost, 6)} | {fmt(qpy)} | {fmt(qps)} |")
    lines.append("")

    lines += ["## 分任务推荐", "", "| 类别 | 题目 | 推荐模型 | 理由 |", "|---|---|---|---|"]
    task_ids = sorted({r["task_id"] for r in scored})
    for task_id in task_ids:
        items = [r for r in scored if r["task_id"] == task_id]
        best = max(items, key=lambda r: r["quality_total"])
        flash = next((r for r in items if "flash" in r["model_id"]), None)
        rec = best
        reason = f"质量最高：{best['quality_total']:.1f}/25"
        if flash and flash.get("quality_total") is not None and flash["quality_total"] >= best["quality_total"] * 0.85:
            best_cost = float(best.get("cost_yuan") or 0)
            flash_cost = float(flash.get("cost_yuan") or 0)
            if best_cost == 0 or flash_cost <= best_cost:
                rec = flash
                reason = f"Flash 达到最高分 {flash['quality_total']/best['quality_total']:.0%}，优先省钱"
        lines.append(f"| {items[0]['category']} | {items[0]['title']} | {rec['model_label']} | {reason} |")
    lines.append("")

    lines += ["## 按类别表现", "", "| 类别 | 模型 | 平均质量/25 | 平均成本¥ | 平均耗时s |", "|---|---|---:|---:|---:|"]
    categories = sorted({r["category"] for r in scored})
    for cat in categories:
        for model_id in sorted(by_model):
            items = [r for r in scored if r["category"] == cat and r["model_id"] == model_id]
            if not items:
                continue
            lines.append(
                f"| {cat} | {items[0]['model_label']} | {fmt(avg([i['quality_total'] for i in items]))} | "
                f"{fmt(avg([float(i.get('cost_yuan') or 0) for i in items]), 6)} | {fmt(avg([float(i.get('total_seconds') or 0) for i in items]))} |"
            )
    lines.append("")

    lines += ["## 典型样例", ""]
    best_examples = sorted(scored, key=lambda r: r["quality_total"], reverse=True)[:3]
    worst_examples = sorted(scored, key=lambda r: r["quality_total"])[:3]
    lines += ["### 高分样例", ""]
    for r in best_examples:
        lines.append(f"- {r['model_label']}｜{r['category']}｜{r['title']}：{r['quality_total']:.1f}/25")
    lines += ["", "### 低分/需谨慎样例", ""]
    for r in worst_examples:
        lines.append(f"- {r['model_label']}｜{r['category']}｜{r['title']}：{r['quality_total']:.1f}/25")
    lines.append("")
    lines += failure_section(rows)
    return "\n".join(lines)


def failure_section(rows: List[Dict[str, Any]]) -> List[str]:
    failed = [r for r in rows if not r.get("ok")]
    if not failed:
        return []
    lines = ["## 调用失败", "", "| 任务 | 模型 | 错误 |", "|---|---|---|"]
    for r in failed[:30]:
        err = str(r.get("error", "")).replace("|", "\\|")[:300]
        lines.append(f"| {r.get('task_id')} | {r.get('model_label')} | {err} |")
    if len(failed) > 30:
        lines.append(f"| ... | ... | 还有 {len(failed)-30} 条失败 |")
    lines.append("")
    return lines


def judge(args: argparse.Namespace) -> None:
    load_env_file(ROOT / ".env.local")
    run_dir = Path(args.run)
    rows = [r for r in load_results(run_dir) if r.get("ok")]
    judge_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("RIGHT_CODES_API_KEY")
    judge_base = os.environ.get("JUDGE_BASE_URL", "https://www.right.codes/codex/v1")
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-5.5")
    if not judge_key:
        raise SystemExit("缺少 JUDGE_API_KEY 或 RIGHT_CODES_API_KEY，无法自动裁判。")

    out_path = run_dir / "judge_scores.csv"
    fieldnames = ["task_id", "category", "title", "model_id", "model_label", *SCORE_FIELDS, "notes"]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            print(f"⚖ {r['task_id']} / {r['model_id']}", flush=True)
            prompt = build_judge_prompt(r)
            try:
                text, _, _, _ = chat_completion(
                    base_url=judge_base,
                    api_key=judge_key,
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                    timeout_seconds=180,
                    stream=False,
                )
                parsed = extract_json_object(text)
                row = {k: r.get(k, "") for k in ["task_id", "category", "title", "model_id", "model_label"]}
                for field in SCORE_FIELDS:
                    row[field] = clamp_score(parsed.get(field))
                row["notes"] = str(parsed.get("notes", ""))[:300]
                writer.writerow(row)
            except Exception as exc:  # noqa: BLE001
                writer.writerow({
                    "task_id": r["task_id"], "category": r["category"], "title": r["title"],
                    "model_id": r["model_id"], "model_label": r["model_label"],
                    **{field: "" for field in SCORE_FIELDS}, "notes": f"judge_error: {exc}",
                })
    print(out_path)


def build_judge_prompt(r: Dict[str, Any]) -> str:
    return f"""
你是严格的中文内容评测员。请按 1-5 分评价这个模型回答，必须只输出 JSON。

评分维度：
- usefulness: 是否真正解决问题
- accuracy: 事实、逻辑、步骤是否靠谱
- expression: 中文自然度、结构、节奏
- jack_fit: 是否符合 Jack 的商业影像/B站转型/克制真实风格
- actionability: 是否有具体执行感

任务类别：{r['category']}
任务标题：{r['title']}
用户问题：
{r['prompt']}

模型回答：
{r['answer']}

只输出 JSON，格式：
{{"usefulness":1-5,"accuracy":1-5,"expression":1-5,"jack_fit":1-5,"actionability":1-5,"notes":"一句话说明"}}
""".strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"未找到 JSON: {text[:200]}")
    return json.loads(match.group(0))


def clamp_score(v: Any) -> int:
    try:
        n = int(round(float(v)))
    except Exception:
        return ""
    return max(1, min(5, n))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Jack 三模型回答质量与性价比横评")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="调用模型并保存原始结果")
    p_run.add_argument("--models", default=str(ROOT / "config/models.json"))
    p_run.add_argument("--tasks", default=str(ROOT / "tasks/jack_workflows.jsonl"))
    p_run.add_argument("--out", default=str(ROOT / "outputs/run_001"))
    p_run.add_argument("--only", help="只跑指定模型 id，多个用逗号分隔")
    p_run.add_argument("--limit", type=int, help="只跑前 N 道题，用于 smoke test")
    p_run.add_argument("--no-stream", action="store_true", help="关闭流式，部分供应商兼容性更好")
    p_run.set_defaults(func=run)

    p_judge = sub.add_parser("judge", help="用裁判模型自动评分")
    p_judge.add_argument("--run", default=str(ROOT / "outputs/run_001"))
    p_judge.set_defaults(func=judge)

    p_report = sub.add_parser("report", help="生成 Markdown 横评报告")
    p_report.add_argument("--run", default=str(ROOT / "outputs/run_001"))
    p_report.set_defaults(func=report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
