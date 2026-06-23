"""
14_query_logs.py — 查询 / 聚合 JSONL 日志
==========================================
特点：
  - 不一次性把整个 JSONL 读进内存（流式逐行处理）
  - 支持 .jsonl 和 .jsonl.gz
  - 输出：总数/平均延迟/按小时分布/最近 N 条
"""
import sys, json, gzip, argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


def iter_events(log_dir: str):
    """流式遍历所有日志文件（不解压全部到内存）"""
    log_path = Path(log_dir)
    files = sorted(log_path.glob("events-*.jsonl*"), reverse=True)  # 最新的优先
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def stats(log_dir: str) -> dict:
    """聚合统计（O(N) 时间、O(K) 内存，K=分组数）"""
    counters = {
        "total": 0,
        "by_type": Counter(),
        "by_hour": Counter(),
        "by_tool": Counter(),
        "total_tokens": 0,
        "total_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "errors": 0,
    }
    for ev in iter_events(log_dir):
        counters["total"] += 1
        t = ev.get("type", "?")
        counters["by_type"][t] += 1
        ts = ev.get("ts", "")[:13]  # YYYY-MM-DDTHH
        counters["by_hour"][ts] += 1
        if t == "tool_start" and "tool" in ev:
            counters["by_tool"][ev["tool"]] += 1
        if "tokens" in ev:
            counters["total_tokens"] += ev["tokens"]
        if "latency_ms" in ev:
            lat = ev["latency_ms"]
            counters["total_latency_ms"] += lat
            if lat > counters["max_latency_ms"]:
                counters["max_latency_ms"] = lat
        if t == "error":
            counters["errors"] += 1

    n = counters["total"]
    counters["avg_latency_ms"] = round(counters["total_latency_ms"] / n, 2) if n else 0
    counters["by_type"] = dict(counters["by_type"])
    counters["by_hour"] = dict(sorted(counters["by_hour"].items()))
    counters["by_tool"] = dict(counters["by_tool"])
    counters["max_latency_ms"] = round(counters["max_latency_ms"], 2)
    return counters


def tail(log_dir: str, n: int = 20) -> list:
    """最近 n 条（流式，deque 限制内存）"""
    from collections import deque
    buf = deque(maxlen=n)
    for ev in iter_events(log_dir):
        buf.append(ev)
    return list(buf)


def disk_usage(log_dir: str) -> dict:
    """按文件统计磁盘占用"""
    log_path = Path(log_dir)
    files = []
    for f in sorted(log_path.glob("events-*.jsonl*")):
        files.append({
            "file": f.name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 3),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
        })
    return {
        "total_files": len(files),
        "total_mb": round(sum(f["size_mb"] for f in files), 3),
        "files": files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print(f"📊 日志聚合报告  ({args.log_dir})")
    print("=" * 60)
    s = stats(args.log_dir)
    print(f"总事件数     ：{s['total']}")
    print(f"总 Token     ：{s['total_tokens']:,}")
    print(f"平均延迟(ms) ：{s['avg_latency_ms']}")
    print(f"最大延迟(ms) ：{s['max_latency_ms']}")
    print(f"错误数       ：{s['errors']}")
    print(f"\n按类型分布：")
    for k, v in sorted(s["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {k:15s} {v:>5}")
    if s["by_tool"]:
        print(f"\n工具调用 TOP：")
        for k, v in sorted(s["by_tool"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {k:25s} {v:>5}")
    if s["by_hour"]:
        print(f"\n按小时分布（最近）：")
        for h, c in list(s["by_hour"].items())[-12:]:
            print(f"  {h}  {'#' * min(c, 50)} {c}")

    print(f"\n磁盘占用：")
    d = disk_usage(args.log_dir)
    print(f"  文件数：{d['total_files']}  总大小：{d['total_mb']} MB")
    for f in d["files"][-10:]:
        print(f"  {f['file']:40s} {f['size_mb']:>8} MB  {f['modified']}")

    print(f"\n最近 {args.tail} 条事件：")
    for ev in tail(args.log_dir, args.tail):
        print(f"  [{ev.get('ts','')}] {ev.get('type',''):12s} run={ev.get('run_id','-')}")
        if "latency_ms" in ev:
            print(f"      latency={ev['latency_ms']}ms tokens={ev.get('tokens','-')}")
        elif "tool" in ev:
            print(f"      tool={ev['tool']}")
