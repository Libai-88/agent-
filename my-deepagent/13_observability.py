"""
13_observability.py — 自建观测：流式 JSONL 日志
================================================
替代 LangSmith 的本地方案。设计目标：
  - 内存占用：O(1)（只维护几个计数器，不缓存事件）
  - 存储占用：可配置上限（默认 50MB × 5 个文件 ≈ 250MB）
  - 写入：每事件立即 flush，进程崩溃也不丢

用法：
  from observability import JsonlLogger, LangChainCallback
  logger = JsonlLogger("./logs")
  cb = LangChainCallback(logger, run_id="abc")
  agent.invoke(..., config={"callbacks": [cb]})
"""
import os, json, gzip, time, threading, glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlLogger:
    """
    流式 JSONL 日志器，带自动轮转 + 自动清理 + 可选压缩。

    文件结构：
        logs/
            events-2026-06-23.jsonl       # 今日
            events-2026-06-23.jsonl.1     # 上一个轮转（最大）
            events-2026-06-23.jsonl.2.gz  # 早期，已压缩
            events-2026-06-22.jsonl.gz    # 昨天
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        max_bytes: int = 50 * 1024 * 1024,   # 50 MB / 文件
        max_files: int = 5,                   # 最多保留 5 个轮转
        retention_days: int = 7,              # 7 天前自动删
        compress_rotated: bool = True,        # 旧文件 gzip
        sampling_rate: float = 1.0,           # 1.0 = 全采样；0.1 = 10%
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.retention_days = retention_days
        self.compress_rotated = compress_rotated
        self.sampling_rate = sampling_rate

        # 内存里只维护几个计数器（O(1) 内存）
        self._lock = threading.Lock()
        self._counters = {
            "total": 0,
            "by_type": {},     # {"llm": 0, "tool": 0, "chain": 0}
            "total_tokens": 0,
            "total_latency_ms": 0,
            "max_latency_ms": 0,
        }

        # 当前文件 handle（延迟打开）
        self._current_date = None
        self._current_file = None
        self._current_size = 0

        # 后台清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    # ============== 公共 API ==============
    def log(self, event_type: str, data: Dict[str, Any], run_id: Optional[str] = None) -> None:
        """记录一条事件（流式写入，立即 flush）"""
        if self.sampling_rate < 1.0 and event_type not in ("error",):
            # 降采样
            import random
            if random.random() > self.sampling_rate:
                return

        event = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": event_type,
            "run_id": run_id,
            **data,
        }
        line = json.dumps(event, ensure_ascii=False) + "\n"

        with self._lock:
            self._ensure_file_open()
            self._current_file.write(line)
            self._current_file.flush()           # 关键：立即刷盘
            self._current_size += len(line.encode("utf-8"))
            self._update_counters(event_type, data)

            # 触发轮转
            if self._current_size >= self.max_bytes:
                self._rotate()

    def stats(self) -> Dict[str, Any]:
        """返回当前内存里的统计（O(1) 时间）"""
        with self._lock:
            n = self._counters["total"]
            avg = self._counters["total_latency_ms"] / n if n else 0
            return {
                **self._counters,
                "avg_latency_ms": round(avg, 2),
                "current_log_file": str(self._current_file.name) if self._current_file else None,
                "current_log_size_mb": round(self._current_size / 1024 / 1024, 2),
            }

    def disk_usage_mb(self) -> float:
        """返回 logs/ 目录总大小（MB）"""
        total = sum(f.stat().st_size for f in self.log_dir.rglob("*") if f.is_file())
        return round(total / 1024 / 1024, 2)

    def tail(self, n: int = 20) -> list:
        """读最后 n 条（不加载整个文件到内存）"""
        path = self._current_file_path()
        if not path.exists():
            return []
        # 用 deque 控制内存
        from collections import deque
        result = deque(maxlen=n)
        with open(path) as f:
            for line in f:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return list(result)

    # ============== 内部 ==============
    def _current_file_path(self) -> Path:
        return self.log_dir / f"events-{datetime.now():%Y-%m-%d}.jsonl"

    def _ensure_file_open(self) -> None:
        today = datetime.now().date()
        path = self._current_file_path()
        if self._current_date != today or self._current_file is None:
            if self._current_file:
                self._current_file.close()
            self._current_file = open(path, "a", buffering=1)  # line-buffered
            self._current_date = today
            self._current_size = path.stat().st_size if path.exists() else 0

    def _rotate(self) -> None:
        """轮转：当前文件 → .N（可选压缩）"""
        if not self._current_file:
            return
        self._current_file.close()
        src = self._current_file_path()

        # 已有轮转编号 .1, .2, ...，把 .N-1 改名 .N
        for i in range(self.max_files, 0, -1):
            old = self.log_dir / f"{src.name}.{i}"
            new = self.log_dir / f"{src.name}.{i+1}"
            if old.exists():
                if i + 1 > self.max_files:
                    old.unlink()  # 超出限制，删
                else:
                    old.rename(new)

        # 当前文件 → .1
        if src.exists():
            rotated = self.log_dir / f"{src.name}.1"
            src.rename(rotated)
            if self.compress_rotated:
                self._gzip(rotated)

        # 重置
        self._current_file = None
        self._current_size = 0

    def _gzip(self, path: Path) -> None:
        """把文件压缩为 .gz，节省 ~10x 存储"""
        gz_path = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.writelines(f_in)
        path.unlink()

    def _update_counters(self, event_type: str, data: Dict[str, Any]) -> None:
        self._counters["total"] += 1
        self._counters["by_type"][event_type] = self._counters["by_type"].get(event_type, 0) + 1
        if "tokens" in data and isinstance(data["tokens"], (int, float)):
            self._counters["total_tokens"] += data["tokens"]
        if "latency_ms" in data and isinstance(data["latency_ms"], (int, float)):
            lat = data["latency_ms"]
            self._counters["total_latency_ms"] += lat
            if lat > self._counters["max_latency_ms"]:
                self._counters["max_latency_ms"] = lat

    def _cleanup_loop(self) -> None:
        """每 6 小时检查一次，删除 N 天前的日志"""
        while True:
            try:
                self._cleanup_old_files()
            except Exception:
                pass
            time.sleep(6 * 3600)

    def _cleanup_old_files(self) -> None:
        cutoff = time.time() - self.retention_days * 86400
        for f in self.log_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()


# ============================================================
# LangChain 集成：自动 hook 到 LLM/tool/chain 调用
# ============================================================
try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object  # 没装 langchain 时兜底


if BaseCallbackHandler is not object:
    class LangChainCallback(BaseCallbackHandler):
        """
        接入 LangChain：自动记录每个 LLM 调用、tool 调用、chain 起点终点的耗时和 token。
        用法：agent.invoke(..., config={"callbacks": [cb]})
        """
        def __init__(self, logger: JsonlLogger, run_id: str = "main"):
            self.logger = logger
            self.run_id = run_id
            self._starts = {}  # 记录每个 run 的开始时间

        def on_chain_start(self, serialized, inputs, **kwargs):
            run_id = kwargs.get("run_id", "")
            self._starts[str(run_id)] = time.time()
            name = (serialized or {}).get("name", "?") if isinstance(serialized, dict) else "?"
            self.logger.log("chain_start", {"name": name}, self.run_id)

        def on_chain_end(self, outputs, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            lat = (time.time() - self._starts.pop(run_id, time.time())) * 1000
            self.logger.log("chain_end", {"latency_ms": round(lat, 2)}, self.run_id)

        def on_llm_start(self, serialized, prompts, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            self._starts[run_id] = time.time()
            self.logger.log("llm_start", {"n_prompts": len(prompts)}, self.run_id)

        def on_llm_end(self, response, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            lat = (time.time() - self._starts.pop(run_id, time.time())) * 1000
            tokens = 0
            try:
                usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                tokens = usage.get("total_tokens", 0)
            except Exception:
                pass
            self.logger.log("llm_end", {
                "latency_ms": round(lat, 2),
                "tokens": tokens,
                "model": response.llm_output.get("model_name", "?") if response.llm_output else "?",
            }, self.run_id)

        def on_tool_start(self, serialized, input_str, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            self._starts[run_id] = time.time()
            self.logger.log("tool_start", {"tool": serialized.get("name", "?")}, self.run_id)

        def on_tool_end(self, output, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            lat = (time.time() - self._starts.pop(run_id, time.time())) * 1000
            self.logger.log("tool_end", {
                "latency_ms": round(lat, 2),
                "output_preview": str(output)[:200],
            }, self.run_id)

        def on_tool_error(self, error, **kwargs):
            self.logger.log("error", {"msg": str(error), "kind": "tool"}, self.run_id)

        def on_llm_error(self, error, **kwargs):
            self.logger.log("error", {"msg": str(error), "kind": "llm"}, self.run_id)


# ============================================================
# 自测：跑 50 次 Agent，验证内存/存储增长可控
# ============================================================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
    import resource  # Unix: 查看 RSS

    load_dotenv()
    logger = JsonlLogger(log_dir="./logs", max_bytes=10*1024*1024)  # 测试用 10MB 上限

    @tool
    def echo(text: str) -> str:
        """回显文本。"""
        return text

    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0,
    )
    agent = create_deep_agent(model=model, tools=[echo], system_prompt="你是测试 Agent。")

    print(f"\n=== 跑 10 次请求，验证日志 + 占用 ===\n")
    for i in range(10):
        cb = LangChainCallback(logger, run_id=f"req-{i}")
        r = agent.invoke(
            {"messages": [{"role": "user", "content": f"用一句话说 hi（第 {i+1} 次）"}]},
            config={"callbacks": [cb]},
        )

    print(f"\n=== 内存统计 ===")
    print(json.dumps(logger.stats(), indent=2, ensure_ascii=False))
    print(f"\n=== 磁盘占用：{logger.disk_usage_mb()} MB ===")
    print(f"=== 进程 RSS：{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB ===\n")
    print("=== 最近 5 条日志 ===")
    for ev in logger.tail(5):
        print(f"  [{ev['ts']}] {ev['type']}: {json.dumps(ev, ensure_ascii=False)[:200]}")
