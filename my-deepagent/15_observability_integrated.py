"""
15_observability_integrated.py — 把观测集成到 Hermes Web 服务
============================================================
基于 07_web_chat.py，把 JsonlLogger + LangChainCallback 嵌入每个请求。

新增能力：
  - 每次请求自动记录：thread_id / 工具调用 / token / 延迟
  - /admin/stats 接口：返回当前统计
  - /admin/logs/tail?n=20 接口：返回最近 n 条
  - /admin/disk 接口：返回磁盘占用
"""
import os, json
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import sys
sys.path.insert(0, str(Path(__file__).parent))
from 13_observability import JsonlLogger, LangChainCallback

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
    streaming=True,
)

# ============== 工具 ==============
@tool
def get_current_time() -> str:
    """返回当前时间。"""
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")

@tool
def run_python(code: str) -> str:
    """执行 Python 代码（10s 超时）。"""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        out = subprocess.run(["python", path], capture_output=True, text=True, timeout=10)
        return (out.stdout or "") + (("\n[stderr]\n" + out.stderr) if out.stderr else "")
    except subprocess.TimeoutExpired:
        return "[error] timeout"
    finally:
        os.unlink(path)

WORKSPACE = Path(__file__).parent / "web_workspace"
WORKSPACE.mkdir(exist_ok=True)

# ============== 观测 logger（单例） ==============
logger = JsonlLogger(
    log_dir=str(Path(__file__).parent / "logs"),
    max_bytes=50 * 1024 * 1024,   # 50MB
    max_files=5,                  # 最多 5 个轮转
    retention_days=7,             # 7 天
    compress_rotated=True,
)

# ============== FastAPI lifespan ==============
agent = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, store
    ckpt_path = str(Path(__file__).parent / "checkpoints.db")
    memory_path = str(Path(__file__).parent / "memory.db")
    async with AsyncSqliteSaver.from_conn_string(ckpt_path) as ckpt:
        store = SqliteStore.from_conn_string(memory_path)
        agent = create_deep_agent(
            model=model,
            tools=[get_current_time, run_python],
            system_prompt="你叫 Hermes，用户的通用助理。简洁、中文、表格化输出。",
            backend=FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True),
            store=store,
            checkpointer=ckpt,
        )
        yield

app = FastAPI(title="Hermes Chat + Obs", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式接口 + 自动观测"""
    config = {"configurable": {"thread_id": req.user_id}}
    run_id = f"{req.user_id}-{int.from_bytes(os.urandom(4), 'big'):08x}"
    cb = LangChainCallback(logger, run_id=run_id)  # 关键：每个请求一个 callback

    async def event_generator():
        try:
            async for event in agent.astream(
                {"messages": [{"role": "user", "content": req.message}]},
                config={**config, "callbacks": [cb]},
                stream_mode="values",
            ):
                msgs = event.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if hasattr(last, "content") and isinstance(last.content, str) and last.content:
                        if getattr(last, "type", "") == "tool":
                            yield {"event": "tool", "data": json.dumps({"content": last.content[:300]}, ensure_ascii=False)}
                        else:
                            yield {"event": "token", "data": json.dumps({"role": getattr(last, "type", "ai"), "content": last.content}, ensure_ascii=False)}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            logger.log("error", {"msg": str(e), "kind": "request"}, run_id)
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


# ============== 观测管理接口 ==============
@app.get("/admin/stats")
async def admin_stats():
    return JSONResponse(logger.stats())

@app.get("/admin/disk")
async def admin_disk():
    return JSONResponse({"disk_mb": logger.disk_usage_mb()})

@app.get("/admin/logs/tail")
async def admin_tail(n: int = 20):
    return JSONResponse(logger.tail(n))


@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>static/index.html missing</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
