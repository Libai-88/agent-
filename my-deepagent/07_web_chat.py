"""
07_web_chat.py — Hermes Web 版（FastAPI + SSE 流式 + 简洁聊天界面）
===================================================================
启动：python 07_web_chat.py
访问：http://localhost:8000

架构：
  浏览器  ──POST /chat──>  FastAPI  ──> Deep Agent (stream)
          <──SSE 流式────          <── 逐 token 推回
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
    """执行 Python 代码（沙箱版，10s 超时）。"""
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

# ============== 工作目录 & Skills ==============
WORKSPACE = Path(__file__).parent / "web_workspace"
WORKSPACE.mkdir(exist_ok=True)
SKILLS = Path(__file__).parent / "skills"
SKILLS.mkdir(exist_ok=True)
if not (SKILLS / "style.md").exists():
    (SKILLS / "style.md").write_text(
        "# 风格\n- 中文、简洁\n- 重要对比用表格\n- 数字带千分位\n", encoding="utf-8"
    )

# ============== FastAPI lifespan：初始化 checkpointer ==============
agent = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, store
    ckpt_path = str(Path(__file__).parent / "checkpoints.db")
    memory_path = str(Path(__file__).parent / "memory.db")
    # AsyncSqliteSaver 是 async context manager
    async with AsyncSqliteSaver.from_conn_string(ckpt_path) as ckpt:
        store = SqliteStore.from_conn_string(memory_path)
        agent = create_deep_agent(
            model=model,
            tools=[get_current_time, run_python],
            system_prompt="你叫 Hermes，用户的通用助理。简洁、中文、表格化输出。",
            skills=[str(SKILLS)],
            subagents=[{
                "name": "code-worker",
                "description": "编程/计算专家",
                "system_prompt": "你是 code-worker。结论必须用 run_python 验证。",
                "tools": [run_python],
            }],
            backend=FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True),
            store=store,
            checkpointer=ckpt,
        )
        yield

app = FastAPI(title="Hermes Chat", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式接口：把 Agent 的中间步骤实时推给浏览器"""
    config = {"configurable": {"thread_id": req.user_id}}

    async def event_generator():
        try:
            async for event in agent.astream(
                {"messages": [{"role": "user", "content": req.message}]},
                config=config,
                stream_mode="values",
            ):
                msgs = event.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if hasattr(last, "content") and isinstance(last.content, str) and last.content:
                        # 跳过纯工具调用
                        if getattr(last, "type", "") == "tool":
                            yield {"event": "tool", "data": json.dumps({
                                "content": last.content[:300],
                            }, ensure_ascii=False)}
                        else:
                            yield {"event": "token", "data": json.dumps({
                                "role": getattr(last, "type", "ai"),
                                "content": last.content,
                            }, ensure_ascii=False)}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


@app.get("/files")
async def list_files():
    """列出 VFS 落盘的文件"""
    files = []
    for f in WORKSPACE.rglob("*"):
        if f.is_file():
            files.append({
                "path": str(f.relative_to(WORKSPACE)),
                "size": f.stat().st_size,
            })
    return JSONResponse(files)


@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>static/index.html missing</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
