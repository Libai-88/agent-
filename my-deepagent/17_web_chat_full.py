"""
17_web_chat_full.py — Hermes 完整 Web 版（带 6 大增强）
=========================================================
在 15_observability_integrated.py 之上叠加：
  1. 历史消息恢复（/api/history）
  2. TODO 进度条（流式 todos 事件）
  3. 工具调用可视化（完整 args + 结果）
  4. 停止生成按钮（task.cancel）
  5. VFS 文件浏览器（/api/files/tree + /api/files/read）
  6. Ch9 Human-in-the-Loop（run_python 需审批）

启动：python 17_web_chat_full.py
访问：http://localhost:8000
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite import SqliteStore
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

import importlib.util as _ilu
_OB_PATH = Path(__file__).parent / "13_observability.py"
_spec = _ilu.spec_from_file_location("_observability", _OB_PATH)
_obs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_obs)
JsonlLogger = _obs.JsonlLogger
LangChainCallback = _obs.LangChainCallback  # noqa: E402

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
    """执行 Python 代码（10s 超时）。HITL 启用时需要用户审批。"""
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
    max_bytes=50 * 1024 * 1024,
    max_files=5,
    retention_days=7,
    compress_rotated=True,
)

# ============== 运行中的请求（用于取消） ==============
# key: thread_id, value: asyncio.Task
running_tasks: dict[str, asyncio.Task] = {}
# key: thread_id, value: list of asyncio.Queue（用于审批事件）
interrupt_queues: dict[str, asyncio.Queue] = {}


# ============== FastAPI lifespan ==============
agent = None
store = None
checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, store, checkpointer
    ckpt_path = str(Path(__file__).parent / "checkpoints.db")
    memory_path = str(Path(__file__).parent / "memory.db")
    async with AsyncSqliteSaver.from_conn_string(ckpt_path) as ckpt:
        checkpointer = ckpt
        store = SqliteStore.from_conn_string(memory_path)
        agent = create_deep_agent(
            model=model,
            tools=[get_current_time, run_python],
            system_prompt=(
                "你叫 Hermes，用户的通用助理。\n"
                "- 简洁、中文、表格化输出。\n"
                "- 【重要】凡是用户要求执行操作、计算、文件处理等多步骤任务，"
                "你【必须】在第一步先调用 write_todos 工具拆解任务清单，"
                "然后逐项执行。哪怕只是 2 步也用 write_todos。\n"
                "- 编程/计算问题先调 run_python 验证再下结论。"
            ),
            # ⚠️ Ch9 HITL：run_python 需要用户审批
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "run_python": InterruptOnConfig(
                            allowed_decisions=["approve", "reject"],
                            description="Agent 想执行 Python 代码，请审批",
                        ),
                    },
                ),
            ],
            backend=FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True),
            store=store,
            checkpointer=ckpt,
        )
        yield


app = FastAPI(title="Hermes Full", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)


# ============== Pydantic 模型 ==============
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ResumeRequest(BaseModel):
    user_id: str
    decisions: list[dict]  # [{"type": "approve"} | {"type": "reject"}]


class CancelRequest(BaseModel):
    user_id: str


# ============== 辅助：把消息转成可序列化 dict ==============
def msg_to_dict(msg) -> dict:
    """把 LangChain message 序列化成 dict 给前端。"""
    out = {"type": getattr(msg, "type", "unknown"), "content": ""}
    if isinstance(msg.content, str):
        out["content"] = msg.content
    elif isinstance(msg.content, list):
        # 多模态/结构化内容
        out["content"] = msg.content
    # tool_calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in msg.tool_calls
        ]
    # tool_call_id
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        out["tool_call_id"] = msg.tool_call_id
    return out


def serialize_state(messages, todos) -> dict:
    """把历史状态（messages + todos）序列化成 JSON 友好格式。"""
    return {
        "messages": [msg_to_dict(m) for m in messages],
        "todos": todos or [],
    }


# ============== 1. 历史消息恢复 ==============
@app.get("/api/history")
async def api_history(user_id: str):
    """根据 thread_id 加载历史消息 + todos。"""
    config = {"configurable": {"thread_id": user_id}}
    state = await agent.aget_state(config)
    values = state.values if state else {}
    messages = values.get("messages", [])
    todos = values.get("todos", [])
    return JSONResponse(serialize_state(messages, todos))


# ============== 2/3. 流式 chat（含 TODO + 工具调用可视化） ==============
@app.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式接口，支持：
       - 增量 token（event: token）
       - 工具调用（event: tool_call）
       - 工具结果（event: tool_result）
       - TODO 状态（event: todos）
       - 中断审批（event: interrupt）
    """
    config = {"configurable": {"thread_id": req.user_id}}
    run_id = f"{req.user_id}-{int.from_bytes(os.urandom(4), 'big'):08x}"
    cb = LangChainCallback(logger, run_id=run_id)

    input_payload: Any = {"messages": [{"role": "user", "content": req.message}]}

    async def event_generator():
        task = asyncio.current_task()
        if task is not None:
            running_tasks[req.user_id] = task
            interrupt_queues[req.user_id] = asyncio.Queue()

        # 发送 run_id 便于前端关联
        yield {"event": "meta", "data": json.dumps({"run_id": run_id})}

        last_todos_json = None
        last_message_count = 0

        try:
            async for event in agent.astream(
                input_payload,
                config={**config, "callbacks": [cb]},
                stream_mode="values",
            ):
                # ✅ 2. TODO 状态
                todos = event.get("todos")
                if todos is not None:
                    todos_json = json.dumps(todos, ensure_ascii=False)
                    if todos_json != last_todos_json:
                        last_todos_json = todos_json
                        yield {"event": "todos", "data": todos_json}

                # ✅ 3. 工具调用 + AI 内容
                messages = event.get("messages", [])
                if len(messages) > last_message_count:
                    for new_msg in messages[last_message_count:]:
                        d = msg_to_dict(new_msg)
                        if d.get("tool_calls"):
                            yield {
                                "event": "tool_call",
                                "data": json.dumps(d["tool_calls"], ensure_ascii=False),
                            }
                        if d.get("type") == "tool" and d.get("content"):
                            yield {
                                "event": "tool_result",
                                "data": json.dumps({
                                    "content": d["content"],
                                    "tool_call_id": d.get("tool_call_id"),
                                }, ensure_ascii=False),
                            }
                    last_message_count = len(messages)

                # 最后一条 AI 消息流式输出（取最后一条 AI 消息的内容）
                if messages:
                    last = messages[-1]
                    if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content:
                        # 跳过纯工具调用
                        if not last.tool_calls:
                            yield {
                                "event": "token",
                                "data": json.dumps({
                                    "role": "ai",
                                    "content": last.content,
                                }, ensure_ascii=False),
                            }

            # 流结束后检查是否有 interrupt
            state = await agent.aget_state(config)
            if state and state.tasks:
                for t in state.tasks:
                    if t.interrupts:
                        interrupt_value = t.interrupts[0].value
                        yield {
                            "event": "interrupt",
                            "data": json.dumps(interrupt_value, ensure_ascii=False),
                        }
                        # 等待用户审批
                        decision = await interrupt_queues[req.user_id].get()
                        # 拿到决策后用 Command(resume=...) 继续
                        async for resume_event in agent.astream(
                            Command(resume={"decisions": [decision]}),
                            config={**config, "callbacks": [cb]},
                            stream_mode="values",
                        ):
                            todos = resume_event.get("todos")
                            if todos is not None:
                                yield {
                                    "event": "todos",
                                    "data": json.dumps(todos, ensure_ascii=False),
                                }
                            messages = resume_event.get("messages", [])
                            for new_msg in messages[last_message_count:]:
                                d = msg_to_dict(new_msg)
                                if d.get("tool_calls"):
                                    yield {
                                        "event": "tool_call",
                                        "data": json.dumps(d["tool_calls"], ensure_ascii=False),
                                    }
                                if d.get("type") == "tool" and d.get("content"):
                                    yield {
                                        "event": "tool_result",
                                        "data": json.dumps({
                                            "content": d["content"],
                                            "tool_call_id": d.get("tool_call_id"),
                                        }, ensure_ascii=False),
                                    }
                                if isinstance(new_msg, AIMessage) and isinstance(new_msg.content, str) and new_msg.content:
                                    if not new_msg.tool_calls:
                                        yield {
                                            "event": "token",
                                            "data": json.dumps({
                                                "role": "ai",
                                                "content": new_msg.content,
                                            }, ensure_ascii=False),
                                        }
                            last_message_count = len(messages)
                        break

            yield {"event": "done", "data": "[DONE]"}
        except asyncio.CancelledError:
            yield {"event": "cancelled", "data": "user cancelled"}
            raise
        except Exception as e:
            logger.log("error", {"msg": str(e), "kind": "request"}, run_id)
            yield {"event": "error", "data": str(e)}
        finally:
            running_tasks.pop(req.user_id, None)
            interrupt_queues.pop(req.user_id, None)

    return EventSourceResponse(event_generator())


# ============== 4. 停止生成 ==============
@app.post("/api/cancel")
async def api_cancel(req: CancelRequest):
    """取消正在运行的请求。"""
    task = running_tasks.get(req.user_id)
    if task and not task.done():
        task.cancel()
        return JSONResponse({"ok": True, "msg": "cancelled"})
    return JSONResponse({"ok": False, "msg": "no running task"})


# ============== 6. HITL 审批 ==============
@app.post("/api/approve")
async def api_approve(req: ResumeRequest):
    """把用户的审批决策送入 interrupt 队列。"""
    q = interrupt_queues.get(req.user_id)
    if not q:
        return JSONResponse({"ok": False, "msg": "no pending interrupt"})
    # 只支持单决策
    await q.put(req.decisions[0] if req.decisions else {"type": "reject"})
    return JSONResponse({"ok": True})


# ============== 5. VFS 文件浏览器 ==============
@app.get("/api/files/tree")
async def api_files_tree():
    """返回 VFS 落盘的树状结构。"""
    def build_tree(path: Path, rel: str = "") -> dict:
        node = {
            "name": path.name,
            "path": rel or path.name,
            "type": "dir" if path.is_dir() else "file",
        }
        if path.is_dir():
            children = []
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                if child.name.startswith("."):
                    continue
                child_rel = f"{rel}/{child.name}" if rel else child.name
                children.append(build_tree(child, child_rel))
            node["children"] = children
        else:
            node["size"] = path.stat().st_size
        return node

    tree = build_tree(WORKSPACE)
    return JSONResponse(tree)


@app.get("/api/files/read")
async def api_files_read(path: str):
    """读取 VFS 中的文件内容。"""
    # 安全：阻止 path traversal
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE.resolve())):
        return JSONResponse({"error": "path traversal blocked"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = "[二进制文件，无法预览]"
    return JSONResponse({
        "path": path,
        "content": content,
        "size": target.stat().st_size,
    })


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


# ============== 首页 ==============
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>static/index.html missing</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
