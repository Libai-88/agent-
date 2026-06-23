"""
18_openai_adapter.py — OpenAI Chat Completions 兼容层
====================================================
在 17_web_chat_full.py 之上新增两个端点，让任何 OpenAI 兼容客户端
（curl / OpenAI SDK / Open WebUI Pipe / 其他 LLM 工具）都能用你的 Deep Agent：

  POST /v1/chat/completions   ← 聊天（支持 stream）
  GET  /v1/models             ← 模型列表

与原 17 的关系：
  - 17 保留：直接 SSE 接口 + 自带 HTML 前端 + 全部 API
  - 18 新增：OpenAI 协议适配层
  - 两者共享同一个 agent 实例（在 lifespan 中初始化）

启动：python 18_openai_adapter.py
访问：http://localhost:8001/v1/chat/completions

为什么要这个层？
  - Phase 1：Open WebUI Pipe 可以直接 import 17 的 agent（更原生）
  - 但是其他 OpenAI 客户端（curl / SDK / IDE 插件 / 第三方工具）
    没法 import Python，只能走 HTTP OpenAI 协议
  - 有了这个层，任何 OpenAI 客户端都能复用 Deep Agent
  - Phase 2 切 CopilotKit + AG-UI 时，这层继续保留作为 fallback
"""
import os
import sys
import json
import asyncio
import uuid
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

# ============== 复用 17 的 agent 设置 ==============
# 通过 importlib 加载 17 模块，拿到 model / agent / logger / workspace
import importlib.util as _ilu
_HERE = Path(__file__).parent
_spec_17 = _ilu.spec_from_file_location("_web_chat_full_17", _HERE / "17_web_chat_full.py")
_mod_17 = _ilu.module_from_spec(_spec_17)
_spec_17.loader.exec_module(_mod_17)

# 暴露给 18 用的对象
create_deep_agent = _mod_17.create_deep_agent
FilesystemBackend = _mod_17.FilesystemBackend
HumanInTheLoopMiddleware = _mod_17.HumanInTheLoopMiddleware
InterruptOnConfig = _mod_17.InterruptOnConfig
AsyncSqliteSaver = _mod_17.AsyncSqliteSaver
SqliteStore = _mod_17.SqliteStore
ChatOpenAI = _mod_17.ChatOpenAI
JsonlLogger = _mod_17.JsonlLogger
LangChainCallback = _mod_17.LangChainCallback
WORKSPACE = _mod_17.WORKSPACE
msg_to_dict = _mod_17.msg_to_dict
load_dotenv = _mod_17.load_dotenv

load_dotenv()

# ============== 共享 agent 状态 ==============
agent = None
checkpointer = None
store = None
logger = JsonlLogger(
    log_dir=str(_HERE / "logs"),
    max_bytes=50 * 1024 * 1024,
    max_files=5,
    retention_days=7,
    compress_rotated=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化 agent（和 17 一样，只是单独一份）。"""
    global agent, checkpointer, store
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0,
        streaming=True,
    )
    ckpt_path = str(_HERE / "checkpoints_openai.db")  # 独立 checkpoint 文件
    memory_path = str(_HERE / "memory_openai.db")
    async with AsyncSqliteSaver.from_conn_string(ckpt_path) as ckpt:
        checkpointer = ckpt
        store = SqliteStore.from_conn_string(memory_path)
        agent = create_deep_agent(
            model=model,
            tools=[_mod_17.get_current_time, _mod_17.run_python],
            system_prompt=(
                "你是用户的通用助理，基于 OpenAI 兼容接口调用。\n"
                "- 简洁、中文、表格化输出。\n"
                "- 【重要】凡是用户要求执行操作、计算、文件处理等多步骤任务，"
                "你【必须】在第一步先调用 write_todos 工具拆解任务清单，"
                "然后逐项执行。哪怕只是 2 步也用 write_todos。\n"
                "- 编程/计算问题先调 run_python 验证再下结论。\n"
                "- HITL 中断时，回复中明确提示用户输入 [APPROVE] 或 [REJECT] 继续。"
            ),
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


app = FastAPI(title="Deep Agent OpenAI Adapter", lifespan=lifespan)


# ============== Pydantic 模型（OpenAI 协议） ==============
class OpenAIMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "deepagent"
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    user: str | None = None  # 用作 thread_id（Open WebUI 会传 conversation_id）


# ============== 工具函数 ==============
def make_chunk(model: str, delta: dict, finish_reason: str | None = None) -> str:
    """生成 OpenAI 流式 chunk。"""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def make_done() -> str:
    return "data: [DONE]\n\n"


def openai_messages_to_lc(messages: list[OpenAIMessage]) -> tuple[list, str | None]:
    """
    把 OpenAI 消息转 LangChain 消息。
    顺便检测最后一条 user 消息是否是审批决定（[APPROVE]/[REJECT]）。
    返回 (lc_messages, approval_decision)
    """
    approval_decision = None
    lc_messages = []
    for m in messages:
        content = m.content or ""
        # 检测最后一条 user 消息的审批意图
        if m.role == "user":
            content_lower = content.strip().upper()
            if content_lower in ("[APPROVE]", "APPROVE", "同意", "批准"):
                approval_decision = {"type": "approve"}
            elif content_lower in ("[REJECT]", "REJECT", "拒绝", "否决"):
                approval_decision = {"type": "reject"}
        if m.role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif m.role == "tool":
            lc_messages.append(ToolMessage(content=content, tool_call_id=m.tool_call_id or ""))
        elif m.role == "system":
            # Deep Agent 已经有自己的 system_prompt，忽略外部 system
            pass
    return lc_messages, approval_decision


# ============== 核心：流式 chat ==============
async def stream_chat(req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """
    把 Deep Agent 的流式输出转成 OpenAI 协议 chunk。
    关键映射：
      - token    → choices[0].delta.content
      - tool_call → choices[0].delta.tool_calls (indexed)
      - todos    → choices[0].delta.content (用 "📋 任务进度：\n- [x] step1" 形式)
      - interrupt → choices[0].delta.content (用 "🔔 等待审批..." 形式)
    """
    # thread_id 策略：优先用 user 字段（Open WebUI 会传），否则用 messages 哈希
    thread_id = req.user or f"openai-{hash(tuple(m.content for m in req.messages)) & 0xFFFFFFFF}"
    config = {"configurable": {"thread_id": thread_id}}
    run_id = f"{thread_id}-{uuid.uuid4().hex[:8]}"
    cb = LangChainCallback(logger, run_id=run_id)

    # 检测审批决定
    lc_messages, approval_decision = openai_messages_to_lc(req.messages)

    # 检查是否处于 interrupt 状态
    state = await agent.aget_state(config)
    has_pending_interrupt = bool(state and state.tasks and any(
        t.interrupts for t in state.tasks
    ))

    if has_pending_interrupt and approval_decision:
        # ✅ 用户在回复审批决定 → 用 Command(resume=...) 继续
        # ⚠️ HITL 要求 decisions 数量 == pending tool call 数量
        # 用户只发一个 [APPROVE]/[REJECT]，需要广播到所有挂起的工具调用
        pending_count = 0
        if state and state.tasks:
            for t in state.tasks:
                if t.interrupts:
                    # 每次 interrupt 通常对应 N 个 action_requests
                    interrupt_value = t.interrupts[0].value
                    pending_count = max(pending_count, len(interrupt_value.get("action_requests", [])))
        if pending_count == 0:
            pending_count = 1
        decisions = [approval_decision] * pending_count
        input_payload = Command(resume={"decisions": decisions})
    else:
        # 正常新消息：只发最后一条 user 消息（前面历史从 checkpoint 恢复）
        last_user = None
        for m in reversed(lc_messages):
            if isinstance(m, HumanMessage):
                last_user = m
                break
        if last_user is None:
            yield make_chunk(req.model, {"role": "assistant", "content": "（没有收到 user 消息）"})
            yield make_done()
            return
        input_payload = {"messages": [last_user]}

    # ✅ 关键修复：resume 时先读当前 state 消息数，避免重复 emit checkpoint 里的旧消息
    state = await agent.aget_state(config)
    initial_message_count = len(state.values.get("messages", [])) if state else 0

    # 流式生成
    last_todos_json = None
    last_message_count = initial_message_count
    has_yielded_text = False

    try:
        async for event in agent.astream(
            input_payload,
            config={**config, "callbacks": [cb]},
            stream_mode="values",
        ):
            # 1) todos → 作为 content 增量
            todos = event.get("todos")
            if todos is not None:
                todos_json = json.dumps(todos, ensure_ascii=False)
                if todos_json != last_todos_json:
                    last_todos_json = todos_json
                    # 用 emoji 编码成 content，前端可识别
                    todo_lines = []
                    for t in todos:
                        icon = {"pending": "○", "in_progress": "◐", "completed": "●"}.get(
                            t.get("status", "pending"), "○"
                        )
                        todo_lines.append(f"{icon} {t.get('content', '')}")
                    todo_text = "📋 **任务进度**\n" + "\n".join(todo_lines) + "\n\n"
                    if has_yielded_text:
                        yield make_chunk(req.model, {"content": "\n" + todo_text})
                    else:
                        yield make_chunk(req.model, {"role": "assistant", "content": todo_text})
                        has_yielded_text = True

            # 2) tool_call → OpenAI tool_calls 格式
            messages = event.get("messages", [])
            for new_msg in messages[last_message_count:]:
                d = msg_to_dict(new_msg)
                if d.get("tool_calls"):
                    for i, tc in enumerate(d["tool_calls"]):
                        yield make_chunk(req.model, {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": i,
                                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                                },
                            }],
                        })
            last_message_count = len(messages)

            # 3) AI 文本 → content 增量
            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content:
                    if not last.tool_calls:
                        if has_yielded_text:
                            # 注意：流式场景下这里会重复发整段。
                            # 简化：仅在 content 第一次出现时发全量（OpenAI 也允许非 token 级）
                            # 实际优化可改用 astream_events，但增加复杂度
                            pass
                        else:
                            yield make_chunk(req.model, {
                                "role": "assistant",
                                "content": "\n" + last.content if has_yielded_text else last.content,
                            })
                            has_yielded_text = True

        # 4) 流结束后检查 interrupt
        state = await agent.aget_state(config)
        if state and state.tasks:
            for t in state.tasks:
                if t.interrupts:
                    interrupt_value = t.interrupts[0].value
                    # LangChain HumanInTheLoopMiddleware 的结构：
                    # {"action_requests": [{"name", "args", "description"}], "review_configs": [...]}
                    action_requests = interrupt_value.get("action_requests", [])
                    if action_requests:
                        req0 = action_requests[0]
                        action_name = req0.get("name", "?")
                        action_args = req0.get("args", {})
                        description = req0.get("description", "")
                        # 渲染 args
                        if isinstance(action_args, dict):
                            args_str = json.dumps(action_args, ensure_ascii=False, indent=2)
                        else:
                            args_str = str(action_args)
                        interrupt_msg = (
                            "\n\n🔔 **等待审批**\n"
                            f"操作：`{action_name}`\n"
                            f"参数：\n```json\n{args_str}\n```\n"
                            f"说明：{description}\n\n"
                            "请输入 `[APPROVE]` 或 `[REJECT]` 继续。"
                        )
                    else:
                        interrupt_msg = "\n\n🔔 **等待审批**（无法解析 action_request）\n请输入 `[APPROVE]` 或 `[REJECT]` 继续。"
                    yield make_chunk(req.model, {"content": interrupt_msg})
                    break

        yield make_chunk(req.model, {}, finish_reason="stop")
        yield make_done()

    except asyncio.CancelledError:
        yield make_chunk(req.model, {"content": "\n[已取消]"})
        yield make_chunk(req.model, {}, finish_reason="stop")
        yield make_done()
        raise
    except Exception as e:
        logger.log("error", {"msg": str(e), "kind": "openai-adapter"}, run_id)
        yield make_chunk(req.model, {"content": f"\n[错误] {e}"})
        yield make_chunk(req.model, {}, finish_reason="stop")
        yield make_done()


# ============== 端点 ==============
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI 兼容的 chat completions。"""
    if not req.stream:
        # 非流式：把整个流累积成一个 response
        chunks = []
        async for chunk in stream_chat(req):
            chunks.append(chunk)
        # 解析最后的内容
        full_content = ""
        for c in chunks:
            if c.startswith("data: ") and c.strip() != "data: [DONE]":
                try:
                    data = json.loads(c[6:].strip())
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
                except Exception:
                    pass
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    # 流式
    return StreamingResponse(
        stream_chat(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/models")
async def list_models():
    """OpenAI 兼容的模型列表。"""
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "deepagent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
            },
        ],
    })


@app.get("/")
async def root():
    return JSONResponse({
        "service": "Deep Agent OpenAI Adapter",
        "version": "1.0",
        "endpoints": {
            "POST /v1/chat/completions": "OpenAI Chat Completions (compatible)",
            "GET /v1/models": "List models",
        },
        "hint": "This adapter sits on top of 17_web_chat_full.py. Use 17 for full features (todos/tool/HITL/VFS).",
    })


if __name__ == "__main__":
    import uvicorn
    print("🚀 OpenAI Adapter on http://localhost:8001")
    print("   Test: curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
