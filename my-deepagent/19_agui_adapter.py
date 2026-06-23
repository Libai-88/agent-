"""
19_agui_adapter.py — AG-UI 协议适配层
====================================
在 17_web_chat_full.py 之上新增 AG-UI 协议端点，让任何 AG-UI 兼容客户端
（CopilotKit / assistant-ui / 自研前端）都能用你的 Deep Agent。

与 17/18 的关系：
  - 17: 直接 SSE 接口（自定义格式）+ 自带 HTML 前端
  - 18: OpenAI Chat Completions 兼容层（Pipe 也能用）
  - 19: AG-UI 协议层（CopilotKit 原生支持的所有功能）

启动：python 19_agui_adapter.py
访问：http://localhost:8002/api/copilotkit

为什么需要这个层？
  - CopilotKit 前端期望 AG-UI 协议事件
  - AG-UI 事件类型约 27 种，覆盖 Deep Agent 全场景：
    - TEXT_MESSAGE_START/CONTENT/END   → token 流
    - TOOL_CALL_START/ARGS/END/RESULT  → 工具调用
    - STATE_SNAPSHOT                   → todos / 全状态
    - INTERRUPT                        → HITL
  - 不用手写转换：ag-ui-langgraph 包自动把 LangGraph 事件 → AG-UI

与 18（OpenAI）的区别：
  - 18：通用客户端，token 级别粗糙
  - 19：Deep Agent 原生，token 级别 + 工具可视化 + HITL 弹窗
"""
import os
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ============== 复用 17 的设置 ==============
import importlib.util as _ilu
_HERE = Path(__file__).parent
_spec_17 = _ilu.spec_from_file_location("_web_chat_full_17", _HERE / "17_web_chat_full.py")
_mod_17 = _ilu.module_from_spec(_spec_17)
_spec_17.loader.exec_module(_mod_17)

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
get_current_time = _mod_17.get_current_time
run_python = _mod_17.run_python
msg_to_dict = _mod_17.msg_to_dict

# AG-UI 集成
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint

load_dotenv()

# ============== 共享 agent 状态 ==============
agent = None
logger = JsonlLogger(
    log_dir=str(_HERE / "logs"),
    max_bytes=50 * 1024 * 1024,
    max_files=5,
    retention_days=7,
    compress_rotated=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化 Deep Agent，添加 CopilotKitMiddleware，暴露为 AG-UI。"""
    global agent
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0,
        streaming=True,
    )
    ckpt_path = str(_HERE / "checkpoints_agui.db")
    memory_path = str(_HERE / "memory_agui.db")
    async with AsyncSqliteSaver.from_conn_string(ckpt_path) as ckpt:
        store = SqliteStore.from_conn_string(memory_path)
        agent = create_deep_agent(
            model=model,
            tools=[get_current_time, run_python],
            system_prompt=(
                "你是用户的通用助理，基于 AG-UI 协议与前端通信。\n"
                "- 简洁、中文、表格化输出。\n"
                "- 【重要】凡是用户要求执行操作、计算、文件处理等多步骤任务，"
                "你【必须】在第一步先调用 write_todos 工具拆解任务清单，"
                "然后逐项执行。哪怕只是 2 步也用 write_todos。\n"
                "- 编程/计算问题先调 run_python 验证再下结论。\n"
                "- HITL 中断时，配合前端 INTERRUPT 弹窗让用户审批。"
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
        # === 关键：把 LangGraph agent 包成 AG-UI 兼容的 LangGraphAgent ===
        agui_agent = LangGraphAgent(
            name="deepagent",
            description="用户的通用 Deep Agent，支持 TODO 计划、Python 执行、文件操作、HITL 审批",
            graph=agent,
        )
        # === 关键：把 AG-UI 端点挂到 FastAPI ===
        # 前端访问 /api/copilotkit，会自动收到 AG-UI 协议事件
        add_langgraph_fastapi_endpoint(
            app=app,
            agent=agui_agent,
            path="/api/copilotkit",
        )
        yield


app = FastAPI(title="Deep Agent AG-UI Adapter", lifespan=lifespan)

# CORS：开发阶段 CopilotKit 前端在另一个端口（5173 / 3000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 辅助端点（与 17 保持一致） ==============
@app.get("/")
async def root():
    return JSONResponse({
        "service": "Deep Agent AG-UI Adapter",
        "version": "1.0",
        "endpoint": "/api/copilotkit",
        "protocol": "AG-UI",
        "compatible_clients": [
            "CopilotKit (React + @copilotkit/react-core)",
            "assistant-ui (React)",
            "Any AG-UI compatible client",
        ],
        "hint": "前端在 .env 里配置 VITE_AGUI_URL=http://localhost:8002/api/copilotkit 即可",
    })


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": agent is not None}


if __name__ == "__main__":
    import uvicorn
    print("🚀 AG-UI Adapter on http://localhost:8002")
    print("   AG-UI endpoint: http://localhost:8002/api/copilotkit")
    print("   健康检查: http://localhost:8002/health")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")
