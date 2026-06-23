"""
Deep Agent Pipe for Open WebUI
==============================

这个 Pipe 函数把 Open WebUI 和你的 Deep Agent 后端（18_openai_adapter.py）连起来。

架构：
  Open WebUI (this Pipe)  ──HTTP/SSE──>  FastAPI 18 (port 8001)
                                       └──> Deep Agent (LangGraph)

Phase 1 设计原则：
  - Pipe 不实现任何 agent 逻辑，只做"协议翻译 + 事件展示"
  - 后端 18 已经被测过（兼容 OpenAI 协议），Pipe 直接复用
  - 所有真正的 agent 状态（HITL、todos、tool）都来自后端
  - Phase 2 切到 CopilotKit + AG-UI 时，这层 Pipe 可以直接删掉

功能映射（Pipe 视角）：
  - 流式文本    →  generator yield
  - 工具调用    →  __event_emitter__({"type": "message", "data": {"content": <HTML 卡片>}})
  - TODO 进度   →  __event_emitter__({"type": "status", "data": {"description": <进度条>}})
  - HITL        →  文本透传，用户输入 [APPROVE]/[REJECT]
  - VFS/历史    →  通过 17 的原生 API（不在 Pipe 范围内，由 Action 函数处理）

线程管理：
  - Open WebUI 每个 chat session 有自己的 chat_id
  - 我们把 chat_id 作为 thread_id 传给后端，保持会话连续性

安装：
  1. 把这个文件放到 Open WebUI 的 functions/ 目录
  2. 在 Admin Panel → Functions 启用
  3. 在 Workspace → Models 把这个 Pipe 添加为新模型

author: Libai
version: 0.1.0
required_open_webui_version: 0.5.0
"""
import os
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, Optional, Callable, Awaitable, Any
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        """Pipe 配置（Admin 可改）"""
        BACKEND_URL: str = Field(
            default="http://host.docker.internal:8001",
            description="Deep Agent 后端（18_openai_adapter.py）的 OpenAI 兼容端点",
        )
        API_KEY: str = Field(
            default="not-needed",
            description="如果有 API 鉴权，填这里",
        )
        MODEL_ID: str = Field(
            default="deepagent",
            description="传给后端的 model 名称",
        )
        TIMEOUT: int = Field(
            default=300,
            description="单次请求超时（秒）",
        )
        SHOW_TODO_PANEL: bool = Field(
            default=True,
            description="是否在状态栏显示 TODO 进度",
        )
        SHOW_TOOL_CARDS: bool = Field(
            default=True,
            description="是否以 HTML 卡片形式显示工具调用",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Deep Agent"
        # 简单 HTML 模板（自包含样式，避免依赖外网）
        self._todo_html_template = """
<details class="deepagent-todo" open>
  <summary>📋 任务进度 ({done}/{total})</summary>
  <ul style="margin:6px 0;padding-left:18px;list-style:none">
    {items}
  </ul>
</details>
"""
        self._tool_html_template = """
<details class="deepagent-tool" open>
  <summary>🔧 <b>{name}</b> {status_icon}</summary>
  {body}
</details>
"""

    # ============ 主入口 ============
    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Open WebUI 调用入口。
        body = OpenAI chat completions 格式
        """
        # 1. 提取 thread_id（用 chat_id 保持会话）
        chat_id = (body.get("metadata", {}) or {}).get("chat_id") or "default"
        # Open WebUI user dict
        user_id = (__user__ or {}).get("id", "anonymous")
        thread_id = f"{user_id}:{chat_id}"

        # 2. 准备请求
        # 透传 messages，但加 user 字段当 thread_id
        req_body = {
            "model": self.valves.MODEL_ID,
            "messages": body.get("messages", []),
            "stream": True,
            "user": thread_id,
        }
        url = f"{self.valves.BACKEND_URL.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.API_KEY}",
        }

        # 3. 状态：开始
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "🤔 思考中...", "done": False},
            })

        # 4. 流式调用
        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=req_body, headers=headers) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        yield f"❌ 后端错误 {resp.status}: {err_text[:500]}"
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": "❌ 出错了", "done": True},
                            })
                        return

                    # 5. 解析 SSE 流
                    full_text_parts = []
                    last_todo_state = None
                    tool_call_buffer = {}  # index -> {name, args, id}
                    async for line in resp.content:
                        if not line:
                            continue
                        try:
                            line_str = line.decode("utf-8").rstrip("\r\n")
                        except UnicodeDecodeError:
                            continue
                        if not line_str or not line_str.startswith("data: "):
                            continue
                        data_str = line_str[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # 5a) tool_calls → 卡片
                        if self.valves.SHOW_TOOL_CARDS and "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_call_buffer:
                                    tool_call_buffer[idx] = {
                                        "id": tc.get("id", f"call_{idx}"),
                                        "name": tc.get("function", {}).get("name", "?"),
                                        "args": "",
                                    }
                                if "function" in tc:
                                    args = tc["function"].get("arguments", "")
                                    tool_call_buffer[idx]["args"] += args

                        # 5b) 文本 content
                        content = delta.get("content")
                        if content:
                            # 区分 TODO 文本、interrupt 文本、普通文本
                            if content.startswith("📋") or "任务进度" in content:
                                # TODO 进度 → 状态栏
                                if self.valves.SHOW_TODO_PANEL and __event_emitter__:
                                    # 解析 TODO 列表
                                    todo_html = self._render_todo_html(content)
                                    await __event_emitter__({
                                        "type": "status",
                                        "data": {"description": todo_html, "done": False},
                                    })
                                last_todo_state = content
                            elif content.startswith("🔔") or "等待审批" in content:
                                # HITL 提示
                                if __event_emitter__:
                                    await __event_emitter__({
                                        "type": "message",
                                        "data": {"content": f"```\n{content.strip()}\n```"},
                                    })
                                # 也输出到主消息流（用户能看到）
                                full_text_parts.append(content)
                                yield content
                            elif content.startswith("[错误]") or content.startswith("[已取消]"):
                                # 错误/取消
                                if __event_emitter__:
                                    await __event_emitter__({
                                        "type": "status",
                                        "data": {"description": content.strip(), "done": True},
                                    })
                                full_text_parts.append(content)
                                yield content
                            else:
                                full_text_parts.append(content)
                                yield content

                        # 5c) 流结束
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason == "stop":
                            # 把累积的 tool_call_buffer 渲染成卡片
                            if tool_call_buffer and self.valves.SHOW_TOOL_CARDS and __event_emitter__:
                                for idx, tc in sorted(tool_call_buffer.items()):
                                    tool_html = self._render_tool_call_html(tc)
                                    await __event_emitter__({
                                        "type": "message",
                                        "data": {"content": tool_html},
                                    })
                            break

        except asyncio.TimeoutError:
            yield "\n⏱️ 请求超时"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "⏱️ 超时", "done": True},
                })
        except aiohttp.ClientError as e:
            yield f"\n❌ 连接后端失败: {e}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "❌ 连接失败", "done": True},
                })
        except Exception as e:
            yield f"\n❌ 未知错误: {type(e).__name__}: {e}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"❌ {type(e).__name__}", "done": True},
                })
        finally:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ 完成", "done": True},
                })

    # ============ HTML 渲染 ============
    def _render_todo_html(self, content: str) -> str:
        """从 📋 任务进度 ... 文本中解析 todo 列表，渲染为 HTML。"""
        items = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("📋"):
                continue
            if line.startswith("- "):
                # 状态图标
                if line.startswith("- ●"):
                    items.append(f'<li style="color:#52c41a">✅ {line[3:].strip()}</li>')
                elif line.startswith("- ◐"):
                    items.append(f'<li style="color:#1890ff">🔵 {line[3:].strip()}</li>')
                elif line.startswith("- ○"):
                    items.append(f'<li style="color:#999">⚪ {line[3:].strip()}</li>')
                else:
                    items.append(f'<li>{line[2:].strip()}</li>')
        done = sum(1 for i in items if "✅" in i)
        return self._todo_html_template.format(
            done=done, total=len(items), items="\n    ".join(items) if items else "<li>暂无任务</li>"
        )

    def _render_tool_call_html(self, tc: dict) -> str:
        """把 tool_call 渲染成折叠卡片。"""
        name = tc.get("name", "?")
        args = tc.get("args", "{}")
        try:
            args_dict = json.loads(args) if args else {}
            body_html = f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;overflow:auto'><code>{json.dumps(args_dict, ensure_ascii=False, indent=2)}</code></pre>"
        except json.JSONDecodeError:
            body_html = f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px'><code>{args}</code></pre>"
        return self._tool_html_template.format(
            name=name, status_icon="⏳", body=body_html
        )
