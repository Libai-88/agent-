"""
Deep Agent VFS & History Action for Open WebUI
==============================================

Open WebUI Action 函数：在聊天界面里提供"文件浏览"和"会话历史"两个命令。

工作机制：
  - Open WebUI 把 Action 注册为聊天命令（如 /vfs, /history）
  - 用户输入 /vfs 后，Action 被调用
  - Action 调用 17 的原生 API，返回结果以消息形式插入对话

调用方式：
  /vfs                    ← 列出 web_workspace/ 全部文件
  /vfs <path>             ← 读取 web_workspace/<path>
  /history                ← 列出当前 thread 的历史消息
  /threads                ← 列出所有活跃 thread

⚠️ 这个 Action 在 Phase 1 暂时不实现复杂功能，只展示架构。
   真正落地需要：
   1. Action 通过 HTTP 调用 17 的 /api/files/tree, /api/files/read
   2. Open WebUI 注入 __user__ 和 chat_id，作为 thread_id 的一部分

author: Libai
version: 0.1.0
required_open_webui_version: 0.5.0
"""
import os
import json
import aiohttp
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        BACKEND_URL: str = Field(
            default="http://host.docker.internal:8000",
            description="Deep Agent 原生后端（17_web_chat_full.py）的地址，VFS API 端口 8000",
        )
        TIMEOUT: int = Field(default=30)

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ):
        """
        Open WebUI Action 入口。
        body 包含：
          - message: 用户输入的完整消息（含 /vfs 等命令）
          - chat_id: 当前会话 ID
        """
        # 解析命令
        message = body.get("message", "").strip()
        chat_id = body.get("chat_id", "default")
        user_id = (__user__ or {}).get("id", "anonymous")
        thread_id = f"{user_id}:{chat_id}"

        # 1. /vfs [path]
        if message.startswith("/vfs"):
            return await self._cmd_vfs(message, thread_id, __event_emitter__)

        # 2. /history
        if message.startswith("/history"):
            return await self._cmd_history(thread_id, __event_emitter__)

        # 3. /threads
        if message.startswith("/threads"):
            return await self._cmd_threads(__event_emitter__)

        # 未知命令
        return "❓ 未知命令。可用：`/vfs [path]`, `/history`, `/threads`"

    async def _cmd_vfs(self, message: str, thread_id: str, emitter) -> str:
        """列出文件树或读文件。"""
        parts = message.split(maxsplit=1)
        sub_path = parts[1] if len(parts) > 1 else ""

        url = f"{self.valves.BACKEND_URL.rstrip('/')}/api/files/tree"
        if sub_path:
            url = f"{self.valves.BACKEND_URL.rstrip('/')}/api/files/read?path={sub_path}"

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return f"❌ 后端错误 {resp.status}: {await resp.text()}"
                    if sub_path:
                        # 读文件
                        d = await resp.json()
                        return f"📄 **{sub_path}**\n```\n{d.get('content', '')}\n```"
                    else:
                        # 文件树
                        d = await resp.json()
                        return self._format_tree(d)
        except Exception as e:
            return f"❌ 调用失败：{type(e).__name__}: {e}"

    async def _cmd_history(self, thread_id: str, emitter) -> str:
        """取历史消息。"""
        # 17 的 /api/history 接受 thread_id
        url = f"{self.valves.BACKEND_URL.rstrip('/')}/api/history?user_id={thread_id}"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.valves.TIMEOUT)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return f"❌ 后端错误 {resp.status}: {await resp.text()}"
                    d = await resp.json()
                    msgs = d.get("messages", [])
                    if not msgs:
                        return "（暂无历史）"
                    out = [f"📜 **{thread_id}** 历史 ({len(msgs)} 条)\n"]
                    for m in msgs[-10:]:
                        role = m.get("type", m.get("role", "?"))
                        content = m.get("content", "")
                        if isinstance(content, str):
                            preview = content[:80] + ("..." if len(content) > 80 else "")
                            out.append(f"- **{role}**: {preview}")
                    return "\n".join(out)
        except Exception as e:
            return f"❌ 调用失败：{type(e).__name__}: {e}"

    async def _cmd_threads(self, emitter) -> str:
        """列出活跃 thread。"""
        # 17 没专门这个接口，先用 /admin/logs/tail 顶一下
        return "⚠️ /threads 待实现：建议在 Phase 2 接入到消息存储层时一起做"

    def _format_tree(self, node: dict, indent: int = 0) -> str:
        """把文件树 dict 格式化成 markdown。"""
        prefix = "  " * indent
        name = node.get("name", "?")
        type_ = node.get("type", "?")
        if type_ == "directory":
            lines = [f"{prefix}📁 **{name}/**"]
            for child in node.get("children", []):
                lines.append(self._format_tree(child, indent + 1))
            return "\n".join(lines)
        else:
            size = node.get("size", 0)
            return f"{prefix}📄 {name} ({size} bytes)"
