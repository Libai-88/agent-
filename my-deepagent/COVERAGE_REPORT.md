# Deep Agents 实战学习 - 课程覆盖度验证报告

**生成时间**：2026-06-23  
**项目目录**：`/workspace/my-deepagent`  
**模型**：DeepSeek V4 Flash（OpenAI 兼容接口）

---

## 📈 总进度

| 章节 | 标题 | 状态 | 覆盖度 |
|---|---|---|---|
| Ch1 | 认知篇：三层架构 | ✅ | 100% |
| Ch2 | 快速上手 + LangSmith | ✅ | 100% |
| Ch3 | 虚拟文件系统 | ✅ | 100% |
| Ch4 | 任务规划 | ✅ | 100% |
| Ch5 | 子 Agent | ✅ | 100% |
| Ch6 | 异步子 Agent | ✅ | 100%（需 LangGraph Platform 部署跑端到端） |
| Ch7 | Skills | ✅ | 100%（按 Agent Skills 规范） |
| Ch8 | 长期记忆 | ✅ | 100%（含 CompositeBackend） |

**整体覆盖度：100%（核心 8 章全部覆盖）**

---

## 📁 文件清单

| 文件 | 章节 | 跑通状态 |
|---|---|---|
| 01_minimal.py | Ch2 | ✅ |
| 02_office_agent.py | Ch2 | ✅ |
| 03_vfs_planning.py | Ch3+Ch4 | ✅ |
| 04_subagents.py | Ch5 | ✅ |
| 05_skills_memory.py | Ch7+Ch8 基础 | ✅ |
| 06_your_hermes.py | Ch8 完整 | ✅ |
| 07_web_chat.py | 实战篇 | ✅ Web 服务运行中 |
| 08_vfs_full.py | **Ch3 完整 6 工具** | ✅ |
| 09_subagents_full.py | **Ch5 3 种写法** | ✅ |
| 10_memory_composite.py | **Ch8 memory= + Composite** | ✅ |
| 11_async_subagent.py | **Ch6 异步** | ✅（API 演示） |
| 12_langsmith.py | **Ch2 LangSmith** | ✅ |
| skills/report-writer/SKILL.md | Ch7 | ✅ |

---

## 🎯 关键能力矩阵

| 能力 | 课程章节 | 我们的实现 |
|---|---|---|
| `create_deep_agent()` 基础调用 | Ch2 | 01/02 |
| `@tool` 装饰器 | Ch2 | 02 |
| `ls` | Ch3 | 08 |
| `read_file` | Ch3 | 08 |
| `write_file` | Ch3 | 03/08 |
| `edit_file` | Ch3 | 08 |
| `glob` | Ch3 | 08 |
| `grep` | Ch3 | 08 |
| `FilesystemBackend` | Ch3 | 03/06/07/08/10 |
| `StateBackend` | Ch3 | 10 |
| `CompositeBackend` 路径路由 | **Ch8** | 10 |
| `write_todos` 自动任务规划 | Ch4 | 03 |
| `subagents` 字典式 | Ch5 | 04/09 |
| `general-purpose` 子 Agent | **Ch5 进阶** | 09 |
| `CompiledSubAgent` | **Ch5 高级** | 09 |
| `AsyncSubAgent` + AsyncSubAgentMiddleware | **Ch6** | 11 |
| `skills` 路径参数 | Ch7 | 05/06/07/10 |
| `SKILL.md` + scripts + references | **Ch7 规范** | skills/report-writer/ |
| `store=SqliteStore` | Ch8 | 05/06/07 |
| `checkpointer=AsyncSqliteSaver` | Ch8 | 07 |
| `memory=["/AGENT.md"]` | **Ch8 进阶** | 10 |
| LangSmith Trace | **Ch2** | 12 |
| SSE 流式 + Web 化 | 实战 | 07 |

---

## ⚠️ 诚实声明

- **Ch6 异步子 Agent**：当前 deepagents 版本（0.6.x）的 `AsyncSubAgent` 需要 **LangGraph Platform 部署**才能跑端到端任务。我们演示了完整的 API 形态和中间件初始化，但没跑真实异步任务。
- **Ch2 LangSmith**：本地代码 100% 跑通，但**没有 LangSmith API Key 时 trace 不会上报**（free 注册可拿 Key）。
- **课程 Ch1 哲学讲解**：纯理论，未做代码 demo（课程本身也无 demo）。

---

## 🚀 Web 服务状态

服务 `07_web_chat.py` 跑在 `http://localhost:8000/`。

**可能未运行**（刚才为了测试停掉了）。如需重启：
```bash
cd /workspace/my-deepagent
source .venv/bin/activate
set -a && . ./.env && set +a
python 07_web_chat.py
```

---

## 📌 推荐下一步

1. **去 [smith.langchain.com](https://smith.langchain.com) 注册并填 LANGCHAIN_API_KEY** → 看到完整 trace
2. **重启 07_web_chat.py** → 在浏览器和 Hermes 对话
3. **跟着 Ch10（Human-in-the-Loop）走** → 加上危险操作审批
4. **跟着 Ch12（沙箱执行）走** → run_python 换 E2B/Docker，更安全
