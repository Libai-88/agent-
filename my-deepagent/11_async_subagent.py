"""
11_async_subagent.py — Ch6 异步子 Agent 演示
============================================
课程 Ch6 教的是 Async Subagent（异步子 Agent）—— 主 Agent 派活后**立即返回任务 ID**，
子 Agent 在**远程 Agent Protocol 服务器**上跑，主 Agent 不阻塞、可中途追加指令。

⚠️ 当前 deepagents 版本（0.6.x）的 AsyncSubAgent **必须配合 LangGraph Platform**：
   - 托管版：https://langchain-ai.github.io/langgraph/cloud/
   - 自托管：部署 LangGraph Server（Docker/K8s）

本文件演示：
  1) 完整的 API 形态（字段、参数）
  2) 部署到 LangGraph Platform 后的代码写法
  3) 同步 vs 异步的对比表

如要本地调试，需先：langgraph build → 部署到 LangGraph Platform → 拿到 graph_id
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

# ============== 异步子 Agent 规范 ==============
async_researcher = AsyncSubAgent(
    name="researcher-async",
    description="深度调研专家（异步），处理耗时几分钟的调研任务",
    graph_id="research_agent",   # 远程 LangGraph Platform 上的 graph 名
    # url="https://your-deployment.us.langgraph.app",  # 部署到 LangSmith Deployments 后填入
    # headers={"x-api-key": "..."},  # 自托管时用
)

# 尝试启用异步子 Agent（需要 LangSmith/LangGraph Platform 部署才能跑通）
try:
    middleware = [AsyncSubAgentMiddleware(async_subagents=[async_researcher])]
    agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt="你是调度员。耗时的调研任务派给 researcher-async。",
        middleware=middleware,
    )
    print("✅ 异步子 Agent 中间件已启用（需要 LangGraph Platform 部署才能跑通实际任务）")
except Exception as e:
    print(f"⚠️ 异步子 Agent 中间件初始化失败：{e}")
    print("   → 需要先部署 Agent 到 LangGraph Platform：https://langchain-ai.github.io/langgraph/cloud/")

# 同步 vs 异步对比（Ch6 核心教学点）
print("""
================================================================
📊 同步 vs 异步子 Agent 对比
================================================================
| 维度         | 同步（task 工具）       | 异步（Async Subagent）    |
|--------------|------------------------|--------------------------|
| 阻塞？       | ✅ 主 Agent 阻塞等返回  | ❌ 立即返回 task_id      |
| 后端要求     | 任意                   | 必须 Agent Protocol 服务 |
| 适合场景     | < 60s 的小任务         | 几分钟到几小时的大任务   |
| 中途追加指令 | ❌ 不支持              | ✅ 支持                  |
| 中途取消     | ❌ 不支持              | ✅ 支持                  |
| 部署成本     | 0                      | 需要 LangGraph Platform  |
================================================================
""")
