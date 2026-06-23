"""
01_minimal.py — 第一个 Deep Agent（最小可跑版）
================================================
目标：验证环境配置 OK，让 Agent 能调用工具
教学点：create_deep_agent 一行就能造出"会规划+会写文件+能调工具"的 Agent
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults  # 可选
from deepagents import create_deep_agent

load_dotenv()

# 1) 选模型 —— 硅基流动 OpenAI 兼容接口
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0,
)

# 2) 给 Agent 一组工具（先给个"互联网搜索"工具让它能联网）
#    没装 tavily 的话可以注释掉这一行
tools = []
if os.getenv("TAVILY_API_KEY"):
    tools.append(TavilySearchResults(max_results=3))

# 3) 一句话造 Agent —— 关键 API
agent = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个通用助手，擅长用中文清晰回答问题。先规划再执行。",
)

# 4) 跑起来
result = agent.invoke({
    "messages": [{"role": "user", "content": "请用 3 句话介绍 Deep Agents 的核心设计思想。"}]
})

print("\n=== Agent 回答 ===")
print(result["messages"][-1].content)
