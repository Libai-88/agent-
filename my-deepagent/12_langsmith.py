"""
12_langsmith.py — Ch2 LangSmith Trace 集成
==========================================
课程 Ch2 教的：通过环境变量启用 LangSmith trace，观测 Agent 的每一步
注意：不需要 LangSmith API Key 也能跑（无 Key 时 trace 不上报，但代码能跑）
       想看可视化 trace：去 https://smith.langchain.com 注册免费账号 → 创建 API Key → 写入环境变量
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

load_dotenv()

# ============== 启用 LangSmith Trace ==============
# 4 行环境变量：项目名 / API Key / endpoint / tracing 开关
os.environ["LANGCHAIN_PROJECT"] = "my-deepagent-demo"
# os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."  # 填入后会上报到 smith.langchain.com
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt="你是测试 Agent。",
)

print("🔍 LangSmith 配置：")
print(f"  - LANGCHAIN_PROJECT: {os.environ.get('LANGCHAIN_PROJECT')}")
print(f"  - LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"  - LANGCHAIN_API_KEY: {'已配置' if os.environ.get('LANGCHAIN_API_KEY') else '未配置（trace 不上报）'}")
print()

r = agent.invoke({"messages": [{"role": "user", "content": "用一句话解释 LangSmith 是什么"}]})
print(">>> Agent 回答：")
print(r["messages"][-1].content[:300])

print("""
📊 如何查看 trace？
1. 访问 https://smith.langchain.com 注册（免费）
2. 创建 API Key，写入 LANGCHAIN_API_KEY 环境变量
3. 重跑本脚本 → 在 smith.langchain.com 的 "my-deepagent-demo" 项目里看到完整调用链路
""")
