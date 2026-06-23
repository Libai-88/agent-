"""
10_memory_composite.py — Ch8 完整版：memory= + CompositeBackend
================================================================
课程 Ch8 教的两个关键能力：
  1) memory=["AGENT.md", ...]：把文件作为"长期记忆"喂给 Agent
  2) CompositeBackend：按路径路由到不同后端
       /memories/*  → 本机磁盘（持久）
       /scratch/*   → StateBackend（短期，会话级）
       /public/*    → 另一个目录
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, CompositeBackend

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

WS = Path(__file__).parent / "memory_demo"
WS.mkdir(exist_ok=True)
(WS / "memories").mkdir(exist_ok=True)
(WS / "public").mkdir(exist_ok=True)

# 预置一份"用户身份"记忆（Ch8 教的"声明性记忆"）
(WS / "memories/AGENT.md").write_text("""# 用户档案
- 姓名：李雷
- 职业：后端工程师（Python/Go）
- 偏好：简洁、表格、代码示例完整
- 时区：Asia/Shanghai
""", encoding="utf-8")

# 关键：CompositeBackend 路由
composite = CompositeBackend(
    default=StateBackend(),  # 默认走 state（短期）
    routes={
        "/memories/": FilesystemBackend(root_dir=str(WS / "memories"), virtual_mode=True),
        "/public/":   FilesystemBackend(root_dir=str(WS / "public"),   virtual_mode=True),
    }
)

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt="你是一个个人助理。优先读取 /memories/AGENT.md 来了解用户。",
    memory=["/memories/AGENT.md"],   # 关键：注入长期记忆文件
    backend=composite,              # 关键：路径路由
)

r = agent.invoke({"messages": [{"role": "user", "content": "根据你知道的关于我的信息，自我介绍下我是谁、然后给我写一段打招呼的话"}]})
print(">>> Agent 回答：")
print(r["messages"][-1].content[:500])

print("\n>>> /public 目录新文件：")
for f in (WS / "public").iterdir():
    print(f"  {f.name}")
    print(f"  {f.read_text(encoding='utf-8')[:200]}")
