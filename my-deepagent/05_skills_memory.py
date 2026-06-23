"""
05_skills_memory.py — Skills + 长期记忆
=======================================
目标：让 Agent 越用越"懂你"——通过 Skills 注入"我会做什么"、通过 Store 注入"我记住了什么"。

教学点：
  - skills 参数：传一个目录路径，里面的 .md 文件就是 Agent 的"能力手册"
  - store 参数：传入一个 Store（内存版/Redis/SQLite）实现跨会话记忆
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.store.memory import InMemoryStore

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

# 1) 准备一个 Skills 目录（Agent 的人设/工作手册）
SKILLS_DIR = Path(__file__).parent / "skills"
SKILLS_DIR.mkdir(exist_ok=True)

(SKILLS_DIR / "code-review.md").write_text("""# Code Review 规范
- 每次 review 要点：可读性 / 边界条件 / 性能 / 安全性
- 输出格式：先给总评（优/良/中/差），再列 3-5 条具体建议
- 用中文
""", encoding="utf-8")

(SKILLS_DIR / "report-style.md").write_text("""# 报告风格
- 结构：摘要 / 关键发现 / 建议
- 数字一律带千分位
- 段落不超过 4 行
""", encoding="utf-8")

# 2) 创建带"长期记忆"的 Store
store = InMemoryStore()

# 预先写入一些"用户偏好"记忆
store.put(("user", "preferences"), "writing_style", {
    "tone": "concise",
    "language": "zh-CN",
    "format": "markdown",
})

# 3) 装配 Agent
agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt="你是一个通用助理，遵守 skills 目录里的所有规范。",
    skills=[str(SKILLS_DIR)],     # 关键：注入能力手册
    store=store,                  # 关键：注入长期记忆
    backend=FilesystemBackend(root_dir=str(Path(__file__).parent / "agent_workspace2")),
)

# 4) 同一 thread_id 才会命中"记忆"
config = {"configurable": {"thread_id": "user-001"}}

r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "写一段 200 字的项目周报"}]},
    config=config,
)
print("=== 第一次回答 ===")
print(r1["messages"][-1].content[:600])

# 5) 第二次对话，Agent 能从 store 里读到"用户偏好"
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "再来一段，主题：6 月 OKR 进度"}]},
    config=config,
)
print("\n=== 第二次回答（应当自动应用之前记住的风格）===")
print(r2["messages"][-1].content[:600])
