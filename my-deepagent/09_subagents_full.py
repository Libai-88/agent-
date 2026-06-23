"""
09_subagents_full.py — Ch5 完整版：字典式 + general-purpose + CompiledSubAgent
===========================================================================
课程 Ch5 教的 3 种子 Agent 写法：
  1) 字典式（基础）
  2) general-purpose（万能子 Agent，主 Agent 可以动态给子任务）
  3) CompiledSubAgent（自定义预编译图）
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent  # 课程 Ch5 教的 API

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

# 一个简单的 tool：英文→中文翻译
@tool
def en_to_zh(text: str) -> str:
    """英译中（占位实现）。"""
    return f"[译] {text}"


# ============ 方式 1：字典式（Ch5 基础） ============
dict_subagent = {
    "name": "code-worker",
    "description": "擅长 Python 编程、数据计算",
    "system_prompt": "你是 code-worker。结论用 run_python 验证。",
    "tools": [],
}

# ============ 方式 2：general-purpose 子 Agent（Ch5 进阶） ============
# 特点：主 Agent 派活时**临时给一个任务描述**，子 Agent 自己规划、自己决定用哪些工具
general_subagent = {
    "name": "general-purpose",
    "description": "通用研究助手：复杂调研任务交给它",
    "system_prompt": "你是一个通用研究助手，接到任务后自主规划、自主调用工具。",
    "tools": [en_to_zh],
}

# ============ 方式 3：CompiledSubAgent（Ch5 高级） ============
# 自己写一个完整的 ReAct 图作为子 Agent
from langgraph.prebuilt import create_react_agent
sub_model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)
inner_graph = create_react_agent(sub_model, tools=[en_to_zh], prompt="你是一个翻译 Agent。")
compiled_subagent = CompiledSubAgent(
    name="translator",
    description="专业翻译 Agent，使用预编译图",
    runnable=inner_graph,
)

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt="你是调度员：编程问题派 code-worker，调研问题派 general-purpose，翻译问题派 translator。",
    subagents=[dict_subagent, general_subagent, compiled_subagent],
)

# 测试
for q in [
    "用 Python 算 5 的阶乘",                # → code-worker
    "调研一下 Deep Agents 的 3 个核心能力",   # → general-purpose
    "把 hello world 翻译成中文",              # → translator
]:
    print(f"\n>>> {q}")
    r = agent.invoke({"messages": [{"role": "user", "content": q}]})
    last = r["messages"][-1]
    # 找到是哪个 subagent 被调
    sub_used = [m for m in r["messages"] if hasattr(m, "name") and "task" in str(type(m)).lower()]
    print(f"<<< {last.content[:300]}")
