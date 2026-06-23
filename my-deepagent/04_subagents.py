"""
04_subagents.py — 子 Agent 委派（团队作战）
=============================================
目标：主 Agent 把"编程任务"派给 code-worker、"办公任务"派给 office-worker。
     每个子 Agent 有独立上下文隔离，互不污染。

教学点：
  - subagents 参数：注册一组专门的小 Agent
  - 主 Agent 通过内置的 `task` 工具把子任务甩给它们
  - 每个子 Agent 有独立 VFS、独立的上下文窗口
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)


# ============== 编程子 Agent 的工具 ==============
@tool
def run_python(code: str) -> str:
    """执行一段 Python 代码并返回 stdout（沙箱版本，限时 10s）。"""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        out = subprocess.run(["python", path], capture_output=True, text=True, timeout=10)
        return (out.stdout or "") + (("\n[stderr]\n" + out.stderr) if out.stderr else "")
    except subprocess.TimeoutExpired:
        return "[error] timeout"
    finally:
        os.unlink(path)


# ============== 办公子 Agent 的工具 ==============
@tool
def word_count(text: str) -> int:
    """统计一段文本的字数。"""
    return len(text)


# ============== 主 Agent 路由逻辑 ==============
ROUTER_PROMPT = """你是一个通用助理的"调度员"。

- 编程/计算/调试类问题 → 派给 code-worker
- 写作/总结/翻译/格式化类问题 → 派给 office-worker
- 简单闲聊 → 直接回答

派活时使用 task 工具，明确说明输入和期望产出。
"""

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=ROUTER_PROMPT,
    subagents=[
        {
            "name": "code-worker",
            "description": "编程/数据处理/算法类专家，擅长用 run_python 自验证",
            "system_prompt": "你是 code-worker。所有结论必须用 run_python 验证。中文回答。",
            "tools": [run_python],
        },
        {
            "name": "office-worker",
            "description": "办公写作类专家，擅长总结、润色、翻译、格式化",
            "system_prompt": "你是 office-worker。输出简洁、结构化、用中文。",
            "tools": [word_count],
        },
    ],
)

if __name__ == "__main__":
    for q in [
        "用 Python 计算斐波那契数列前 20 项的和",
        "把这句话润色成商务邮件开场白：'你好，我想问一下你们的产品能不能打折'",
    ]:
        print(f"\n>>> {q}")
        r = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"<<< {r['messages'][-1].content[:500]}")
