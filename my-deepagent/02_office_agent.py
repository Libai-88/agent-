"""
02_office_agent.py — 通用型 Agent（办公+编程双能力）
=====================================================
目标：在最小版基础上，挂上"办公 + 编程"两类工具，让 Agent 真正能干活
教学点：
  1) 工具就是 Python 函数，Deep Agent 会自动转成 LLM 可调用的工具
  2) system_prompt 是给 Agent 的"岗位说明书"，越具体越好
  3) 这就是 Hermes / Claude / Codex 的"原型"
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from langchain_core.tools import tool

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0,
)

# ============================================================
# 自定义工具集（办公 + 编程）
# ============================================================

# --- 办公类 ---
@tool
def get_current_time() -> str:
    """返回当前时间（ISO 格式），用于邮件/报告落款。"""
    return datetime.now().isoformat(timespec="seconds")


@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """把长文本压缩成几句话（占位：实际生产用 LLM 调用更稳）。"""
    sentences = [s.strip() for s in text.replace("。", ".").split(".") if s.strip()]
    return "。".join(sentences[:max_sentences]) + ("。" if sentences else "")


# --- 编程类 ---
@tool
def run_python(code: str) -> str:
    """
    在受限沙箱里执行一段 Python 代码，返回 stdout。
    注意：演示用途；生产环境请用 Docker / E2B 等真沙箱。
    """
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        out = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=15
        )
        return (out.stdout or "") + (("\n[stderr]\n" + out.stderr) if out.stderr else "")
    except subprocess.TimeoutExpired:
        return "[error] 执行超时（>15s）"
    finally:
        os.unlink(path)


# ============================================================
# 系统提示词（"岗位说明书"）
# ============================================================
SYSTEM_PROMPT = """你是一个通用型个人助理（对标 Hermes/Claude/Codex），同时擅长两件事：

【办公任务】邮件、报告、会议纪要、信息检索、数据汇总 —— 输出要简洁、清晰、带小标题。
【编程任务】理解需求、写代码、运行代码、根据报错迭代 —— 优先用 run_python 自验证。

工作原则：
1. 接到复杂任务时，先在内部规划 1-3 步再动手
2. 涉及文件输出时，把内容写入虚拟文件系统（ls/write/read）
3. 任何时候不确定，宁可先调用工具确认，也不要瞎猜
4. 用中文回答
"""

agent = create_deep_agent(
    model=model,
    tools=[get_current_time, summarize_text, run_python],
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    # 试试两类典型任务
    for q in [
        "用 Python 帮我算一下 1 到 100 的平方和，并用 run_python 跑出来给我看",
        "把这段文字压缩成 2 句话：'Deep Agents 是 LangChain 团队基于 LangGraph 构建的高级 Agent 框架，"
        "它通过内置的任务规划、虚拟文件系统、子 Agent 委派等能力，"
        "让开发者可以快速构建生产级的 AI Agent 系统。'",
    ]:
        print(f"\n>>> 用户：{q}")
        r = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"<<< Agent：{r['messages'][-1].content[:600]}")
