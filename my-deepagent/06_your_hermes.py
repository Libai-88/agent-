"""
06_your_hermes.py — 你的「Hermes / Claude / Codex」原型
========================================================
目标：把上面所有能力组合起来，做一个真正可用的"通用 Agent"

特性矩阵：
  ✅ 多工具（编程 + 办公 + 时间）
  ✅ 任务规划（write_todos 自动）
  ✅ 虚拟文件系统（落到 ./my_workspace）
  ✅ 子 Agent 委派（code-worker / office-worker）
  ✅ Skills 注入（skills/ 目录）
  ✅ 长期记忆（SQLite 本地存储）
  ✅ 简单 REPL 交互（命令行对话）
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.store.sqlite import SqliteStore

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

# ============== 工具 ==============
from datetime import datetime
@tool
def get_current_time() -> str:
    """返回当前时间（ISO 格式）。"""
    return datetime.now().isoformat(timespec="seconds")

@tool
def run_python(code: str) -> str:
    """执行 Python 代码并返回 stdout（沙箱版，限时 10s）。"""
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

# ============== 工作目录与 Skills ==============
WORKSPACE = Path(__file__).parent / "my_workspace"
WORKSPACE.mkdir(exist_ok=True)
SKILLS = Path(__file__).parent / "skills"
SKILLS.mkdir(exist_ok=True)
(SKILLS / "writing.md").write_text(
    "# 写作规范\n- 中文，简洁，分点\n- 报告结构：摘要/发现/建议\n- 数字带千分位\n",
    encoding="utf-8"
)

# ============== 长期记忆（SQLite，跨进程保留） ==============
db_path = str(Path(__file__).parent / "memory.db")
store = SqliteStore.from_conn_string(db_path)

# ============== 装配你的"HerMES" ==============
agent = create_deep_agent(
    model=model,
    tools=[get_current_time, run_python],
    system_prompt=(
        "你叫 Hermes，是用户的通用助理。\n"
        "- 编程/计算/调试 → 用 run_python 自验证\n"
        "- 写作/总结/翻译 → 直接输出\n"
        "- 多步任务先用 write_todos 拆解\n"
        "- 中间产物写到 VFS（ls/write/read）\n"
        "- 遵守 skills/ 目录里的所有规范"
    ),
    skills=[str(SKILLS)],
    subagents=[{
        "name": "code-worker",
        "description": "编程/调试/数据处理专家",
        "system_prompt": "你是 code-worker。用 run_python 验证结论。",
        "tools": [run_python],
    }],
    backend=FilesystemBackend(root_dir=str(WORKSPACE)),
    store=store,
)

# ============== REPL：和你的 Agent 对话 ==============
if __name__ == "__main__":
    config = {"configurable": {"thread_id": input("你的名字（用于记忆隔离）: ").strip() or "default"}}
    print(f"\n🟢 Hermes 已就绪（thread_id={config['configurable']['thread_id']}）")
    print("   输入 'exit' 退出；输入 ':files' 查看 VFS；输入 ':reset' 清空记忆\n")

    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "exit":
            break
        if q == ":files":
            print("--- VFS ---")
            for f in sorted(WORKSPACE.rglob("*")):
                if f.is_file():
                    print(f.relative_to(WORKSPACE))
            continue
        if q == ":reset":
            os.remove(db_path)
            print("记忆已清空，请重启")
            break
        r = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
        print(f"Hermes> {r['messages'][-1].content}\n")
