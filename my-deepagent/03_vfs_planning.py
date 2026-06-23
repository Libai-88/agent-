"""
03_vfs_planning.py — 虚拟文件系统 + 任务规划
=============================================
目标：让 Agent 拿到"分析 sales.csv 并出报告"这种多步任务时，
     1) 主动列 todo 拆任务（write_todos）
     2) 把中间结果落进 VFS（write_file）
     3) 最后的报告也写在 VFS 里（你可以截取/落盘）

教学点：
  - 把本地目录"挂载"到 Agent 的虚拟文件系统（backend=）
  - 通过 interrupt 之前的 messages 看 Agent 内部决策
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend  # 把 VFS 落到本机目录

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

# 把 VFS 挂到 ./agent_workspace 目录 —— Agent 写的"虚拟文件"会真实落盘
WORKSPACE = os.path.join(os.path.dirname(__file__), "agent_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

# 把 sales.csv 也复制到工作区，让 Agent 能 ls 到它
import shutil
shutil.copy(os.path.join(os.path.dirname(__file__), "sales.csv"),
            os.path.join(WORKSPACE, "sales.csv"))

agent = create_deep_agent(
    model=model,
    tools=[],  # 这版不挂外部工具，让 Agent 专注"思考+写文件"
    system_prompt=(
        "你是一个数据分析师。接到任务时：\n"
        "1) 先用 write_todos 拆成 2-4 步\n"
        "2) 把每一阶段的产物写到 VFS（如 analysis.md, chart_desc.md, report.md）\n"
        "3) 最后产出放在 final_report.md\n"
        "4) 用中文，所有数字保留千分位"
    ),
    backend=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),  # 关键：把虚拟文件落地
)

task = (
    "请分析 sales.csv（5 个产品 × 4 个季度）：\n"
    "1) 算出每个产品的全年总销售额与同比增长（Q4 vs Q1）\n"
    "2) 找出最值得关注的 1 个产品\n"
    "3) 写一份 1 页的市场报告 final_report.md，包含：摘要、关键发现、建议"
)

print(f">>> 任务：{task}\n")
result = agent.invoke({"messages": [{"role": "user", "content": task}]})

# 打印 Agent 内部轨迹（消息流）
print("\n=== Agent 内部决策轨迹（节选）===")
for i, m in enumerate(result["messages"]):
    role = m.type if hasattr(m, "type") else m.__class__.__name__
    content = (m.content if isinstance(m.content, str) else str(m.content))[:200]
    print(f"[{i:02d}] {role}: {content}")

print(f"\n=== VFS 落盘内容（{WORKSPACE}）===")
for f in sorted(os.listdir(WORKSPACE)):
    full = os.path.join(WORKSPACE, f)
    if os.path.isfile(full):
        print(f"\n--- {f} ---")
        with open(full) as fp:
            print(fp.read()[:1500])
