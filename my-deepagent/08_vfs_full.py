"""
08_vfs_full.py — 完整演示 Ch3 的 6 大 VFS 工具
================================================
课程 Ch3 教的：ls / read_file / write_file / edit_file / glob / grep
我们这里让 Agent 主动调用这 6 个工具——把 3 个产品文档整理成索引。
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0,
)

# 工作区：放 3 个产品文档
WS = Path(__file__).parent / "vfs_demo"
WS.mkdir(exist_ok=True)
(WS / "products").mkdir(exist_ok=True)
(WS / "products/alpha.md").write_text(
    "# 产品 Alpha\n核心卖点：高性能、低延迟。\n目标客户：金融行业。\n定价：¥999/月。\n", encoding="utf-8"
)
(WS / "products/beta.md").write_text(
    "# 产品 Beta\n核心卖点：易用、稳定。\n目标客户：中小企业。\n定价：¥299/月。\n", encoding="utf-8"
)
(WS / "products/gamma.md").write_text(
    "# 产品 Gamma\n核心卖点：开源、可扩展。\n目标客户：开发者。\n定价：免费 + 增值服务。\n", encoding="utf-8"
)

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=(
        "你是一个文档整理助手。任务：\n"
        "1) 用 glob 查找 products/*.md\n"
        "2) 用 read_file 读取每个文件\n"
        "3) 用 grep 找出所有 '定价' 相关内容\n"
        "4) 用 write_file 写一份 index.md（含产品名+定价+核心卖点表格）\n"
        "5) 用 edit_file 在 index.md 顶部插入 '更新时间：2026-06-23' 一行"
    ),
    backend=FilesystemBackend(root_dir=str(WS), virtual_mode=True),
)

result = agent.invoke({"messages": [{
    "role": "user",
    "content": "请按 system_prompt 的步骤整理 products 目录下的所有文档，产出 index.md"
}]})

# 打印 Agent 实际调用了哪些工具
print("\n=== Agent 内部工具调用轨迹 ===")
for m in result["messages"]:
    role = getattr(m, "type", m.__class__.__name__)
    if role == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
        for tc in m.tool_calls:
            print(f"  🔧 {tc['name']}({list(tc['args'].keys())})")
    elif role == "tool":
        print(f"  ✅ 工具返回: {m.content[:80]}...")

# 打印生成的 index.md
print("\n=== 生成的 index.md ===")
idx = WS / "index.md"
if idx.exists():
    print(idx.read_text(encoding="utf-8"))
else:
    print("(未生成)")
