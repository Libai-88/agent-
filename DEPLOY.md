# Deep Agent + Open WebUI 部署指南

> Phase 1：把自研 Deep Agent 后端接到成熟前端（Open WebUI），出第一个 MVP。

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│  Docker Compose                                               │
│                                                                │
│  ┌────────────────────┐   HTTP/SSE   ┌──────────────────┐    │
│  │ open-webui:8080    │ ──────────> │ agent-api:8001   │    │
│  │  - Open WebUI      │             │  - FastAPI       │    │
│  │  - Pipe (Phase1)   │             │  - Deep Agent    │    │
│  │  - Action (VFS)    │             │  - 18_openai_... │    │
│  └────────────────────┘              └──────────────────┘    │
│         │                                  │                  │
│         ▼                                  ▼                  │
│  openwebui-data (vol)              agent-data (vol)          │
└──────────────────────────────────────────────────────────────┘
```

## 步骤

### 1. 准备 .env

```bash
cp .env.example .env
# 编辑 .env，填入：
#   SILICONFLOW_API_KEY=sk-...
#   SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
#   MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507
```

### 2. 启动

```bash
docker-compose up -d
```

第一次启动会：
- 构建 `agent-api` 镜像（装 Python 3.14 + langchain + deepagents，约 5-10 分钟）
- 拉取 `open-webui` 镜像（约 2 分钟）

查看日志：
```bash
docker-compose logs -f agent-api
docker-compose logs -f open-webui
```

### 3. 初始化 Open WebUI

访问 http://localhost:8080

- 第一个注册的账号自动成为管理员
- 登录后进入 Workspace → Models，能看到 `Deep Agent` 模型（来自 Pipe）
- 选这个模型开始聊天

### 4. 测试基础功能

**简单对话**：
```
你好，介绍下你自己
```

**触发 TODO 进度**：
```
先算 1+1=? 然后 2+2=?
```
应该看到侧边栏出现 📋 任务进度。

**触发工具调用卡片**：
问题同上，应该看到聊天里出现 🔧 run_python 折叠卡片。

**触发 HITL**：
```
执行这段代码：print("hello world")
```
应该看到消息末尾出现 🔔 等待审批，然后输入 [APPROVE] 继续。

### 5. 测试 Action

在聊天框输入：
```
/vfs
```
应该看到 web_workspace 目录树。

```
/vfs hello.txt
```
读取 hello.txt 内容。

## 文件清单

| 路径 | 作用 |
|---|---|
| `docker-compose.yml` | 两服务编排 |
| `Dockerfile.backend` | agent-api 镜像构建 |
| `.env.example` | 环境变量模板 |
| `my-deepagent/18_openai_adapter.py` | OpenAI 兼容 FastAPI 服务 |
| `my-deepagent/pipes/deep_agent_pipe.py` | Open WebUI Pipe |
| `my-deepagent/pipes/deep_agent_action.py` | VFS/历史 Action |
| `my-deepagent/17_web_chat_full.py` | 原有完整后端（保留） |

## 回滚策略

每个组件都打了 git tag，可以回滚到任一稳定点：

```bash
# 查看所有 tag
git tag -l

# 回滚到 18 adapter 完成时
git checkout phase1-adapter -- my-deepagent/18_openai_adapter.py

# 回滚整个 Open WebUI 集成
git checkout main -- docker-compose.yml Dockerfile.backend my-deepagent/pipes/
```

具体 tag 列表（开发过程中会更新）：
- `feat/open-webui-pipe-start` — 18 适配器稳定版
- `phase1-pipe` — Pipe 完成
- `phase1-docker` — Docker 化完成
- `phase1-test-pass` — 端到端测试通过

## 已知限制（Phase 1）

1. **HITL 是纯文本透传** — 用户必须手动输入 [APPROVE] / [REJECT]
   - Phase 2 切到 CopilotKit 后会有原生审批弹窗

2. **TODO 进度靠事件名解析** — 后端把 TODO 编码成 📋 开头的文本，前端 Pipe 解析
   - Phase 2 用 CopilotKit 的 STATE_SNAPSHOT 事件直接渲染

3. **工具调用卡片是折叠 HTML** — 复杂工具的渲染受 Open WebUI markdown 限制
   - Phase 2 用 React 组件全自定义

4. **Action 简化实现** — `/threads` 等命令未完整
   - 留到 Phase 2

5. **Python 3.14 vs 3.11** — 后端用 3.14，Open WebUI 用 3.11
   - Pipe 函数只用 3.11 兼容语法（已确认无问题）

## Phase 2 升级路径

切到 CopilotKit + AG-UI 时，只需：
1. 保留 18_openai_adapter.py 作为 OpenAI fallback
2. 新增 19_agui_adapter.py（AG-UI 协议端点）
3. 删掉 open-webui 服务，添加 frontend 服务（React + CopilotKit）
4. Pipe 和 Action 直接删除

后端 SSE 事件格式（todos / tool_call / tool_result / interrupt）保持不变，前端只是换了渲染层。
