---
name: report-writer
description: 按公司模板生成周报/月报。当用户提到"周报""月报""工作报告""汇报"时触发。
---

# Report Writer Skill

按公司模板生成结构化工作报告。

## 工作流

1. 读取用户输入的本周完成事项
2. 调用 references/template.md 里的模板
3. 按以下结构输出：
   - 摘要（不超过 3 行）
   - 关键成果（分点，含数字）
   - 数据/指标（表格化）
   - 下周计划

## 风格要求

- 中文
- 数字带千分位
- 表格优先
- 不超过 1 页

## 模板

参见 [references/template.md](references/template.md)
