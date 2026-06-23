#!/usr/bin/env python3
"""Skill 自带脚本：把本周事项字典填进模板，返回最终 Markdown。"""
import json, sys
template = open(__file__.replace("render.py", "../references/template.md")).read()
data = json.loads(sys.stdin.read())
# 简化版：直接替换占位符
out = template
for k, v in data.items():
    out = out.replace(f"[{k}]", str(v))
print(out)
