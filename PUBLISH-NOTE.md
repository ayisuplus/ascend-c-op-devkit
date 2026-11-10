# 发布说明

本仓库是作品 **ascend-c-op-devkit** 的公开快照，不是创作的唯一工作面。

| 项 | 值 |
| --- | --- |
| 源目录 | `/mnt/e/projects/昇腾社区` |
| 源仓库 HEAD | `3107f13`（导出于一点时未提交改动 11 处） |
| 导出时间 | 2026-11-10 19:27 |
| 收录 | 35 个文件 / 428K |
| 滤掉媒体文件 | 0 个 |
| 滤掉 >40MB 文件 | 0 个 |

## 为什么不完整

公开仓库只收**源码、文档、脚本与配置**。成片、素材、模型权重、工程文件（.blend / 达芬奇工程）
体积到 GB 级，GitHub 单文件上限 100MB、仓库建议 1GB，传不上去；其中一部分还包含第三方版权素材，
不适合公开分发。这些内容保留在本地源目录。

## 重新同步

源目录有更新时，重新导出再提交：

```bash
rm -rf /mnt/e/projects/public/ascend-c-op-devkit
/mnt/e/projects/.publish/export.sh ascend-c-op-devkit "/mnt/e/projects/昇腾社区"
```
