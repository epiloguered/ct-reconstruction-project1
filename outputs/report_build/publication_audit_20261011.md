# 2026-10-11 GitHub 更新准备审计

## 检查范围与当前状态

- 仓库：`https://github.com/epiloguered/ct-reconstruction-project1`。
- 本次只读检查 Git 状态、文件大小、发布约定、典型凭据格式和入口依赖；未修改 `.gitignore`、未提交、未推送、未运行实验。
- `git ls-remote origin refs/heads/main` 成功，远程提交为 `1bdae00a1f989c390645774c699ba4c6375b4bd7`，Git 远程读取认证可用。
- 既有 `docs/GitHub发布与复现说明.md` 将仓库记录为私有；当前 `gh` 不在 PATH，常见安装位置未找到，本次未独立核验 GitHub 的实时可见性。
- 私有仓库建议在报告中称为“项目 GitHub 仓库”或“代码仓库”，添加地址即可；不得由地址或“开源”措辞推断其已改为公开。

## 文件规模快照

以下统计涵盖已跟踪文件和 Git 可见的未跟踪文件，不包含已有 `.gitignore` 排除项。它是修改报告前的快照，最终提交前应再次核对暂存区。

| 范围 | 文件数 | 大小 MiB | 用途与建议 |
| --- | ---: | ---: | --- |
| 全集 | 3898 | 406.94 | 现有版本加新实验和报告 |
| 本次未跟踪文件 | 1471 | 163.17 | 正式新实验、报告及少量临时文件 |
| `outputs/extension/` | 2296 | 211.98 | 已提交的传统稀疏角度和 DIP 正式实验；保留 |
| `outputs/improvements/` | 112 | 63.61 | 留出投影 DIP 的记录、24 个新权重、选中数组和汇总；保留 |
| `outputs/improvements_refined/` | 454 | 34.57 | 非负 TV/TGV 的正式结果及候选数组；保留 |
| `outputs/min_angles/` | 723 | 52.96 | 非负 TV、小波+TV、PnP 的少角度正式结果、候选与诊断；保留 |
| `outputs/min_angles_boundary/` | 105 | 3.14 | 7/8/9 角补充边界实验；保留 |
| `outputs/report_build/` | 32 | 7.32 | 报告来源 JSON、审计和构建证据；保留必要源数据，排除渲染临时文件 |
| `report/` | 8 | 1.30 | DOCX、Markdown、图片和阅读说明；保留正式产物 |
| `outputs/final/` | 8 | 0.13 | 报告用汇总 CSV 和来源哈希；保留 |

没有超过 100 MiB 的候选文件，最大单文件为已跟踪的 `Data/XCAT128/A_csr.npz`（10.27 MiB）。现阶段不需要为这一批文件引入 Git LFS。

## 应排除项

发现 `report/` 下的 Word 锁文件 `~$*.docx`，它属于打开文档时产生的临时文件。建议新增以下 `.gitignore` 条目：

```gitignore
# Office 打开文档时的锁文件
~$*
# 报告排版检查时生成的 PDF 与逐页 PNG
outputs/report_build/render/
```

`outputs/report_build/renderer_initial.log` 为初次尝试渲染的失败日志，交付与再生成不依赖它，可额外排除。不要整体忽略 `outputs/report_build/`：`tools/prepare_report_bibliography.py` 和报告生成器读取其中的参考文献及审计 JSON。

既有排除规则仍应保留：`.conda/`、`outputs/cache/`、原巨大 `Data/*/A.csv`、`__pycache__/`、Matplotlib/pip 缓存、试运行和退化网络、发布检查副本、个人 `outputs/rerun/`。已排除的 `outputs/improvements/classical/` 是本轮精修前的中间实验目录，正文主结果读取 `outputs/improvements_refined/classical/`。

## 凭据与个人环境路径

- 对 Git 可见、10 MB 以下的常见文本文件扫描 GitHub token、AWS access key、私钥头和 Authorization Bearer 格式，未命中。扫描只报告文件名与模式名，不输出任何疑似值。这是限定格式检查，不是秘密信息不存在的绝对证明。
- 运行环境 JSON 中保留 `C:\Users\wsh\...\.conda\python.exe`，属于历史执行位置与来源信息，不是凭据；建议保留实验 provenance。
- `tools/check_report.py` 当前硬编码本机 Poppler 路径，上传不会泄露凭据，但他人需要修改路径才能运行排版检查。建议改为参数、环境变量或 PATH 查找，或在报告再生成说明中注明配置方法。

## 建议上传范围与最终门槛

1. 上传当前源码、测试、依赖配置、两份详细文档、正式报告、报告生成脚本、正式实验 CSV/JSON/数组/权重/候选及输入压缩矩阵，延续既有发布约定。
2. 不上传 Word 锁文件、报告渲染中间图/PDF、环境、缓存或试运行。
3. 更新报告完成后再暂存；核对暂存文件列表和最大文件、运行既有验证命令、确认参考链接和报告字体/目录，再提交推送。
4. 推送后读取远程 `main` 提交并与本地 `HEAD` 对照，才可报告 GitHub 已同步。

