# Project 1：CT 重建实验

## 作业报告入口（2026-10-11）

- [正式实验报告 Word 版](report/CT图像重建与稀疏角度优化实验报告修订版.docx)：提交作业前填写姓名、学号。
- [同源 Markdown 版](report/CT图像重建与稀疏角度优化实验报告修订版.md)：便于检索和修改。
- [报告阅读与再生成说明](report/README.md)：主线、证据来源和图表更新方法。

报告按“90角基线→正则化→少角度扩展”组织；DIP、TGV和小波探索在附录。指标与方法在正文相应位置引用论文，并注明实现差异和真值辅助选参限制。以下两份详细文档保留完整实施和历史结果记录。

当前修订版首页带目录，汉字宋体、英文数字Times New Roman。第九节的40行主表位于报告第14页；机器可读版本为 [各方法最佳结果CSV](outputs/final/best_by_method.csv)。

两份主要文档：

- [实验实施记录](docs/实验实施记录.md)：环境、数据约定、算法、实际排障、复现和修改入口。
- [实验结果](docs/基线实验结果.md)：基线、正则化、稀疏角度、DIP 的表格、图像和结果边界。

2026-10-10更正：少角度质量应对照**原90角LS+TV／ART+TV**，不是历史15角结果。最新三指标对照、设置差异、高SSIM的区域诊断见结果文档第14节；复查可运行 `& '.\.conda\python.exe' audit_fullview_comparison.py`。

- [GitHub 发布与复现说明](docs/GitHub发布与复现说明.md)：上传范围、数据格式、下载验证和后续提交。

- [初学者代码阅读指南](docs/初学者代码阅读指南.md)：阅读顺序、变量形状、关键约定和修改后的检查方法。

## 在当前项目环境运行

在本目录打开 PowerShell：

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
& '.\.conda\python.exe' -m unittest -v test_dip_extension.py test_sparse_views.py test_baselines.py test_regularization.py
& '.\.conda\python.exe' verify_extension.py --summary outputs/github_release/audit_summary.json
& '.\.conda\python.exe' plot_extension_summary.py
```

首次建环境可用 `conda env create -p .conda -f environment.yml`；该整份 YAML 重建没有另行验证，本机使用的是已实际安装并执行的项目环境。CUDA PyTorch 需要相容 GPU／驱动；本次使用 RTX 5070。

## 下载后重新运行

仓库包含无损 `Data/<数据集>/A_csr.npz`，不需要原始巨大 `A.csv`。原结果保留历史源码／输入指纹。本次发布增加了压缩数据读取功能，重新实验必须使用新输出目录：

```powershell
& '.\.conda\python.exe' run_baselines.py --output-dir outputs/rerun/baselines
& '.\.conda\python.exe' run_regularization.py --output-dir outputs/rerun/regularized
& '.\.conda\python.exe' run_sparse_views.py --output-dir outputs/rerun/classical
& '.\.conda\python.exe' run_dip_extension.py --output-dir outputs/rerun/dip
```

续跑自己生成的扩展实验时，在相同命令后加 `--resume`，保持数据、代码和配置不变。`outputs/rerun/` 默认不提交。

## 入口与输出

| 入口 | 任务 | 输出目录 |
| --- | --- | --- |
| `run_baselines.py` | 90 角 LSQR、ART | `outputs/<数据集>/` |
| `run_regularization.py` | 90 角 L2／L1／TV | `outputs/regularized/` |
| `run_sparse_views.py` | 90／45／30／15 角，五种传统方法 | `outputs/extension/classical/` |
| `run_dip_extension.py` | 30／15 角、20／30 dB，DIP／DIP+TV，三个种子 | `outputs/extension/dip/` |
| `verify_extension.py` | 完整性与保存结果独立复核 | `outputs/extension/audit_summary.json` |
| `plot_extension_summary.py` | 共同条件的指标与图像对照 | `outputs/extension/summary/` |
| `run_sparse_improvements.py` | 非负 TV、二阶 TGV、留出投影选择 DIP+TV 截点 | `outputs/improvements_refined/classical/`、`outputs/improvements/neural/` |
| `summarize_sparse_improvements.py` | 新实验独立指标核验、模型恢复、表格与图像 | `outputs/improvements/summary/` |
| `run_min_angles.py` | 非负TV、小波＋TV、PnP-BM3D，探索30至3角 | `outputs/min_angles/`、`outputs/min_angles_boundary/` |
| `run_bm3d_profiles.py`、`run_boundary_profiles.py` | 常规BM3D配置及7至9角边界检查 | `outputs/min_angles/profile_check/`、`profile_boundary/` |
| `summarize_min_angles.py` | 全部新产物核验、重跑变化、角度与质量权衡 | `outputs/min_angles/summary/` |
| `plot_min_angle_images.py` | 只重画已核验图像 | `outputs/min_angles/summary/*_images.png` |

所有图像向量按列展开和还原：`reshape(shape, order="F")`。A 使用 CSR 稀疏格式；选角必须保持连续探测器行块，并同步抽取 b。

新增少角度改进实验的过程见 [实施记录第13节](docs/实验实施记录.md#13-少角度改进实验2026-10-10)，完整三项图像指标见 [结果第12节](docs/基线实验结果.md#12-少角度改进实验非负-tvtgv-与投影验证截点)。传统方法仍用 GT 选参；新增网络截点只用留出投影选择。重新运行请为每阶段指定新的 `--output-dir`，避免历史源码指纹冲突。

不使用DIP的最新实验见 [结果第13节](docs/基线实验结果.md#13-非-dip-的更少角度实验小波tv-与-pnp-bm3d) 与 [实施记录第14节](docs/实验实施记录.md#14-非-dip-的更少角度实验2026-10-10)。新增依赖用 `.\.conda\python.exe -m pip install -r requirements-min-angles.txt` 安装。24项测试通过，704个正式数组的产物核验通过；4线程BM3D存在重跑变化，审计为 `passed_with_replay_variation`，并非严格逐位复现。传统参数和部分预算截点仍由GT选择。

传统正则化结果按真值 NRMSE 选参数，属于 oracle 实验。DIP 的主结果是固定 2000 次终点，另外单独保存真值辅助的 oracle 截点诊断；三个种子全部纳入均值与样本标准差。训练不使用真值损失。试运行、退化网络的记录仅在原本地项目保留，不上传；目录为 `pilot*` 和 `dip_initial_unstable` 子目录，不纳入正式结果。
