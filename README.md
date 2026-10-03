# Project 1：CT 重建实验

两份主要文档：

- [实验实施记录](docs/实验实施记录.md)：环境、数据约定、算法、实际排障、复现和修改入口。
- [实验结果](docs/基线实验结果.md)：基线、正则化、稀疏角度、DIP 的表格、图像和结果边界。

- [GitHub 发布与复现说明](docs/GitHub发布与复现说明.md)：上传范围、数据格式、下载验证和后续提交。

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

所有图像向量按列展开和还原：`reshape(shape, order="F")`。A 使用 CSR 稀疏格式；选角必须保持连续探测器行块，并同步抽取 b。

传统正则化结果按真值 NRMSE 选参数，属于 oracle 实验。DIP 的主结果是固定 2000 次终点，另外单独保存真值辅助的 oracle 截点诊断；三个种子全部纳入均值与样本标准差。训练不使用真值损失。试运行、退化网络的记录仅在原本地项目保留，不上传；目录为 `pilot*` 和 `dip_initial_unstable` 子目录，不纳入正式结果。
