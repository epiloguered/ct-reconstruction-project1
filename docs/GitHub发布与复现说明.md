# GitHub 发布与复现说明

## 1. 仓库与上传范围

目标仓库：`epiloguered/ct-reconstruction-project1`，私有，主分支 `main`。

上传 Python 入口与测试、Conda 配置、原作业 PPT、两份主要实验文档、压缩投影矩阵、真值与带噪观测，以及正式实验 CSV／JSON／图像／重建数组／48 个模型权重。保留传统实验的候选数组和参数记录，便于核对 oracle 选参。

`.gitignore` 排除 `.conda/`、缓存、原始巨大 `A.csv`、Matplotlib 配置与 pip 文件、试运行与初次退化网络目录、发布检查副本、个人重新运行的 `outputs/rerun/`。这些文件仍在原本地项目保留。未额外选择开源许可证。

既有实验的 manifest 和日志记录原始执行环境与源码指纹，保留为历史证据。

## 2. 压缩矩阵与代码适配

两个实际用于实验的 CSR 矩阵复制到 `Data/<名称>/A_csr.npz`，逐项检查索引与 float64 数值完全相等。SHA256、形状、非零元素数写入 `Data/distribution_manifest.json`。本次没有重新生成 A、b 或 GT。

`run_baselines.py` 新增 `matrix_source()`；`load_matrix()` 在没有 CSV 时直接读取 NPZ，并检查列数、有限值及 CSR 格式。`run_sparse_views.py`、`run_dip_extension.py` 的输入指纹同步选择实际矩阵文件。新增测试模拟只有压缩数据的下载目录。

## 3. 下载后复核

在具备此私有仓库权限且 GitHub Git 已登录的电脑上：

```powershell
git clone https://github.com/epiloguered/ct-reconstruction-project1.git
cd ct-reconstruction-project1
conda env create -p .conda -f environment.yml
$env:MKL_THREADING_LAYER='SEQUENTIAL'
& '.\.conda\python.exe' -m unittest -v test_dip_extension.py test_sparse_views.py test_baselines.py test_regularization.py
& '.\.conda\python.exe' verify_extension.py --summary outputs/github_release/audit_summary.json
```

本机环境已实际运行测试。整份 environment.yml 在另一台电脑从零创建仍未验证；CUDA 包与 GPU／驱动兼容性需要在目标电脑检查。独立核验重算已有结果，并恢复指定的两个模型，不等于重新训练全部网络。

## 4. 重新跑实验

使用 README 的 `outputs/rerun/` 命令。本次发布使数据来源路径与读取源码发生变化，不要对下载的历史结果直接使用 `--resume`；指纹检查会正确拒绝不一致的源码／输入。自己重新生成的目录可以在数据、代码、配置不变时续跑。

比较新旧结果时，保持数据集、角度、SNR、参数网格、迭代预算和种子一致。GPU 重训不承诺逐位一致。传统权重和网络 oracle 截点用真值辅助选择；网络主表采用固定 2000 步终点，评价边界见实验文档。

## 5. 后续提交

```powershell
git status --short
git add README.md docs run_baselines.py
git diff --cached --stat
git commit -m "Describe the changes"
git push
```

`git add` 指定自己实际修改的文件。先检查暂存内容，再提交与推送；不要加入 `.conda` 或原矩阵 CSV。修改算法后使用新的输出目录，同步更新实施记录与结果文档。

## 6. 本次发布检查证据

正式上传前，使用 `git checkout-index` 导出只含暂存文件的副本。副本不含原始矩阵 CSV、Conda 环境或缓存，用原本机 Conda Python 实际读取 NPZ 并运行独立核验：`passed`，7953 个断言、864 个传统候选、160 个选中重建、48 次网络拟合、96 个截点，错误和缺失均为空；两次模型恢复均与保存输出完全相等。具体记录见 [发布版本核验.json](发布版本核验.json)。这证明发布文件集合可复核已有结果；尚未在另一台电脑从零安装环境或重训全部实验。

发布前暂存文件约 244 MiB，最大单文件为 XCAT 压缩矩阵（10.27 MiB）。14 项测试通过。原始历史产物未被发布检查覆盖写入。
