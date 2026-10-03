# 数据分发说明

真值和四档带噪观测保留原 CSV。投影矩阵以 SciPy CSR 的压缩 NPZ 保存：`SheppLogan64/A_csr.npz`、`XCAT128/A_csr.npz`。

这是文件格式转换，矩阵 float64 数值、形状、非零元素和行列顺序不变。发布前逐项检查了 CSR 的 `indptr`、`indices`、`data`，与原 CSV 转换后、实际用于实验的矩阵完全相等。尺寸、非零元素数、字节数及 SHA256 见 [distribution_manifest.json](distribution_manifest.json)。

`run_baselines.matrix_source()` 优先读取本地原始 `A.csv`；没有 CSV 时使用 `A_csr.npz`。所有实验共用 `load_matrix()`，无需转换为稠密矩阵。图像向量保持 `order="F"`；选角仍按完整探测器行块同步抽取 A 和 b。

原矩阵 CSV 分别约 70.79 MiB、536.23 MiB，NPZ 分别约 1.82 MiB、10.27 MiB。原 CSV 本地保留但不提交 Git。
