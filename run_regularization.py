# 正则化实验：在拟合 Ax≈b 的同时限制图像的大小、稀疏性或相邻像素变化。
# LS+L2 用阻尼 LSQR；LS+L1/TV 用 FISTA；ART 在每轮结束后施加图像先验。
# 本实验用已知真值选参数（oracle），不是未知真值时的自动选参方案。
#
"""Run L2/L1/TV regularization experiments for LS and five-sweep ART."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr
from skimage.restoration import denoise_tv_chambolle

from run_baselines import DATASETS, NOISE_LEVELS, ROOT, image_metrics, load_matrix, save_csv

L2_LAMBDAS = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
L1_FRACTIONS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
TV_FRACTIONS = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01)
ART_PIXEL_FRACTIONS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3)
ART_L2_STRENGTHS = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0)
METHODS = ("LS_L2", "LS_L1", "LS_TV", "ART_L2", "ART_L1", "ART_TV")


# 幂迭代估计 A.T@A 的最大特征值，作为二次误差梯度的步长尺度 L。
# 返回估计值的 1.05 倍作为余量；这不是对任意矩阵都严格成立的上界证明。
def spectral_lipschitz(matrix, iterations: int = 80) -> float:
    """Estimate the quadratic gradient Lipschitz constant with a 5% step margin."""
    rng = np.random.default_rng(2026)
    vector = rng.standard_normal(matrix.shape[1])
    vector /= np.linalg.norm(vector)
    last = 0.0
    for _ in range(iterations):
        product = matrix.T @ (matrix @ vector)
        norm = np.linalg.norm(product)
        if norm == 0:
            raise ValueError("Projection matrix is zero")
        vector = product / norm
        last = float(vector @ (matrix.T @ (matrix @ vector)))
    return 1.05 * last


# 各向同性 TV（总变差）：把每个像素的水平／垂直差分合成为长度，再求和。
def tv_isotropic(image: np.ndarray) -> float:
    down = np.zeros_like(image)
    right = np.zeros_like(image)
    # 前向差分；最后一行／列保持零，相当于不跨越图像边界计算邻域差异。
    down[:-1] = image[1:] - image[:-1]
    right[:, :-1] = image[:, 1:] - image[:, :-1]
    return float(np.hypot(down, right).sum())


# 近端算子：在靠近输入向量与减小正则项之间折中；L1 为软阈值，TV 为迭代去噪。
def proximal(vector: np.ndarray, kind: str, strength: float, shape: tuple[int, int]) -> np.ndarray:
    if kind == "L1":
        # 软阈值保留符号、缩小绝对值，小于阈值的分量归零，鼓励像素域稀疏。
        return np.sign(vector) * np.maximum(np.abs(vector) - strength, 0.0)
    if kind == "TV":
        image = vector.reshape(shape, order="F")
        denoised = denoise_tv_chambolle(image, weight=strength, eps=1e-4, max_num_iter=80)
        return denoised.reshape(-1, order="F")
    raise ValueError(kind)


# 计算 R(x) 的值，用于记录目标函数；真正更新图像的操作在 proximal()。
def regularizer(vector: np.ndarray, kind: str, shape: tuple[int, int]) -> float:
    if kind == "L1":
        return float(np.abs(vector).sum())
    if kind == "TV":
        return tv_isotropic(vector.reshape(shape, order="F"))
    raise ValueError(kind)


# FISTA（快速迭代收缩阈值算法）：梯度步拟合数据，近端步施加先验，动量加速迭代。
# 输入包括权重 lam、步长尺度 lipschitz 和图像形状；返回重建向量与目标值轨迹。
def fista(matrix, b: np.ndarray, kind: str, lam: float, lipschitz: float,
          shape: tuple[int, int], iterations: int) -> tuple[np.ndarray, list[dict]]:
    """FISTA for 0.5||Ax-b||^2 + lam*R(x), R=L1 or isotropic TV."""
    if kind not in ("L1", "TV") or lam < 0 or lipschitz <= 0:
        raise ValueError("Invalid FISTA inputs")
    x = np.zeros(matrix.shape[1], dtype=np.float64)
    y = x.copy()
    momentum = 1.0
    at_b = matrix.T @ b
    history: list[dict] = []
    for step in range(1, iterations + 1):
        gradient = matrix.T @ (matrix @ y) - at_b
        # 步长为 1/L；近端权重必须同步取 lam/L，不能只调整梯度步长。
        new_x = proximal(y - gradient / lipschitz, kind, lam / lipschitz, shape)
        new_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        # y 是带动量的外推点，x 才是本轮解；加速不保证每轮目标值都单调下降。
        y = new_x + ((momentum - 1) / new_momentum) * (new_x - x)
        x = new_x
        momentum = new_momentum
        if step == 1 or step % 50 == 0 or step == iterations:
            residual = matrix @ x - b
            objective = 0.5 * float(residual @ residual) + lam * regularizer(x, kind, shape)
            history.append({"iteration": step, "objective": objective})
    return x, history


# 每完整遍历 A 一轮，再施加一次 L2／L1／TV 先验，并保持非负。
# 这是 ART 与先验组合的启发式流程，不能直接当作精确求解正则化 LS 的算法。
def art_with_prior(matrix, b: np.ndarray, kind: str, strength: float,
                   shape: tuple[int, int], epochs: int = 5) -> np.ndarray:
    """Cyclic ART with a proximal image prior after each complete sweep."""
    if kind not in ("L2", "L1", "TV") or strength < 0:
        raise ValueError("Invalid ART prior inputs")
    x = np.zeros(matrix.shape[1], dtype=np.float64)
    indptr, indices, values = matrix.indptr, matrix.indices, matrix.data
    squared_norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    valid_rows = np.flatnonzero(squared_norms > 0)
    for _ in range(epochs):
        for i in valid_rows:
            start, stop = indptr[i], indptr[i + 1]
            cols = indices[start:stop]
            row = values[start:stop]
            correction = (b[i] - np.dot(row, x[cols])) / squared_norms[i]
            x[cols] = np.maximum(0.0, x[cols] + correction * row)
        if kind == "L2":
            x /= 1 + strength
        elif kind == "L1":
            x = np.maximum(0.0, x - strength)
        else:
            x = np.maximum(0.0, proximal(x, "TV", strength, shape))
    return x


# 读取基线与正则化数组，按相同数据集／SNR 和灰度范围绘制对照。
def write_figures(name: str, truth: np.ndarray, output: Path, selected: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_config"))
    (output / "matplotlib_config").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_lookup = {(int(r["snr_label_db"]), r["method"]): r for r in selected}
    baseline_dir = ROOT / "outputs" / name / "reconstructions"
    recon_dir = output / "reconstructions"
    plot_dir = output / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    low, high = float(truth.min()), float(truth.max())
    for family, methods in (("LS", ("LSQR", "LS_L2", "LS_L1", "LS_TV")),
                            ("ART", ("ART_fixed", "ART_L2", "ART_L1", "ART_TV"))):
        fig, axes = plt.subplots(4, 5, figsize=(15, 12), layout="constrained")
        for row_index, snr in enumerate(NOISE_LEVELS):
            panels = [("Truth", truth, None)]
            for method in methods:
                path = (baseline_dir if method in ("LSQR", "ART_fixed") else recon_dir) / f"{name}_{snr}db_{method}.npy"
                image = np.load(path).reshape(truth.shape, order="F")
                panels.append((method, image, selected_lookup.get((snr, method))))
            for col_index, (title, image, score) in enumerate(panels):
                ax = axes[row_index, col_index]
                ax.imshow(image, cmap="gray", vmin=low, vmax=high)
                ax.set_title(f"{snr} dB | {title}", fontsize=10)
                ax.axis("off")
                if score is not None:
                    ax.text(0.5, -0.05, f"NRMSE {float(score['nrmse']):.3f}",
                            transform=ax.transAxes, ha="center", va="top", fontsize=8)
        fig.savefig(plot_dir / f"{name}_{family}_regularization.png", dpi=150)
        plt.close(fig)


# 逐档噪声、逐种方法遍历候选参数，用真值 NRMSE 选中并保存最优候选。
def run_dataset(name: str, args: argparse.Namespace) -> None:
    data_dir = ROOT / "Data" / name
    output = args.output_dir / name
    recon_dir = output / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_matrix(name, ROOT / "outputs" / "cache")
    truth = np.loadtxt(data_dir / "gt.csv", delimiter=",", dtype=float)
    shape = truth.shape
    amplitude = float(truth.max() - truth.min())
    lipschitz = spectral_lipschitz(matrix)
    print(f"[data] {name}: lipschitz_bound={lipschitz:.6g}", flush=True)
    all_rows: list[dict] = []
    selected: list[dict] = []
    convergence: list[dict] = []
    for snr in NOISE_LEVELS:
        b = np.loadtxt(data_dir / f"b_hat_{snr}db.csv", delimiter=",", dtype=float).reshape(-1)
        for method in METHODS:
            if method == "LS_L2":
                candidates = [("lambda", value) for value in L2_LAMBDAS]
            elif method in ("LS_L1", "LS_TV"):
                fractions = L1_FRACTIONS if method == "LS_L1" else TV_FRACTIONS
                candidates = [("prox_weight_over_range", frac) for frac in fractions]
            elif method == "ART_L2":
                candidates = [("epoch_prox_strength", value) for value in ART_L2_STRENGTHS]
            else:
                candidates = [("epoch_prox_weight_over_range", frac) for frac in ART_PIXEL_FRACTIONS]
            best: tuple[dict, np.ndarray, list[dict]] | None = None
            for parameter_name, parameter_value in candidates:
                started = time.perf_counter()
                trace: list[dict] = []
                lam: float | str = ""
                if method == "LS_L2":
                    lam = parameter_value
                    # SciPy 的 damp 在目标中按平方出现，所以 L2 权重 lam 对应 damp=sqrt(lam)。
                    result = lsqr(matrix, b, damp=np.sqrt(lam), atol=1e-6, btol=1e-6,
                                  iter_lim=5000, show=False)
                    x = result[0]
                    iterations, stop_code = int(result[2]), int(result[1])
                    objective_end = 0.5 * float(np.linalg.norm(matrix @ x - b) ** 2 + lam * np.linalg.norm(x) ** 2)
                elif method in ("LS_L1", "LS_TV"):
                    lam = parameter_value * amplitude * lipschitz
                    iteration_limit = args.l1_iterations if method == "LS_L1" else args.tv_iterations
                    x, trace = fista(matrix, b, method[-2:], lam, lipschitz, shape, iteration_limit)
                    iterations, stop_code = iteration_limit, "iteration_limit"
                    objective_end = trace[-1]["objective"]
                else:
                    kind = method[-2:]
                    strength = parameter_value if kind == "L2" else parameter_value * amplitude
                    x = art_with_prior(matrix, b, kind, strength, shape, args.art_epochs)
                    iterations, stop_code = args.art_epochs, "fixed_epochs"
                    objective_end = ""  # ART with a prior is not the LS objective above.
                seconds = time.perf_counter() - started
                score = image_metrics(x, truth, matrix, b)
                row = {"dataset": name, "snr_label_db": snr, "method": method,
                       "parameter_name": parameter_name, "parameter_value": parameter_value,
                       "objective_lambda": lam, "iterations": iterations, "stop_code": stop_code,
                       "seconds": seconds, "objective_end": objective_end, **score}
                all_rows.append(row)
                # 使用真值误差选网格最优参数（oracle）；真实未知图像不能直接照搬此选参规则。
                if best is None or score["nrmse"] < best[0]["nrmse"]:
                    best = (row, x.copy(), trace)
            assert best is not None
            row, x, trace = best
            selected.append(row)
            np.save(recon_dir / f"{name}_{snr}db_{method}.npy", x)
            for point in trace:
                convergence.append({"dataset": name, "snr_label_db": snr, "method": method,
                                    "parameter_value": row["parameter_value"], **point})
            print(f"[{method}] {name} {snr}dB: parameter={row['parameter_value']:.4g}, "
                  f"nrmse={row['nrmse']:.5f}, iterations={row['iterations']}, stop={row['stop_code']}", flush=True)
            save_csv(output / "candidate_metrics.csv", all_rows)
            save_csv(output / "selected_metrics.csv", selected)
    save_csv(output / "selected_convergence.csv", convergence)
    write_figures(name, truth, output, selected)


# 命令行入口：读取参数，检查配置，再调用本文件的实验或核验流程。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "regularized")
    parser.add_argument("--l1-iterations", type=int, default=1200)
    parser.add_argument("--tv-iterations", type=int, default=600)
    parser.add_argument("--art-epochs", type=int, default=5)
    args = parser.parse_args()
    if args.l1_iterations < 1 or args.tv_iterations < 1 or args.art_epochs < 1:
        parser.error("Iteration counts must be positive")
    args.output_dir = args.output_dir.resolve()
    chosen = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name in chosen:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
