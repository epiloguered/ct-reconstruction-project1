# 传统方法的稀疏角度扩展：保留较少角度的完整投影行块，并比较五种重建方法。
# 核心数据流：select_angles() 生成子问题 -> run_dataset() 跑参数候选 -> export_tables() 汇总。
# 这里减少的是投影角度数，没有直接模拟真实 CT 的辐射剂量或光子计数。
#
"""Reproducible sparse-angle classical CT extension, including oracle grids.

Regularized LS uses (1/(2*m))*||Ax-b||^2 + lambda_normalized*R(x).
The existing solvers take raw objectives, so raw lambda = m*lambda_normalized.
GT enters metric calculation, explicitly labelled oracle selection, and prior
grid scale calibration through the known image intensity range.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from run_baselines import DATASETS, NOISE_LEVELS, ROOT, cyclic_art, image_metrics, load_matrix, matrix_source, save_csv
from run_regularization import ART_PIXEL_FRACTIONS, TV_FRACTIONS, art_with_prior, fista, spectral_lipschitz

ANGLES = (90, 45, 30, 15)
METHODS = ("LSQR", "ART_fixed", "LS_L2", "LS_TV", "ART_TV")
L2_NORMALIZED_LAMBDAS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
DEFAULT_OUTPUT = ROOT / "outputs" / "extension" / "classical"
EXTENDED_TV_FRACTIONS = (*TV_FRACTIONS, 0.03, 0.1)
EXPERIMENT_VERSION = 2


# 输入完整 A、b 和保留角度数；返回子矩阵、子观测、角度编号、原矩阵行编号。
# 一角对应连续多条探测器测量，必须整块保留；A 和 b 必须使用同一组 row_ids。
def select_angles(matrix, b: np.ndarray, angle_count: int, total_angles: int = 90):
    """Keep complete detector blocks: PPT one-based row i+(j-1)*N_lines."""
    if not isinstance(angle_count, (int, np.integer)) or not 1 <= angle_count <= total_angles:
        raise ValueError("angle_count must be an integer between 1 and total_angles")
    if matrix.shape[0] % total_angles:
        raise ValueError("Rows must divide into total_angles complete detector blocks")
    b = np.asarray(b).reshape(-1)
    if b.size != matrix.shape[0]:
        raise ValueError("b length must match A rows")
    n_lines = matrix.shape[0] // total_angles
    # 从 0 到 89 均匀抽取角度编号；例如 15 角保留 0,6,...,84，并非只保留前 15 角。
    angle_ids = np.floor(np.arange(angle_count) * total_angles / angle_count).astype(np.int64)
    # [:, None] 增加一维，通过广播把每个角度展开成它的全部探测器行号。
    row_ids = (angle_ids[:, None] * n_lines + np.arange(n_lines)).reshape(-1)
    # 只减少测量行、不减少图像像素列；b 使用相同 row_ids 保持物理对应。
    return matrix[row_ids, :].tocsr(), b[row_ids], angle_ids, row_ids


# 读取 UTF-8 JSON：保存的参数、指标和轨迹会还原成 Python 字典／列表。
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# 先写临时文件，再替换目标文件，降低中断时留下半份 JSON 的风险。
def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


# 用数据集、角度、SNR、方法和候选序号生成文件标识，防止不同条件互相覆盖。
def artifact_key(name: str, angle_count: int, snr: int, method: str, index: int) -> str:
    return f"{name}_{angle_count}angles_{snr}db_{method}_candidate{index:02d}"


# 记录算法预算、参数网格、图像展开和 oracle 规则，供复现与续跑检查。
def configuration(args) -> dict:
    return {
        "version": EXPERIMENT_VERSION, "total_angles": 90,
        "lsqr_tolerance": args.lsqr_tolerance, "lsqr_limit": args.lsqr_limit,
        "l2_tolerance": 1e-6, "l2_limit": 5000, "lsqr_conlim": 1e8,
        "tv_iterations": args.tv_iterations, "tv_prox_eps": 1e-4, "tv_prox_max_iter": 80,
        "art_epochs": 5, "art_relaxation": 1.0, "art_nonnegative_after_each_row": True,
        "L2_NORMALIZED_LAMBDAS": list(L2_NORMALIZED_LAMBDAS),
        "TV_FRACTIONS": list(EXTENDED_TV_FRACTIONS), "ART_PIXEL_FRACTIONS": list(ART_PIXEL_FRACTIONS),
        "tv_grid_history": {
            "initial_grid": list(TV_FRACTIONS), "added_for_all_32_cases": [0.03, 0.1],
            "reason": "Four XCAT initial-grid minima were at upper boundary .01; extend the common grid before final comparison",
            "initial_upper_boundary_cases": [["XCAT128", 45, 15], ["XCAT128", 30, 15],
                                             ["XCAT128", 15, 15], ["XCAT128", 15, 20]],
        },
        "selection": "oracle minimum GT NRMSE within each method/case grid",
        "vectorization": "column-major F", "truth_amplitude_used_for_prior_grid": True,
        "data_term": "squared Euclidean residual divided by 2*m; m includes zero rows",
        "angle_selection": "floor(arange(angle_count)*90/angle_count), complete detector blocks",
        "reuse_existing_full_view": False,
    }


# 记录实际输入文件及源码指纹；续跑前比较，避免把不同版本结果混成一组。
def resume_fingerprints() -> dict:
    """Protect resumed candidate files against changed solver code or input data."""
    fingerprints = {}
    for name in DATASETS:
        paths = [matrix_source(name)] + [ROOT / "Data" / name / filename
                for filename in ("gt.csv", *(f"b_hat_{snr}db.csv" for snr in NOISE_LEVELS))]
        for path in paths:
            info = path.stat()
            record = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            if path.name != "A.csv":
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprints[path.relative_to(ROOT).as_posix()] = record
    for filename in ("run_baselines.py", "run_regularization.py", "run_sparse_views.py"):
        fingerprints[filename] = {"sha256": hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()}
    return fingerprints


# 从逐候选／选中 JSON 汇总 CSV；历史原始目标值除以测量数后另列归一化值。
def export_tables(output: Path) -> tuple[list[dict], list[dict]]:
    all_rows = []
    selected_rows = []
    histories = []
    for path in sorted((output / "candidate_records").glob("*.json")):
        payload = read_json(path)
        all_rows.append(payload["metrics"])
        for point in payload.get("trace", []):
            base = payload["metrics"]
            histories.append({"dataset": base["dataset"], "angle_count": base["angle_count"],
                              "snr_label_db": base["snr_label_db"], "method": base["method"],
                              "candidate_index": base["candidate_index"],
                              "iteration": point.get("iteration", point.get("epoch", "")),
                              "objective_raw": point.get("objective", ""),
                              "objective_normalized": point.get("objective", 0) / base["n_rows"]
                                  if "objective" in point else "",
                              "nrmse": point.get("nrmse", ""), "ssim": point.get("ssim", "")})
    for path in sorted((output / "selected_records").glob("*.json")):
        selected_rows.append(read_json(path))
    save_csv(output / "candidate_metrics.csv", all_rows)
    save_csv(output / "selected_metrics.csv", selected_rows)
    save_csv(output / "iteration_history.csv", histories)
    return all_rows, selected_rows


# 遍历角度、SNR、方法和参数候选；保存每个候选，再按真值 NRMSE 选中。
def run_dataset(name: str, args) -> None:
    output = args.output_dir
    source_dir = ROOT / "Data" / name
    matrix = load_matrix(name, ROOT / "outputs" / "cache")
    truth = np.loadtxt(source_dir / "gt.csv", delimiter=",")
    amplitude = float(truth.max() - truth.min())
    x_truth = truth.reshape(-1, order="F")
    for angle_count in args.angles:
        metadata_file = output / "geometry" / f"{name}_{angle_count}angles.json"
        # Geometry does not depend on the observed values of b.
        A, _, angle_ids, row_ids = select_angles(matrix, np.zeros(matrix.shape[0]), angle_count)
        row_norms = np.asarray(A.multiply(A).sum(axis=1)).ravel()
        lipschitz = spectral_lipschitz(A)
        m = A.shape[0]
        geometry = {"dataset": name, "angle_count": angle_count, "shape": list(A.shape),
                    "nnz": A.nnz, "n_lines_per_angle": matrix.shape[0] // 90,
                    "zero_rows": int(np.count_nonzero(row_norms == 0)),
                    "nonzero_rows": int(np.count_nonzero(row_norms)),
                    "angle_ids_zero_based": angle_ids.tolist(),
                    "physical_angles_degrees": (angle_ids * 2).tolist(),
                    "lipschitz_raw": lipschitz, "lipschitz_normalized": lipschitz / m,
                    "row_ids_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(),
                    "unknowns": A.shape[1], "dimensionally_underdetermined": m < A.shape[1]}
        write_json(metadata_file, geometry)
        np.save(metadata_file.with_suffix(".rows.npy"), row_ids)
        print(f"[geometry] {name} angles={angle_count} shape={A.shape} zero={geometry['zero_rows']} L/m={lipschitz/m:.6g}", flush=True)
        clean = A @ x_truth
        for snr in args.snrs:
            full_b = np.loadtxt(source_dir / f"b_hat_{snr}db.csv", delimiter=",").reshape(-1)
            b = full_b[row_ids]
            actual_snr = float(20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(b - clean)))
            for method in METHODS:
                if method == "LS_L2":
                    values = L2_NORMALIZED_LAMBDAS
                    parameter_name = "lambda_normalized"
                elif method == "LS_TV":
                    values = EXTENDED_TV_FRACTIONS
                    parameter_name = "prox_weight_over_truth_range"
                elif method == "ART_TV":
                    values = ART_PIXEL_FRACTIONS
                    parameter_name = "epoch_prox_weight_over_truth_range"
                else:
                    values = (0.0,)
                    parameter_name = "none"
                candidate_records = []
                for index, value in enumerate(values):
                    key = artifact_key(name, angle_count, snr, method, index)
                    record_path = output / "candidate_records" / f"{key}.json"
                    array_path = output / "candidate_reconstructions" / f"{key}.npy"
                    # 候选 JSON 与数组都存在才可复用；配置和输入指纹在 main() 中先检查。
                    if args.resume and record_path.exists() and array_path.exists():
                        candidate_records.append(read_json(record_path)["metrics"])
                        continue
                    start = time.perf_counter()
                    trace = []
                    lam_raw = lam_normalized = objective_end = ""
                    prior_strength = ""
                    if method == "LSQR":
                        result = lsqr(A, b, atol=args.lsqr_tolerance, btol=args.lsqr_tolerance,
                                      iter_lim=args.lsqr_limit, show=False)
                        x, iterations, stop_code = result[0], int(result[2]), str(result[1])
                        hit_cap = iterations >= args.lsqr_limit or stop_code == "7"
                        objective_end = 0.5 * float(np.linalg.norm(A @ x - b) ** 2) / m
                    elif method == "ART_fixed":
                        _, x, _, trace, _ = cyclic_art(A, b, truth, 5, 1.0, 5)
                        iterations, stop_code, hit_cap = 5, "fixed_epochs", False
                    elif method == "LS_L2":
                        # 归一化目标的数据项除以 m；转为原始求解器目标时，正则权重要乘 m。
                        lam_normalized, lam_raw = value, value * m
                        result = lsqr(A, b, damp=np.sqrt(lam_raw), atol=1e-6, btol=1e-6,
                                      iter_lim=5000, show=False)
                        x, iterations, stop_code = result[0], int(result[2]), str(result[1])
                        hit_cap = iterations >= 5000 or stop_code == "7"
                        objective_end = (0.5 * float(np.linalg.norm(A @ x - b) ** 2 +
                                                    lam_raw * np.linalg.norm(x) ** 2)) / m
                    elif method == "LS_TV":
                        prior_strength = value * amplitude
                        lam_raw = prior_strength * lipschitz
                        lam_normalized = lam_raw / m
                        x, trace = fista(A, b, "TV", lam_raw, lipschitz, truth.shape, args.tv_iterations)
                        iterations, stop_code, hit_cap = args.tv_iterations, "fixed_iteration_budget", True
                        objective_end = trace[-1]["objective"] / m
                    else:
                        prior_strength = value * amplitude
                        x = art_with_prior(A, b, "TV", prior_strength, truth.shape, epochs=5)
                        iterations, stop_code, hit_cap = 5, "fixed_epochs", False
                    elapsed = time.perf_counter() - start
                    score = image_metrics(x, truth, A, b)
                    row = {"dataset": name, "angle_count": angle_count, "snr_label_db": snr,
                           "actual_subset_snr_db": actual_snr, "n_rows": m,
                           "n_nonzero_rows": geometry["nonzero_rows"], "method": method,
                           "candidate_index": index, "parameter_name": parameter_name,
                           "parameter_value": value, "lambda_normalized": lam_normalized,
                           "lambda_raw": lam_raw, "prior_strength": prior_strength,
                           "iterations": iterations, "stop_code": stop_code,
                           "hit_iteration_budget": hit_cap, "seconds": elapsed,
                           "objective_normalized_end": objective_end,
                           "candidate_array": str(array_path.relative_to(output)).replace("\\", "/"), **score}
                    array_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(array_path, x)
                    write_json(record_path, {"metrics": row, "trace": trace})
                    candidate_records.append(row)
                best = min(candidate_records, key=lambda r: r["nrmse"])
                selected_path = output / "reconstructions" / f"{name}_{angle_count}angles_{snr}db_{method}.npy"
                selected_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(selected_path, np.load(output / best["candidate_array"]))
                selected_row = {**best, "selection_type": "fixed baseline" if method in ("LSQR", "ART_fixed")
                                else "oracle GT NRMSE minimum",
                                "selected_grid_boundary": "lower" if best["candidate_index"] == 0 and len(values) > 1
                                    else "upper" if best["candidate_index"] == len(values)-1 and len(values) > 1 else "interior",
                                "grid_size": len(values),
                                "grid_seconds": sum(r["seconds"] for r in candidate_records),
                                "selected_array": str(selected_path.relative_to(output)).replace("\\", "/")}
                write_json(output / "selected_records" / f"{name}_{angle_count}angles_{snr}db_{method}.json", selected_row)
                export_tables(output)
                print(f"[{method}] {name} angles={angle_count} {snr}dB p={best['parameter_value']:.4g} "
                      f"NRMSE={best['nrmse']:.6f} SSIM={best['ssim']:.6f} it={best['iterations']} stop={best['stop_code']}", flush=True)


# 读取选中数组，画稀疏角度图像及指标对照；不重新求解 Ax≈b。
def plot_figures(output: Path, selected: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    lookup = {(r["dataset"], int(r["angle_count"]), int(r["snr_label_db"]), r["method"]): r for r in selected}
    for name in DATASETS:
        available = {k[1] for k in lookup if k[0] == name}
        if not available:
            continue
        for metric, label in (("nrmse", "NRMSE (lower is better)"), ("ssim", "SSIM (higher is better)")):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
            for snr, ax in zip(NOISE_LEVELS, axes.ravel()):
                for method in METHODS:
                    angles = sorted(a for a in available if (name, a, snr, method) in lookup)
                    ax.plot(angles, [lookup[(name, a, snr, method)][metric] for a in angles], marker="o", label=method)
                ax.set(xlabel="Number of projection angles", ylabel=label, title=f"{name}, {snr} dB")
                ax.set_xticks(sorted(available))
                if metric == "nrmse":
                    ax.set_yscale("log")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            fig.savefig(plots / f"{name}_angles_vs_{metric}.png", dpi=160)
            plt.close(fig)
        truth = np.loadtxt(ROOT / "Data" / name / "gt.csv", delimiter=",")
        panel_angles = sorted(available, reverse=True)
        if not all((name, a, 30, method) in lookup for a in panel_angles for method in METHODS):
            continue
        fig, axes = plt.subplots(len(panel_angles), len(METHODS)+1,
                                 figsize=(18, 3*len(panel_angles)), squeeze=False, layout="constrained")
        for i, a in enumerate(panel_angles):
            axes[i, 0].imshow(truth, cmap="gray", vmin=float(truth.min()), vmax=float(truth.max()))
            axes[i, 0].set_title(f"{a} angles | Truth", fontsize=10)
            axes[i, 0].axis("off")
            for j, method in enumerate(METHODS, start=1):
                row = lookup[(name, a, 30, method)]
                x = np.load(output / row["selected_array"])
                ax = axes[i, j]
                ax.imshow(x.reshape(truth.shape, order="F"), cmap="gray", vmin=float(truth.min()), vmax=float(truth.max()))
                ax.set_title(f"{method}\nNRMSE={row['nrmse']:.3f}, SSIM={row['ssim']:.3f}", fontsize=9)
                ax.axis("off")
        fig.savefig(plots / f"{name}_30db_angles_comparison.png", dpi=160)
        plt.close(fig)


# 命令行入口：读取参数，检查配置，再调用本文件的实验或核验流程。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--angles", nargs="+", type=int, default=list(ANGLES))
    parser.add_argument("--snrs", nargs="+", type=int, default=list(NOISE_LEVELS))
    parser.add_argument("--tv-iterations", type=int, default=600)
    parser.add_argument("--lsqr-limit", type=int, default=10000)
    parser.add_argument("--lsqr-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if any(not 1 <= a <= 90 for a in args.angles) or any(s not in NOISE_LEVELS for s in args.snrs):
        parser.error("Angles must be in 1..90 and SNRs from 15,20,25,30")
    if args.tv_iterations < 1 or args.lsqr_limit < 1 or args.lsqr_tolerance <= 0:
        parser.error("Iteration counts and tolerance must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "configuration.json"
    expected = configuration(args)
    if config_path.exists() and read_json(config_path) != expected:
        legacy = {**expected, "version": 1, "TV_FRACTIONS": list(TV_FRACTIONS)}
        legacy.pop("tv_grid_history")
        if not args.resume or read_json(config_path) != legacy:
            raise ValueError("Output directory configuration differs; use another output directory")
        write_json(args.output_dir / "configuration_initial_grid.json", read_json(config_path))
    write_json(config_path, expected)
    fingerprint_path = args.output_dir / "resume_fingerprints.json"
    fingerprints = resume_fingerprints()
    if fingerprint_path.exists() and read_json(fingerprint_path) != fingerprints:
        raise ValueError("Input data or solver source changed; use another output directory")
    write_json(fingerprint_path, fingerprints)
    started = time.perf_counter()
    if not args.plot_only:
        names = DATASETS if args.dataset == "all" else [args.dataset]
        for name in names:
            run_dataset(name, args)
    candidates, selected = export_tables(args.output_dir)
    plot_figures(args.output_dir, selected)
    write_json(args.output_dir / "run_status.json", {
        "status": "requested run completed", "dataset_argument": args.dataset, "angles_argument": args.angles,
        "snrs_argument": args.snrs, "saved_candidate_count": len(candidates), "saved_selected_count": len(selected),
        "this_invocation_seconds": time.perf_counter()-started,
        "total_saved_candidate_solver_seconds": sum(r["seconds"] for r in candidates),
        "selected_boundary_counts": {boundary: sum(r["selected_grid_boundary"] == boundary for r in selected)
                                    for boundary in ("lower", "upper", "interior")},
        "selected_solver_cap_count": sum(bool(r["hit_iteration_budget"]) for r in selected),
        "grid_version": EXPERIMENT_VERSION,
    })


if __name__ == "__main__":
    main()
