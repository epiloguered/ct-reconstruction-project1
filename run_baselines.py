"""Run reproducible LSQR and cyclic ART baselines for Project 1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
DATASETS = {"SheppLogan64": 64, "XCAT128": 128}
NOISE_LEVELS = (15, 20, 25, 30)


def read_sparse_csv(path: Path, ncols: int) -> sparse.csr_matrix:
    """Stream a dense CSV into CSR without allocating its full dense matrix."""
    indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    indptr = [0]
    with path.open("r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle, start=1):
            row = np.fromstring(line, sep=",", dtype=np.float64)
            if row.size != ncols:
                raise ValueError(f"{path}: row {row_number} has {row.size} columns, expected {ncols}")
            nonzero = np.flatnonzero(row)
            indices_parts.append(nonzero.astype(np.int32, copy=False))
            values_parts.append(row[nonzero])
            indptr.append(indptr[-1] + nonzero.size)
    if len(indptr) == 1:
        raise ValueError(f"Empty matrix: {path}")
    indices = np.concatenate(indices_parts)
    values = np.concatenate(values_parts)
    matrix = sparse.csr_matrix(
        (values, indices, np.asarray(indptr, dtype=np.int32)),
        shape=(len(indptr) - 1, ncols),
    )
    matrix.sort_indices()
    return matrix


def matrix_source(name: str) -> Path:
    """Prefer original CSV locally; published checkouts use lossless CSR NPZ."""
    directory = ROOT / "Data" / name
    for filename in ("A.csv", "A_csr.npz"):
        path = directory / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"No A.csv or A_csr.npz in {directory}")


def load_matrix(name: str, cache_dir: Path, force: bool = False) -> sparse.csr_matrix:
    source = matrix_source(name)
    ncols = DATASETS[name] ** 2
    if source.suffix == ".npz":
        matrix = sparse.load_npz(source).tocsr()
        if matrix.shape[1] != ncols or matrix.shape[0] == 0 or not np.isfinite(matrix.data).all():
            raise ValueError(f"Invalid published matrix: {source}, shape={matrix.shape}")
        matrix.check_format(full_check=True)
        print(f"[distributed] {name}: shape={matrix.shape}, nnz={matrix.nnz}", flush=True)
        return matrix
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{name}_A_csr.npz"
    metadata_path = cache_dir / f"{name}_A_csr.json"
    source_info = {"size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns, "ncols": ncols}
    if not force and cached.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == source_info:
            matrix = sparse.load_npz(cached).tocsr()
            print(f"[cache] {name}: shape={matrix.shape}, nnz={matrix.nnz}", flush=True)
            return matrix
    started = time.perf_counter()
    matrix = read_sparse_csv(source, ncols)
    sparse.save_npz(cached, matrix, compressed=True)
    metadata_path.write_text(json.dumps(source_info, indent=2), encoding="utf-8")
    print(f"[convert] {name}: shape={matrix.shape}, nnz={matrix.nnz}, seconds={time.perf_counter()-started:.1f}", flush=True)
    return matrix


def image_metrics(x: np.ndarray, truth: np.ndarray, matrix: sparse.csr_matrix, b: np.ndarray) -> dict:
    size = truth.shape[0]
    estimate = x.reshape((size, size), order="F")
    difference = estimate - truth
    mse = float(np.mean(difference**2))
    data_range = float(truth.max() - truth.min())
    return {
        "nrmse": float(np.linalg.norm(difference) / np.linalg.norm(truth)),
        "psnr_db": float(10 * np.log10(data_range**2 / mse)) if mse > 0 else float("inf"),
        "ssim": float(structural_similarity(truth, estimate, data_range=data_range)),
        "relative_residual": float(np.linalg.norm(matrix @ x - b) / np.linalg.norm(b)),
    }


def cyclic_art(
    matrix: sparse.csr_matrix,
    b: np.ndarray,
    truth: np.ndarray,
    epochs: int,
    relaxation: float,
    fixed_epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    """Cyclic Kaczmarz with the slide's per-row nonnegativity projection."""
    if not 0 < relaxation < 2:
        raise ValueError("ART relaxation must be in (0, 2)")
    if not 1 <= fixed_epoch <= epochs:
        raise ValueError("ART fixed epoch must be between 1 and total epochs")
    x = np.zeros(matrix.shape[1], dtype=np.float64)
    indptr, indices, values = matrix.indptr, matrix.indices, matrix.data
    squared_norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    valid_rows = np.flatnonzero(squared_norms > 0)
    history: list[dict] = []
    best_x = x.copy()
    fixed_x = x.copy()
    best_epoch = 0
    best_nrmse = float("inf")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        for i in valid_rows:
            start, stop = indptr[i], indptr[i + 1]
            cols = indices[start:stop]
            row = values[start:stop]
            correction = relaxation * (b[i] - np.dot(row, x[cols])) / squared_norms[i]
            x[cols] = np.maximum(0.0, x[cols] + correction * row)
        score = image_metrics(x, truth, matrix, b)
        record = {"epoch": epoch, **score, "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        if epoch == fixed_epoch:
            fixed_x = x.copy()
        if score["nrmse"] < best_nrmse:
            best_nrmse = score["nrmse"]
            best_epoch = epoch
            best_x = x.copy()
    return x, fixed_x, best_x, history, best_epoch


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(name: str, truth: np.ndarray, out_dir: Path, rows: list[dict], histories: dict[int, list[dict]]) -> None:
    # Matplotlib's cache must stay in the writable project workspace.
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "matplotlib_config"))
    (out_dir / "matplotlib_config").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(int(row["snr_label_db"]), row["method"]): row for row in rows}
    low, high = float(truth.min()), float(truth.max())
    fig, axes = plt.subplots(4, 4, figsize=(13, 13), layout="constrained")
    for ridx, snr in enumerate(NOISE_LEVELS):
        panels = [("Ground truth", truth)]
        for method in ("LSQR", "ART_fixed", "ART_best"):
            vector = np.load(out_dir / "reconstructions" / f"{name}_{snr}db_{method}.npy")
            panels.append((method.replace("_", " "), vector.reshape(truth.shape, order="F")))
        for cidx, (title, image) in enumerate(panels):
            ax = axes[ridx, cidx]
            ax.imshow(image, cmap="gray", vmin=low, vmax=high)
            ax.set_title(f"{snr} dB | {title}")
            ax.axis("off")
            if cidx:
                result = by_key[(snr, ("LSQR", "ART_fixed", "ART_best")[cidx - 1])]
                ax.text(0.5, -0.05, f"NRMSE={result['nrmse']:.4f}, SSIM={result['ssim']:.4f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=8)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{name}_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for snr in NOISE_LEVELS:
        history = histories[snr]
        ax.plot([row["epoch"] for row in history], [row["nrmse"] for row in history], label=f"{snr} dB")
    ax.set(xlabel="ART epoch", ylabel="NRMSE", title=f"{name}: ART convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(figure_dir / f"{name}_art_convergence.png", dpi=160)
    plt.close(fig)


def run_dataset(name: str, args: argparse.Namespace) -> None:
    dataset_dir = ROOT / "Data" / name
    out_dir = args.output_dir / name
    (out_dir / "reconstructions").mkdir(parents=True, exist_ok=True)
    matrix = load_matrix(name, args.output_dir / "cache", args.force_cache)
    truth = np.loadtxt(dataset_dir / "gt.csv", delimiter=",", dtype=np.float64)
    if truth.shape != (DATASETS[name], DATASETS[name]):
        raise ValueError(f"Unexpected ground-truth shape for {name}: {truth.shape}")
    x_truth = truth.reshape(-1, order="F")
    if matrix.shape[1] != x_truth.size:
        raise ValueError("A column count does not match ground-truth image")
    clean_projection = matrix @ x_truth
    squared_row_norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    print(f"[data] {name}: zero_rows={np.count_nonzero(squared_row_norms == 0)}, nnz={matrix.nnz}", flush=True)
    results: list[dict] = []
    histories: dict[int, list[dict]] = {}
    for snr in NOISE_LEVELS:
        b = np.loadtxt(dataset_dir / f"b_hat_{snr}db.csv", delimiter=",", dtype=np.float64).reshape(-1)
        if b.size != matrix.shape[0]:
            raise ValueError(f"Unexpected b length for {name} {snr} dB")
        actual_snr = float(20 * np.log10(np.linalg.norm(clean_projection) / np.linalg.norm(b - clean_projection)))

        started = time.perf_counter()
        ls_result = lsqr(matrix, b, atol=args.lsqr_tolerance, btol=args.lsqr_tolerance,
                         iter_lim=args.lsqr_limit, show=False)
        ls_seconds = time.perf_counter() - started
        ls_x = ls_result[0]
        np.save(out_dir / "reconstructions" / f"{name}_{snr}db_LSQR.npy", ls_x)
        ls_metrics = image_metrics(ls_x, truth, matrix, b)
        results.append({"dataset": name, "snr_label_db": snr, "actual_projection_snr_db": actual_snr,
                        "method": "LSQR", "iterations": int(ls_result[2]), "stop_code": int(ls_result[1]),
                        "seconds": ls_seconds, **ls_metrics})
        print(f"[LSQR] {name} {snr}dB: it={ls_result[2]}, stop={ls_result[1]}, nrmse={ls_metrics['nrmse']:.5f}, s={ls_seconds:.1f}", flush=True)

        started = time.perf_counter()
        final_x, fixed_x, best_x, history, best_epoch = cyclic_art(
            matrix, b, truth, args.art_epochs, args.relaxation, args.art_fixed_epoch
        )
        art_seconds = time.perf_counter() - started
        histories[snr] = history
        for record in history:
            record.update({"dataset": name, "snr_label_db": snr})
        np.save(out_dir / "reconstructions" / f"{name}_{snr}db_ART_final.npy", final_x)
        np.save(out_dir / "reconstructions" / f"{name}_{snr}db_ART_fixed.npy", fixed_x)
        np.save(out_dir / "reconstructions" / f"{name}_{snr}db_ART_best.npy", best_x)
        for method, vector, epoch in (("ART_fixed", fixed_x, args.art_fixed_epoch),
                                      ("ART_final", final_x, args.art_epochs),
                                      ("ART_best", best_x, best_epoch)):
            metrics = image_metrics(vector, truth, matrix, b)
            results.append({"dataset": name, "snr_label_db": snr, "actual_projection_snr_db": actual_snr,
                            "method": method, "iterations": epoch, "stop_code": "",
                            "seconds": art_seconds, **metrics})
        print(f"[ART]  {name} {snr}dB: best_epoch={best_epoch}, best_nrmse={history[best_epoch-1]['nrmse']:.5f}, s={art_seconds:.1f}", flush=True)
        save_csv(out_dir / "metrics.csv", results)
        save_csv(out_dir / "art_history.csv", [record for level in NOISE_LEVELS if level in histories for record in histories[level]])
    save_figures(name, truth, out_dir, results, histories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--art-epochs", type=int, default=40)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--art-fixed-epoch", type=int, default=5)
    parser.add_argument("--lsqr-limit", type=int, default=10000)
    parser.add_argument("--lsqr-tolerance", type=float, default=1e-5)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    if args.art_epochs < 1 or args.lsqr_limit < 1:
        parser.error("Iteration limits must be positive")
    if not 1 <= args.art_fixed_epoch <= args.art_epochs:
        parser.error("--art-fixed-epoch must be between 1 and --art-epochs")
    args.output_dir = args.output_dir.resolve()
    chosen = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name in chosen:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
