"""Adapted Deep Image Prior for sparse-view CT; no paired training data.

Truth is used only for evaluation and a separately labelled oracle checkpoint.
The fixed endpoint is the primary result. DIP and DIP+TV share initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

# Use sequential MKL to avoid duplicate Intel OpenMP runtimes in Windows Conda + pip torch.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from run_baselines import DATASETS, ROOT, image_metrics, load_matrix, matrix_source, save_csv
from run_sparse_views import select_angles, write_json, read_json

DEFAULT_OUTPUT = ROOT / "outputs" / "extension" / "dip"
VERSION = 2


def image_vector(image: torch.Tensor) -> torch.Tensor:
    """Match the assignment's column-major vectorization without a reshape trap."""
    return image[0, 0].T.contiguous().reshape(-1, 1)


def sparse_tensor(matrix, device: str, dtype=torch.float32):
    matrix = matrix.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(matrix.indptr.astype(np.int64), device=device),
        torch.as_tensor(matrix.indices.astype(np.int64), device=device),
        torch.as_tensor(matrix.data, dtype=dtype, device=device),
        size=matrix.shape, device=device, dtype=dtype, check_invariants=True,
    )


class SparseProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vector, matrix, transpose):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, vector)

    @staticmethod
    def backward(ctx, gradient):
        return torch.sparse.mm(ctx.transpose, gradient), None, None


def mean_tv(image: torch.Tensor, epsilon: float = 1e-6):
    """Smoothed isotropic TV with forward differences and zero boundary flux."""
    down = F.pad(image[:, :, 1:, :] - image[:, :, :-1, :], (0, 0, 0, 1))
    right = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 1, 0, 0))
    return (torch.sqrt(down.square() + right.square() + epsilon**2) - epsilon).mean()


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect"),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.LeakyReLU(0.2),
    )


class PriorNet(nn.Module):
    """Four-scale encoder/decoder, four-channel skips, bilinear upsampling."""
    def __init__(self, input_channels=8, channels=(32, 64, 64, 64), skip_channels=4):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        prev = input_channels
        for width in channels:
            self.encoders.append(conv_block(prev, width))
            self.skips.append(nn.Conv2d(width, skip_channels, 1))
            prev = width
        self.bottom = conv_block(channels[-1], channels[-1])
        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for width in reversed(channels):
            self.decoders.append(conv_block(prev + skip_channels, width))
            prev = width
        self.output = nn.Conv2d(channels[0], 1, 1)

    def forward(self, noise):
        skips = []
        h = noise
        for encoder, skip in zip(self.encoders, self.skips):
            h = encoder(h)
            skips.append(skip(h))
            h = F.avg_pool2d(h, 2)
        h = self.bottom(h)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(torch.cat((h, skip), dim=1))
        return F.softplus(self.output(h))


def initialization(seed: int, size: int, device: str):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    net = PriorNet().to(device)
    noise = torch.rand(1, 8, size, size, device=device) * 0.1
    digest = hashlib.sha256(noise.detach().cpu().numpy().tobytes())
    for parameter in net.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return net, noise, digest.hexdigest()


def observation_scale(matrix, b):
    projected_one = matrix @ np.ones(matrix.shape[1])
    scale = float(np.linalg.norm(b) / np.linalg.norm(projected_one))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Observation scale must be finite and positive")
    return scale


def fit_case(A, b, truth, method, seed, args, base):
    device = args.device
    net, noise, init_hash = initialization(seed, truth.shape[0], device)
    matrix = sparse_tensor(A, device)
    transpose = sparse_tensor(A.T, device)
    b_tensor = torch.as_tensor(b, dtype=torch.float32, device=device).reshape(-1, 1)
    energy = b_tensor.square().sum()
    scale = observation_scale(A, b)
    beta = 0.0 if method == "DIP" else 10.0 ** (-base["snr_label_db"] / 10)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    best_score = float("inf")
    best_vector = None
    best_step = None
    history = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        scaled_image = net(noise)
        image = scaled_image * scale
        residual = SparseProjection.apply(image_vector(image), matrix, transpose) - b_tensor
        data_loss = residual.square().sum() / (2 * energy)
        tv_loss = mean_tv(scaled_image)
        loss = data_loss + beta * tv_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {base}, {method}, {seed}, step {step}")
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.sample_every == 0 or step == args.iterations:
            # Evaluate after the update; these metrics never enter the optimizer.
            with torch.no_grad():
                current_scaled = net(noise)
                vector = image_vector(current_scaled * scale).reshape(-1).cpu().numpy().astype(np.float64)
                current_residual = SparseProjection.apply(image_vector(current_scaled * scale), matrix, transpose) - b_tensor
                dl = float(current_residual.square().sum() / (2 * energy))
                tv = float(mean_tv(current_scaled))
            scores = image_metrics(vector, truth, A, b)
            point = {"iteration": step, "loss": dl + beta * tv, "data_loss": dl,
                     "mean_tv_scaled": tv, **scores}
            history.append(point)
            if scores["nrmse"] < best_score:
                best_score, best_step, best_vector = scores["nrmse"], step, vector.copy()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    key = f"{base['dataset']}_{base['angle_count']}angles_{base['snr_label_db']}db_{method}_seed{seed}"
    recon_dir = args.output_dir / "reconstructions"
    model_dir = args.output_dir / "models"
    recon_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = recon_dir / f"{key}_fixed.npy"
    oracle_path = recon_dir / f"{key}_oracle.npy"
    np.save(fixed_path, vector)
    np.save(oracle_path, best_vector)
    torch.save({"state_dict": {k: v.detach().cpu() for k,v in net.state_dict().items()},
                "noise": noise.detach().cpu(), "scale": scale, "iteration": args.iterations,
                "architecture": "PriorNet-v2"}, model_dir / f"{key}_fixed.pt")
    shared = {**base, "method": method, "seed": seed, "beta": beta, "observation_scale": scale,
              "initialization_sha256": init_hash, "runtime_seconds": elapsed,
              "trainable_parameters": sum(p.numel() for p in net.parameters()),
              "learning_rate": args.learning_rate, "budget_iterations": args.iterations,
              "checkpoint_stride": args.sample_every, "nonnegative": True}
    records = [
        {**shared, "checkpoint": "fixed", "iteration": args.iterations,
         "truth_used_for_checkpoint": False, "reconstruction_path": str(fixed_path.relative_to(ROOT)),
         **image_metrics(vector, truth, A, b)},
        {**shared, "checkpoint": "oracle", "iteration": best_step,
         "truth_used_for_checkpoint": True, "reconstruction_path": str(oracle_path.relative_to(ROOT)),
         **image_metrics(best_vector, truth, A, b)},
    ]
    write_json(args.output_dir / "records" / f"{key}.json", {"config": config(args), "metrics": records, "history": history})
    print(f"[DIP] {key} fixed={records[0]['nrmse']:.6f} oracle={records[1]['nrmse']:.6f}@{best_step} seconds={elapsed:.1f}", flush=True)


def config(args):
    return {"version": VERSION, "iterations": args.iterations, "sample_every": args.sample_every,
            "learning_rate": args.learning_rate, "device": args.device,
            "channels": [32, 64, 64, 64], "skip_channels": 4, "noise_channels": 8,
            "normalization": "BatchNorm2d, affine=True, track_running_stats=False, eps=1e-5",
            "noise_distribution": "Uniform[0,0.1], fixed throughout each fit",
            "optimizer": "Adam, defaults except lr", "output": "observation_scale * softplus",
            "scale": "norm(b)/norm(A@ones)", "loss": "0.5*norm(Ax-b)^2/norm(b)^2 + beta*meanTV(x/scale)",
            "TV_epsilon": 1e-6, "TV_beta": "DIP:0; DIP_TV:10**(-SNR_label_db/10)",
            "paired_initialization": True, "GT_in_training_or_amplitude_scaling": False,
            "GT_in_oracle_checkpoint": True, "torch": torch.__version__,
            "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER")}


def fingerprints(args):
    result = {}
    files = [Path(__file__), ROOT / "run_baselines.py", ROOT / "run_sparse_views.py"]
    for name in args.datasets:
        files += [matrix_source(name), ROOT / "Data" / name / "gt.csv"]
        files += [ROOT / "Data" / name / f"b_hat_{snr}db.csv" for snr in args.snrs]
    for path in files:
        stat = path.stat()
        item = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size < 10_000_000:
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(ROOT))] = item
    return result


def export_tables(output):
    records, histories = [], []
    for path in sorted((output / "records").glob("*.json")):
        payload = read_json(path)
        records.extend(payload["metrics"])
        base = payload["metrics"][0]
        for point in payload["history"]:
            histories.append({k: base[k] for k in ("dataset", "angle_count", "snr_label_db", "method", "seed")} | point)
    save_csv(output / "individual_metrics.csv", records)
    save_csv(output / "iteration_history.csv", histories)
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ("dataset", "angle_count", "snr_label_db", "method", "checkpoint"))
        groups.setdefault(key, []).append(row)
    aggregated = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(("dataset", "angle_count", "snr_label_db", "method", "checkpoint"), key))
        item["seeds"] = ",".join(str(r["seed"]) for r in rows)
        item["n_seeds"] = len(rows)
        for metric in ("nrmse", "psnr_db", "ssim", "relative_residual", "runtime_seconds", "iteration"):
            values = [r[metric] for r in rows]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(rows) > 1 else None
        item["oracle_at_budget_count"] = sum(r["iteration"] == r["budget_iterations"] for r in rows) if key[-1] == "oracle" else 0
        aggregated.append(item)
    save_csv(output / "aggregate_metrics.csv", aggregated)
    return records, aggregated, histories


def plot_results(args, records, aggregate, histories):
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "matplotlib_config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        truth = np.loadtxt(ROOT / "Data" / name / "gt.csv", delimiter=",")
        selected = [r for r in records if r["dataset"] == name and r["seed"] == 0 and r["snr_label_db"] == 30]
        if not all(any(r["angle_count"] == a and r["method"] == m and r["checkpoint"] == c
                       for r in selected) for a in (30, 15) for m in ("DIP", "DIP_TV") for c in ("fixed", "oracle")):
            continue
        fig, axes = plt.subplots(2, 5, figsize=(14, 6), layout="constrained")
        for i, angles in enumerate((30, 15)):
            axes[i, 0].imshow(truth, cmap="gray", vmin=truth.min(), vmax=truth.max())
            axes[i, 0].set_title(f"Truth | {angles} views, 30 dB")
            for j, (method, checkpoint) in enumerate((("DIP", "fixed"), ("DIP_TV", "fixed"), ("DIP", "oracle"), ("DIP_TV", "oracle")), 1):
                row = next(r for r in selected if r["angle_count"] == angles and r["method"] == method and r["checkpoint"] == checkpoint)
                x = np.load(ROOT / row["reconstruction_path"]).reshape(truth.shape, order="F")
                axes[i, j].imshow(x, cmap="gray", vmin=truth.min(), vmax=truth.max())
                axes[i, j].set_title(f"{method} {checkpoint}\nNRMSE={row['nrmse']:.3f}, step={row['iteration']}", fontsize=9)
            for ax in axes[i]:
                ax.axis("off")
        fig.suptitle(f"{name}: seed 0 illustration; tables use all three seeds")
        fig.savefig(figures / f"{name}_DIP_comparison.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, (angles, snr) in zip(axes.flat, ((30, 20), (30, 30), (15, 20), (15, 30))):
            for method in ("DIP", "DIP_TV"):
                for seed in args.seeds:
                    trace = [p for p in histories if p["dataset"] == name and p["angle_count"] == angles and p["snr_label_db"] == snr and p["method"] == method and p["seed"] == seed]
                    if trace:
                        ax.plot([p["iteration"] for p in trace], [p["nrmse"] for p in trace], label=f"{method} seed {seed}", linewidth=1)
            ax.set_title(f"{angles} views | {snr} dB")
            ax.set_xlabel("Adam iterations")
            ax.set_ylabel("NRMSE (truth evaluation)")
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.savefig(figures / f"{name}_DIP_history.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--angles", type=int, nargs="+", default=[30, 15])
    parser.add_argument("--snrs", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=["DIP", "DIP_TV"], default=["DIP", "DIP_TV"])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.sample_every < 1 or args.learning_rate <= 0:
        parser.error("iterations/sample-every must be positive integers and learning-rate > 0")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tables_only:
        records, aggregated, histories = export_tables(args.output_dir)
        plot_results(args, records, aggregated, histories)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; use --device cpu explicitly if needed")
    run_config = config(args)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["config"] != run_config:
        raise ValueError("Existing output has a different configuration; use a new --output-dir")
    inputs = fingerprints(args)
    if manifest_path.exists() and read_json(manifest_path).get("fingerprints") != inputs:
        raise ValueError("Source files changed; use a new --output-dir")
    write_json(manifest_path, {"config": run_config, "datasets": args.datasets, "angles": args.angles,
                              "snrs": args.snrs, "seeds": args.seeds, "methods": args.methods,
                              "python": platform.python_version(), "numpy": np.__version__,
                              "GPU": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                              "cuda_runtime": torch.version.cuda, "fingerprints": inputs})
    for name in args.datasets:
        full_A = load_matrix(name, ROOT / "outputs" / "cache")
        truth = np.loadtxt(ROOT / "Data" / name / "gt.csv", delimiter=",")
        for angles in args.angles:
            for snr in args.snrs:
                full_b = np.loadtxt(ROOT / "Data" / name / f"b_hat_{snr}db.csv", delimiter=",").reshape(-1)
                A, b, angle_ids, row_ids = select_angles(full_A, full_b, angles)
                base = {"dataset": name, "angle_count": angles, "snr_label_db": snr,
                        "n_rows": A.shape[0], "unknowns": A.shape[1], "zero_rows": int(np.sum(np.diff(A.indptr) == 0)),
                        "row_ids_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(),
                        "actual_snr_db": float(20 * np.log10(np.linalg.norm(A @ truth.reshape(-1, order="F")) /
                                                             np.linalg.norm(b - A @ truth.reshape(-1, order="F"))))}
                for seed in args.seeds:
                    for method in args.methods:
                        key = f"{name}_{angles}angles_{snr}db_{method}_seed{seed}"
                        path = args.output_dir / "records" / f"{key}.json"
                        if path.exists():
                            existing = read_json(path)
                            if existing["config"] != run_config:
                                raise ValueError(f"Configuration differs: {path}")
                            for record in existing["metrics"]:
                                if not (ROOT / record["reconstruction_path"]).exists():
                                    raise FileNotFoundError(record["reconstruction_path"])
                            if not (args.output_dir / "models" / f"{key}_fixed.pt").exists():
                                raise FileNotFoundError(f"Model checkpoint missing: {key}")
                            print(f"[resume] {key}", flush=True)
                            continue
                        fit_case(A, b, truth, method, seed, args, base)
                        export_tables(args.output_dir)
    records, aggregated, histories = export_tables(args.output_dir)
    plot_results(args, records, aggregated, histories)
    print(f"[complete] {len(records)//2} fits; {len(aggregated)} aggregated endpoints", flush=True)


if __name__ == "__main__":
    main()
