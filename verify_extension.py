# 独立结果核验：检查覆盖、重算指标、核对 oracle 选择与种子统计，并恢复两个指定模型。
# 结构检查不等于数值检查；核验保存的产物不等于重新训练全部模型。
#
"""Audit saved extension artifacts without rerunning either reconstruction solver.

Default: require the complete 864-candidate/160-selected classical experiment and
48 DIP fits, recompute saved metrics, check oracle checkpoints and sample standard
deviations, and restore one fixed model for each dataset. For an experiment still
running, use --allow-partial --structure-only; its incomplete inventory is reported
without treating the absent future cases as corrupt results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch
import numpy as np
from skimage.metrics import structural_similarity

from run_baselines import ROOT, DATASETS, load_matrix
from run_dip_extension import PriorNet, image_vector

METRICS = ("nrmse", "psnr_db", "ssim", "relative_residual")
CLASSICAL_METHODS = ("LSQR", "ART_fixed", "LS_L2", "LS_TV", "ART_TV")


# 集中保存核验计数、失败证据与缺失条件；部分结果也必须明确报告未完成项。
class Audit:
    # 初始化核验计数和证据容器。
    def __init__(self, allow_partial: bool):
        self.allow_partial = allow_partial
        self.errors = []
        self.incomplete = []
        self.checked = 0
        self.counts = {}
        self.model_restorations = []

    # 增加一条核验计数，失败时保存标签与证据，继续收集其余问题。
    def check(self, condition, label: str, detail=None):
        self.checked += 1
        if not condition:
            self.errors.append({"check": label, "detail": detail})

    # 比较实际／预期条件集合，分别记录意外条件和缺失条件。
    def coverage(self, actual: set, expected: set, label: str):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        self.check(not unexpected, label + ": unexpected conditions", unexpected)
        if missing:
            self.incomplete.append({"check": label, "missing_count": len(missing), "missing": missing})
            if not self.allow_partial:
                self.check(False, label + ": missing conditions", missing)

    # 使用相对和绝对容差比较浮点数，避免把正常舍入误差当成产物损坏。
    def close(self, actual, expected, label, rtol=1e-10, atol=1e-11):
        self.check(bool(np.isclose(float(actual), float(expected), rtol=rtol, atol=atol)),
                   label, {"actual": float(actual), "expected": float(expected)})


# 读取 UTF-8 JSON，将参数、指标和轨迹还原成字典／列表。
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# 每行读成字典；CSV 字段仍为字符串，计算前需要转成数值。
def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# 将条件字段转成统一类型的元组，用作 CSV、JSON 和数组之间的对应标识。
def key_of(row, fields):
    integers = {"angle_count", "snr_label_db", "seed", "candidate_index"}
    return tuple(int(row[f]) if f in integers else row[f] for f in fields)


# 独立实现四项指标，不调用训练时的指标函数，降低同一实现错误被重复验证的风险。
def independent_metrics(vector, truth, matrix, observation):
    """Compute the documented definitions independently of the training helper."""
    estimate = vector.reshape(truth.shape, order="F")
    error = estimate - truth
    mse = float(np.mean(np.square(error)))
    image_range = float(np.ptp(truth))
    return {
        "nrmse": float(np.sqrt(np.sum(error * error) / np.sum(truth * truth))),
        "psnr_db": float(10 * np.log10(image_range * image_range / mse)) if mse else float("inf"),
        "ssim": float(structural_similarity(truth, estimate, data_range=image_range)),
        "relative_residual": float(np.sqrt(np.sum(np.square(matrix @ vector - observation)) /
                                            np.sum(observation * observation))),
    }


# 缓存 A、真值和选角子问题，避免复算多个结果时反复加载大矩阵。
class DataAccess:
    # 建立数据集和测量条件的缓存容器。
    def __init__(self):
        self.matrices = {}
        self.truths = {}
        self.cases = {}

    # 按数据集／角度／SNR 取得同一测量子问题，并缓存供指标复算。
    def case(self, name, angles, snr):
        key = (name, angles, snr)
        if key not in self.cases:
            if name not in self.matrices:
                self.matrices[name] = load_matrix(name, ROOT / "outputs" / "cache")
                self.truths[name] = np.loadtxt(ROOT / "Data" / name / "gt.csv", delimiter=",")
            matrix = self.matrices[name]
            detector_count = matrix.shape[0] // 90
            ids = np.floor(np.arange(angles) * 90 / angles).astype(np.int64)
            rows = (ids[:, None] * detector_count + np.arange(detector_count)).ravel()
            full_b = np.loadtxt(ROOT / "Data" / name / f"b_hat_{snr}db.csv", delimiter=",").ravel()
            self.cases[key] = (matrix[rows].tocsr(), full_b[rows], self.truths[name], rows)
        return self.cases[key]


# 检查候选与选中记录覆盖、数组指标，以及网格内真值最优候选的选择。
def audit_classical(audit, output, data, structure_only):
    config = read_json(output / "configuration.json")
    grid = {"LSQR": [0.0], "ART_fixed": [0.0],
            "LS_L2": config["L2_NORMALIZED_LAMBDAS"], "LS_TV": config["TV_FRACTIONS"],
            "ART_TV": config["ART_PIXEL_FRACTIONS"]}
    candidate_fields = ("dataset", "angle_count", "snr_label_db", "method", "candidate_index")
    selected_fields = candidate_fields[:-1]
    cases = list(itertools.product(DATASETS, (90, 45, 30, 15), (15, 20, 25, 30), CLASSICAL_METHODS))
    expected_candidates = {(*case, i) for case in cases for i in range(len(grid[case[-1]]))}
    candidates = {}
    for path in sorted((output / "candidate_records").glob("*.json")):
        row = read_json(path)["metrics"]
        key = key_of(row, candidate_fields)
        audit.check(key not in candidates, "classical candidate uniqueness", path.name)
        candidates[key] = row
        audit.check((output / row["candidate_array"]).is_file(), "classical candidate array exists", path.name)
        if key in expected_candidates:
            audit.close(row["parameter_value"], grid[row["method"]][key[-1]], "classical grid value " + path.stem)
    selected = {}
    for path in sorted((output / "selected_records").glob("*.json")):
        row = read_json(path)
        key = key_of(row, selected_fields)
        audit.check(key not in selected, "classical selected uniqueness", path.name)
        selected[key] = row
        audit.check((output / row["selected_array"]).is_file(), "classical selected array exists", path.name)
    audit.counts["classical"] = {"candidate_records": len(candidates), "expected_candidates": 864,
                                  "selected_records": len(selected), "expected_selected": 160}
    audit.check(len(expected_candidates) == 864, "classical configured grid count must be 864")
    audit.coverage(set(candidates), expected_candidates, "classical candidates")
    audit.coverage(set(selected), set(cases), "classical selected")
    if structure_only:
        return
    candidate_csv = read_csv(output / "candidate_metrics.csv")
    selected_csv = read_csv(output / "selected_metrics.csv")
    candidate_csv_keys = {key_of(r, candidate_fields) for r in candidate_csv}
    audit.check(len(candidate_csv) == len(candidate_csv_keys), "classical candidate CSV has no duplicates")
    audit.coverage(candidate_csv_keys, expected_candidates, "classical candidate CSV")
    csv_map = {key_of(r, selected_fields): r for r in selected_csv}
    audit.check(len(selected_csv) == len(csv_map), "classical selected CSV has no duplicates")
    audit.coverage(set(csv_map), set(cases), "classical selected CSV")
    for key, row in selected.items():
        A, b, truth, rows = data.case(*key[:3])
        vector = np.load(output / row["selected_array"])
        audit.check(vector.shape == (A.shape[1],) and bool(np.isfinite(vector).all()),
                    "classical finite vector with expected shape " + str(key))
        recomputed = independent_metrics(vector, truth, A, b)
        for metric in METRICS:
            audit.close(row[metric], recomputed[metric], "classical recomputed " + metric + str(key))
            if key in csv_map:
                audit.close(csv_map[key][metric], row[metric], "classical JSON/CSV " + metric + str(key))
        selected_candidate = candidates.get((*key, int(row["candidate_index"])))
        audit.check(selected_candidate is not None, "classical selected candidate present " + str(key))
        if selected_candidate:
            audit.check(bool(np.array_equal(vector, np.load(output / selected_candidate["candidate_array"]))),
                        "classical selected vector equals candidate " + str(key))
        cohort = [r for k, r in candidates.items() if k[:-1] == key]
        if cohort:
            audit.close(row["nrmse"], min(r["nrmse"] for r in cohort), "classical oracle minimum " + str(key))
            audit.check(int(row["grid_size"]) == len(cohort), "classical complete per-method grid " + str(key))
        audit.check(row["n_rows"] == A.shape[0], "classical row count " + str(key))
        if row["method"] in ("LS_L2", "LS_TV"):
            audit.close(row["lambda_raw"], float(row["lambda_normalized"]) * A.shape[0],
                        "classical normalized lambda " + str(key))
        geometry = read_json(output / "geometry" / f"{key[0]}_{key[1]}angles.json")
        audit.check(geometry["row_ids_sha256"] == hashlib.sha256(rows.tobytes()).hexdigest(),
                    "classical independent row hash " + str(key))


# 核对网络条件／种子覆盖、fixed/oracle 截点、配对初始化和样本标准差。
def audit_dip(audit, output, data, structure_only):
    manifest = read_json(output / "manifest.json")
    config = manifest["config"]
    fit_fields = ("dataset", "angle_count", "snr_label_db", "seed", "method")
    expected_fits = set(itertools.product(DATASETS, (30, 15), (20, 30), (0, 1, 2), ("DIP", "DIP_TV")))
    payloads, rows = {}, {}
    for path in sorted((output / "records").glob("*.json")):
        payload = read_json(path)
        audit.check(payload["config"] == config, "DIP configuration matches manifest " + path.stem)
        audit.check(len(payload["metrics"]) == 2, "DIP two endpoints " + path.stem)
        fit = key_of(payload["metrics"][0], fit_fields)
        audit.check(fit not in payloads, "DIP fit uniqueness", path.name)
        payloads[fit] = payload
        for row in payload["metrics"]:
            endpoint = (*key_of(row, fit_fields), row["checkpoint"])
            audit.check(endpoint not in rows, "DIP endpoint uniqueness", path.name)
            rows[endpoint] = row
            audit.check((ROOT / row["reconstruction_path"]).is_file(), "DIP saved endpoint exists", endpoint)
        model_path = output / "models" / f"{path.stem}_fixed.pt"
        audit.check(model_path.is_file(), "DIP fixed model exists", path.name)
        history = payload["history"]
        expected_steps = sorted({1, config["iterations"], *range(config["sample_every"], config["iterations"] + 1,
                                                                 config["sample_every"])})
        audit.check([p["iteration"] for p in history] == expected_steps, "DIP sampled history coverage " + path.stem)
        fixed = next(r for r in payload["metrics"] if r["checkpoint"] == "fixed")
        oracle = next(r for r in payload["metrics"] if r["checkpoint"] == "oracle")
        best = min(history, key=lambda p: p["nrmse"])
        audit.check(oracle["iteration"] == best["iteration"], "DIP oracle first minimum step " + path.stem)
        audit.check(fixed["iteration"] == config["iterations"] and not fixed["truth_used_for_checkpoint"],
                    "DIP fixed budget/truth flag " + path.stem)
        audit.check(bool(oracle["truth_used_for_checkpoint"]), "DIP oracle truth flag " + path.stem)
        for metric in METRICS:
            audit.close(oracle[metric], best[metric], "DIP oracle sampled " + metric + path.stem)
            audit.close(fixed[metric], history[-1][metric], "DIP final history " + metric + path.stem)
    expected_endpoints = {(*fit, checkpoint) for fit in expected_fits for checkpoint in ("fixed", "oracle")}
    audit.counts["dip"] = {"fits": len(payloads), "expected_fits": 48,
                           "endpoints": len(rows), "expected_endpoints": 96,
                           "aggregate_rows": len(read_csv(output / "aggregate_metrics.csv")), "expected_aggregates": 32}
    audit.coverage(set(payloads), expected_fits, "DIP fits")
    audit.coverage(set(rows), expected_endpoints, "DIP endpoints")
    for name, angles, snr, seed in itertools.product(DATASETS, (30, 15), (20, 30), (0, 1, 2)):
        pair = [rows.get((name, angles, snr, seed, method, "fixed")) for method in ("DIP", "DIP_TV")]
        if all(pair):
            audit.check(pair[0]["initialization_sha256"] == pair[1]["initialization_sha256"],
                        "DIP paired initialization " + str((name, angles, snr, seed)))
    if structure_only:
        return rows
    csv_fields = (*fit_fields, "checkpoint")
    individual_csv = read_csv(output / "individual_metrics.csv")
    individual = {key_of(r, csv_fields): r for r in individual_csv}
    audit.check(len(individual_csv) == len(individual), "DIP individual CSV has no duplicates")
    audit.coverage(set(individual), expected_endpoints, "DIP individual CSV")
    for key, row in rows.items():
        A, b, truth, row_ids = data.case(*key[:3])
        vector = np.load(ROOT / row["reconstruction_path"])
        audit.check(vector.shape == (A.shape[1],) and bool(np.isfinite(vector).all()) and bool((vector >= 0).all()),
                    "DIP finite nonnegative vector with expected shape " + str(key))
        expected_scale = np.linalg.norm(b) / np.linalg.norm(A @ np.ones(A.shape[1]))
        audit.close(row["observation_scale"], expected_scale, "DIP observation-derived scale " + str(key))
        audit.check(row["row_ids_sha256"] == hashlib.sha256(row_ids.tobytes()).hexdigest(),
                    "DIP independent row hash " + str(key))
        recomputed = independent_metrics(vector, truth, A, b)
        for metric in METRICS:
            audit.close(row[metric], recomputed[metric], "DIP recomputed " + metric + str(key))
            if key in individual:
                audit.close(individual[key][metric], row[metric], "DIP JSON/CSV " + metric + str(key))
    aggregate_fields = ("dataset", "angle_count", "snr_label_db", "method", "checkpoint")
    groups = defaultdict(list)
    for row in rows.values():
        groups[key_of(row, aggregate_fields)].append(row)
    aggregate_csv = read_csv(output / "aggregate_metrics.csv")
    aggregate = {key_of(r, aggregate_fields): r for r in aggregate_csv}
    audit.check(len(aggregate_csv) == len(aggregate), "DIP aggregate CSV has no duplicates")
    expected_groups = set(itertools.product(DATASETS, (30, 15), (20, 30), ("DIP", "DIP_TV"), ("fixed", "oracle")))
    audit.coverage(set(aggregate), expected_groups, "DIP aggregate CSV")
    for key, group in groups.items():
        audit.check(sorted(r["seed"] for r in group) == [0, 1, 2], "DIP aggregate three seeds " + str(key))
        if key not in aggregate:
            continue
        summary = aggregate[key]
        audit.check(int(summary["n_seeds"]) == len(group), "DIP aggregate count " + str(key))
        audit.check(sorted(int(s) for s in summary["seeds"].split(",")) == sorted(r["seed"] for r in group),
                    "DIP aggregate declared seeds " + str(key))
        for metric in (*METRICS, "runtime_seconds", "iteration"):
            values = [r[metric] for r in group]
            audit.close(summary[metric + "_mean"], np.mean(values), "DIP mean " + metric + str(key))
            if len(group) > 1:
                audit.close(summary[metric + "_std"], np.std(values, ddof=1),
                            "DIP sample std(ddof=1) " + metric + str(key))
        boundary_count = sum(r["iteration"] == r["budget_iterations"] for r in group) if key[-1] == "oracle" else 0
        audit.check(int(summary["oracle_at_budget_count"]) == boundary_count, "DIP oracle budget-boundary count " + str(key))
    return rows


# 每个数据集恢复一个指定 fixed 模型，用保存的输入重新前向并对比保存数组。
def restore_models(audit, output, rows, device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in DATASETS:
        key = (name, 30, 30, 0, "DIP", "fixed")
        if key not in rows:
            audit.incomplete.append({"check": "model restoration target absent", "missing": key})
            if not audit.allow_partial:
                audit.check(False, "required model restoration target " + name)
            continue
        row = rows[key]
        model_path = output / "models" / f"{name}_30angles_30db_DIP_seed0_fixed.pt"
        # 恢复本项目保存的权重及输入信息，不执行新的训练。
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        net = PriorNet().to(device)
        net.load_state_dict(checkpoint["state_dict"])
        audit.check(checkpoint["architecture"] == "PriorNet-v2", "restored model architecture " + name)
        audit.check(checkpoint["iteration"] == row["iteration"], "restored model iteration " + name)
        audit.close(checkpoint["scale"], row["observation_scale"], "restored model scale " + name)
        bn_layers = [layer for layer in net.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
        audit.check(bool(bn_layers) and all(not layer.track_running_stats for layer in bn_layers),
                    "BatchNorm has no running statistics " + name)
        noise = checkpoint["noise"].to(device)
        with torch.no_grad():
            net.train()
            train_output = image_vector(net(noise) * checkpoint["scale"]).ravel().cpu().numpy()
            net.eval()
            eval_output = image_vector(net(noise) * checkpoint["scale"]).ravel().cpu().numpy()
        # 直接比较重新前向与保存数组，不只检查模型文件是否能打开。
        saved = np.load(ROOT / row["reconstruction_path"])
        max_absolute_error = float(np.max(np.abs(eval_output - saved)))
        relative_error = float(np.linalg.norm(eval_output - saved) / np.linalg.norm(saved))
        audit.check(bool(np.allclose(train_output, eval_output, rtol=1e-6, atol=checkpoint["scale"] * 1e-7)),
                    "BatchNorm train/eval outputs agree " + name)
        audit.check(relative_error < 2e-5 and max_absolute_error < checkpoint["scale"] * 5e-5,
                    "restored fixed model reproduces saved vector " + name,
                    {"relative_error": relative_error, "max_absolute_error": max_absolute_error})
        audit.model_restorations.append({"dataset": name, "device": device,
                                        "relative_error": relative_error, "max_absolute_error": max_absolute_error,
                                        "model": str(model_path.relative_to(ROOT))})
        del net, checkpoint


# 命令行入口：解析参数、检查配置，再调用本文件的实验／核验流程。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-dir", type=Path, default=ROOT / "outputs" / "extension" / "classical")
    parser.add_argument("--dip-dir", type=Path, default=ROOT / "outputs" / "extension" / "dip")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "extension" / "audit_summary.json")
    parser.add_argument("--model-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    audit, data = Audit(args.allow_partial), DataAccess()
    try:
        audit_classical(audit, args.classical_dir.resolve(), data, args.structure_only)
        rows = audit_dip(audit, args.dip_dir.resolve(), data, args.structure_only)
        if not args.structure_only:
            restore_models(audit, args.dip_dir.resolve(), rows, args.model_device)
    except Exception as exc:
        audit.errors.append({"check": "audit execution", "detail": f"{type(exc).__name__}: {exc}"})
    status = "failed" if audit.errors else "incomplete" if audit.incomplete else "passed"
    summary = {"status": status, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "structure_only": args.structure_only, "allow_partial": args.allow_partial,
               "checked_assertions": audit.checked, "counts": audit.counts, "errors": audit.errors,
               "incomplete": audit.incomplete, "model_restorations": audit.model_restorations,
               "elapsed_seconds": time.perf_counter() - started,
               "scope": "artifact coverage, metrics recomputation, oracle sampled minima, paired seeds, ddof=1 aggregation, fixed model restoration"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps({"status": status, "counts": audit.counts, "errors": audit.errors,
                      "checked_assertions": audit.checked, "summary": str(args.summary.resolve())}, ensure_ascii=False))
    raise SystemExit(1 if audit.errors else 0)


if __name__ == "__main__":
    main()
