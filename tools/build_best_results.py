"""从已有实验汇总生成各方法最佳已测条件主表，不重新运行重建实验。"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "LSQR": "LSQR", "ART_fixed": "ART（固定5轮）",
    "LS_L2": "LS+L2", "LS_L1": "LS+L1", "LS_TV": "LS+TV",
    "ART_L2": "ART+L2", "ART_L1": "ART+L1", "ART_TV": "ART+TV",
    "DIP": "DIP", "DIP_TV": "DIP+TV", "tv": "非负TV",
    "TGV_nonnegative": "TGV+非负", "wave": "小波+TV+非负",
    "pnp": "PnP-BM3D（LC配置）", "pnp_np": "PnP-BM3D（常规配置）",
}
BASIC_ORDER = ["LSQR", "ART_fixed", "LS_L2", "LS_L1", "LS_TV", "ART_L2", "ART_L1", "ART_TV"]
EXTENSION_ORDER = ["LSQR", "ART_fixed", "LS_L2", "LS_TV", "ART_TV", "DIP", "DIP_TV", "tv", "TGV_nonnegative", "wave", "pnp", "pnp_np"]
SOURCES = [
    ("basic", "baseline_regularization.csv", BASIC_ORDER),
    ("extension", "sparse_classical.csv", EXTENSION_ORDER[:5]),
    ("extension", "dip_aggregate.csv", ["DIP", "DIP_TV"]),
    ("extension", "non_dip_fixed.csv", ["tv", "wave", "pnp", "pnp_np"]),
    ("extension", "tv_tgv_controls.csv", ["TGV_nonnegative"]),
]
FIELDS = [
    "stage", "dataset", "method", "method_label", "snr", "angle_count",
    "nrmse", "psnr_db", "ssim", "nrmse_std", "psnr_db_std", "ssim_std",
    "iterations", "endpoint", "budget", "parameter_config", "source_csv",
    "source_export", "source_data_row", "array", "eligible_angles", "eligible_snrs",
]


def get_budget(method: str, iterations: str) -> str:
    if method in ("DIP", "DIP_TV"):
        return "固定2000步；3种子均值±样本标准差"
    if method in ("pnp", "pnp_np"):
        return "TV起点12000步＋PnP固定160步"
    if method.startswith("ART"):
        return "固定5轮"
    if method in ("tv", "wave", "TGV_nonnegative"):
        return "固定12000步"
    if method in ("LSQR", "LS_L2"):
        return f"按数值停止条件；{iterations}次迭代"
    return f"固定{iterations}步"


def main() -> None:
    best = []
    provenance = []
    hashes = {}
    for stage, filename, methods in SOURCES:
        path = ROOT / "outputs" / "final" / filename
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        groups = defaultdict(list)
        for row_number, row in enumerate(rows, 1):
            if row["method"] not in methods:
                continue
            if stage == "extension" and int(row["angle_count"]) >= 90:
                continue
            # DIP只比较固定终点的三种子统计，不把GT最佳截点混入主表。
            if filename == "dip_aggregate.csv" and row["checkpoint"] != "fixed":
                continue
            groups[(row["dataset"], row["method"])].append((row_number, row))
        for (dataset, method), eligible in groups.items():
            is_dip = filename == "dip_aggregate.csv"
            metric_key = "nrmse_mean" if is_dip else "nrmse"
            row_number, row = min(eligible, key=lambda pair: float(pair[1][metric_key]))
            iterations = row.get("total_iterations") or row.get("iterations") or row["iteration_mean"]
            config = {
                key: row[key]
                for key in ("parameter_name", "parameter_value", "objective_lambda", "fraction",
                            "tgv_ratio", "lambda_raw", "p", "ratio", "rho", "sigma", "stop_code", "seeds")
                if row.get(key, "") != ""
            }
            if method in ("pnp", "pnp_np"):
                config.update(rho=row["p"], sigma=row["ratio"], bm3d_profile="np" if method == "pnp_np" else "lc")
            elif method in ("tv", "wave"):
                config.update(tv_weight=row["p"], wavelet_to_tv_ratio=row["ratio"])
            record = {
                "stage": stage, "dataset": dataset, "method": method, "method_label": LABELS[method],
                "snr": int(row["snr_label_db"]), "angle_count": int(row["angle_count"]),
                "nrmse": float(row[metric_key]),
                # 三指标必须来自同一条选中记录，不能分别取各列最优值。
                "psnr_db": float(row["psnr_db_mean"] if is_dip else row["psnr_db"]),
                "ssim": float(row["ssim_mean"] if is_dip else row["ssim"]),
                "nrmse_std": float(row["nrmse_std"]) if is_dip else "",
                "psnr_db_std": float(row["psnr_db_std"]) if is_dip else "",
                "ssim_std": float(row["ssim_std"]) if is_dip else "",
                "iterations": float(iterations),
                "endpoint": "fixed 2000-step three-seed mean" if is_dip else "prescribed endpoint / numerical stopping",
                "budget": get_budget(method, iterations),
                "parameter_config": json.dumps(config, ensure_ascii=False, sort_keys=True),
                "source_csv": row["source_csv"], "source_export": path.relative_to(ROOT).as_posix(),
                "source_data_row": row_number, "array": row.get("array") or row.get("selected_array", ""),
                "eligible_angles": ",".join(map(str, sorted({int(r["angle_count"]) for _, r in eligible}))),
                "eligible_snrs": ",".join(map(str, sorted({int(r["snr_label_db"]) for _, r in eligible}))),
            }
            best.append(record)
            provenance.append({"selected": record, "original_row": row, "eligible_condition_count": len(eligible)})
    best.sort(key=lambda r: (
        ("basic", "extension").index(r["stage"]),
        ("SheppLogan64", "XCAT128").index(r["dataset"]),
        (BASIC_ORDER if r["stage"] == "basic" else EXTENSION_ORDER).index(r["method"]),
    ))
    assert len(best) == 40
    assert sum(r["stage"] == "basic" for r in best) == 16
    assert all(r["angle_count"] == 90 for r in best if r["stage"] == "basic")
    assert all(r["angle_count"] < 90 for r in best if r["stage"] == "extension")
    assert all(r["iterations"] == 2000 for r in best if r["method"] in ("DIP", "DIP_TV"))
    output = ROOT / "outputs" / "final" / "best_by_method.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(best)
    audit_path = ROOT / "outputs" / "report_build" / "best_results_provenance.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "selection": "Per dataset and method, minimum NRMSE across tested fixed endpoint conditions; extension excludes 90 angles; DIP uses fixed 2000-step three-seed means.",
        "input_sha256": hashes, "selected_rows": provenance,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(best)} rows: {output}")


if __name__ == "__main__":
    main()
