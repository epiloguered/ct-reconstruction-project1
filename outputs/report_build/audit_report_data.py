"""Read existing CT outputs and export report data; never calls a reconstruction solver."""
from pathlib import Path
from collections import defaultdict
from itertools import product
import csv
import json
import math
import os
import hashlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
from scipy import sparse
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/report_build"
DATASETS = ("SheppLogan64", "XCAT128")
SNRS = (15, 20, 25, 30)

def typed(v):
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
        return int(f) if math.isfinite(f) and f.is_integer() else f
    except ValueError:
        return v

def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [{k: typed(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    return [{**r, "source_csv": path} for r in rows]

def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))

def path_of(row):
    if row.get("reconstruction_path"):
        return row["reconstruction_path"]
    if row.get("array"):
        return row["array"]
    if row.get("selected_array"):
        return str(Path(row["source_csv"]).parent / row["selected_array"])
    if row["source_csv"].startswith("outputs/regularized"):
        return f"outputs/regularized/{row['dataset']}/reconstructions/{row['dataset']}_{row['snr_label_db']}db_{row['method']}.npy"
    return f"outputs/{row['dataset']}/reconstructions/{row['dataset']}_{row['snr_label_db']}db_{row['method']}.npy"

def enhance(rows, stage, angle=None, protocol=None):
    ans = []
    for r in rows:
        a = dict(r)
        a["stage"] = stage
        if angle is not None:
            a["angle_count"] = angle
        if protocol:
            a["protocol"] = protocol
        if "nrmse" in r:
            a["array_path"] = path_of(r).replace("\\", "/")
            a["array_absolute_path"] = str(ROOT / a["array_path"])
        ans.append(a)
    return ans

baseline = enhance(sum([read_csv(f"outputs/{d}/metrics.csv") for d in DATASETS], []), "original_90angle_baseline", 90,
    "LSQR atol=btol=1e-5, limit 10000; ART 40-epoch trajectory with fixed5/final40/oracle-best views")
regularized = enhance(sum([read_csv(f"outputs/regularized/{d}/selected_metrics.csv") for d in DATASETS], []), "original_90angle_regularization", 90,
    "oracle GT NRMSE parameter selection; LS-L1 1200, LS-TV600; ART priors fixed5")
reg_candidates = sum([read_csv(f"outputs/regularized/{d}/candidate_metrics.csv") for d in DATASETS], [])
classical = enhance(read_csv("outputs/extension/classical/selected_metrics.csv"), "sparse_classical_main",
    protocol="90/45/30/15 angles x all4SNR; oracle regularization parameters; LS-TV600, ART5")
classical_candidates = read_csv("outputs/extension/classical/candidate_metrics.csv")
dip_individual = enhance(read_csv("outputs/extension/dip/individual_metrics.csv"), "DIP_original_individual")
dip_aggregate = enhance(read_csv("outputs/extension/dip/aggregate_metrics.csv"), "DIP_original_aggregate",
    protocol="3 seeds, fixed2000 main; oracle 41 sampled checkpoint diagnostic")
refined = enhance(read_csv("outputs/improvements/summary/classical_final_metrics.csv"), "refined_classical_final",
    protocol="GT selects parameter after6000 then continues same state6000; report12000")
refined_selected6000 = enhance(read_csv("outputs/improvements_refined/classical/selected_metrics.csv"), "refined_classical_selection6000")
neural_individual = enhance(read_csv("outputs/improvements/neural/individual_metrics.csv"), "DIP_TV_heldout_individual")
neural_aggregate = enhance(read_csv("outputs/improvements/neural/aggregate_metrics.csv"), "DIP_TV_heldout_aggregate",
    protocol="DIP+TV 2000 iterations, 20pct observed angle blocks held out, validation selects checkpoint withoutGT; no full-data refit")
latest_final = enhance(read_csv("outputs/min_angles/summary/final_metrics.csv"), "minimum_angles_final",
    protocol="oracle parameters at6000 TV/wave or40 PnP; final12000 TV/wave or160 PnP; PnP starts fromTV12000")
latest_best = enhance(read_csv("outputs/min_angles/summary/best_available_metrics.csv"), "minimum_angles_best_endpoint",
    protocol="extra GT choice between grid-selected and continued endpoints; not deployable unknownGT stopping")
fullview = enhance(read_csv("outputs/min_angles/fullview_audit/comparison.csv"), "original90_vs_oracle_best10")

checks = []
def check(condition, text):
    if not condition:
        raise AssertionError(text)
    checks.append(text)

def check_grid(rows, expected, fields, label):
    expected = list(expected)
    actual = [tuple(r[f] for f in fields) for r in rows]
    check(len(actual) == len(set(actual)), f"{label}: no duplicate condition/method rows")
    check(set(actual) == set(expected), f"{label}: complete intended grid ({len(expected)} rows)")

check_grid(baseline, product(DATASETS,SNRS,("LSQR","ART_fixed","ART_final","ART_best")), ("dataset","snr_label_db","method"), "original baseline")
check_grid(regularized, product(DATASETS,SNRS,("LS_L2","LS_L1","LS_TV","ART_L2","ART_L1","ART_TV")), ("dataset","snr_label_db","method"), "original regularization")
check_grid(classical, product(DATASETS,(90,45,30,15),SNRS,("LSQR","ART_fixed","LS_L2","LS_TV","ART_TV")), ("dataset","angle_count","snr_label_db","method"), "sparse main")
check_grid(dip_aggregate, product(DATASETS,(30,15),(20,30),("DIP","DIP_TV"),("fixed","oracle")), ("dataset","angle_count","snr_label_db","method","checkpoint"), "DIP aggregates")
check(len(dip_individual) == 96, "DIP individual: 48fits, 96endpoints")
check(len(reg_candidates) == 352, "original regularization: 352candidate rows")
check(len(classical_candidates) == 864, "sparse main: 864candidate rows")
check(len(refined) == len(refined_selected6000) == 28, "refined classical: 28selected6000 and28final12000 rows")
check(len(neural_individual) == 48 and len(neural_aggregate) == 16, "heldout DIP+TV: 24fits48endpoints16aggregates")
check(len(latest_final) == len(latest_best) == 76, "latest methods: 76fixed-budget final and76best-endpoint rows")

for selected,candidates,fields,label in ((regularized,reg_candidates,("dataset","snr_label_db","method"),"original regularization"),(classical,classical_candidates,("dataset","angle_count","snr_label_db","method"),"sparse main")):
    grouped = defaultdict(list)
    for r in candidates:
        grouped[tuple(r[k] for k in fields)].append(r)
    for r in selected:
        minimum = min(v["nrmse"] for v in grouped[tuple(r[k] for k in fields)])
        check(np.isclose(r["nrmse"], minimum, rtol=1e-12, atol=1e-12), f"{label} oracle-min {tuple(r[k] for k in fields)}")

for individuals,aggregates,keys in ((dip_individual,dip_aggregate,("dataset","angle_count","snr_label_db","method","checkpoint")),(neural_individual,neural_aggregate,("dataset","angle_count","snr_label_db","checkpoint"))):
    grouped = defaultdict(list)
    for r in individuals:
        grouped[tuple(r[k] for k in keys)].append(r)
    for agg in aggregates:
        members = grouped[tuple(agg[k] for k in keys)]
        check(sorted(r["seed"] for r in members) == [0,1,2], f"3seeds {tuple(agg[k] for k in keys)}")
        for metric in ("nrmse","psnr_db","ssim","relative_residual"):
            values = [r[metric] for r in members]
            check(np.isclose(np.mean(values), agg[metric+"_mean"], rtol=1e-12, atol=1e-12), "aggregate mean "+metric)
            check(np.isclose(np.std(values,ddof=1), agg[metric+"_std"], rtol=1e-12, atol=1e-12), "aggregate sample std "+metric)

# Independent image/residual recomputation from saved selected vectors only.
# These operations evaluate old outputs and never run optimization/training.
truths,matrices,observations,inputs = {},{},{},{}
for dataset in DATASETS:
    truth = np.loadtxt(ROOT/f"Data/{dataset}/gt.csv", delimiter=",")
    matrix = sparse.load_npz(ROOT/f"outputs/cache/{dataset}_A_csr.npz")
    truths[dataset],matrices[dataset] = truth,matrix
    observations[dataset] = {s:np.loadtxt(ROOT/f"Data/{dataset}/b_hat_{s}db.csv",delimiter=",").reshape(-1) for s in SNRS}
    inputs[dataset] = {"truth":f"Data/{dataset}/gt.csv", "projection_csv":f"Data/{dataset}/A.csv", "sparse_cache":f"outputs/cache/{dataset}_A_csr.npz", "observations":{str(s):f"Data/{dataset}/b_hat_{s}db.csv" for s in SNRS}, "truth_shape":list(truth.shape), "matrix_shape":list(matrix.shape), "nnz":int(matrix.nnz), "zero_rows":int(np.sum(np.diff(matrix.indptr)==0)), "truth_min":float(truth.min()), "truth_max":float(truth.max())}

subsets = {}
def subset(dataset,angles,snr):
    key = dataset,angles,snr
    if key not in subsets:
        matrix = matrices[dataset]
        detector_count = matrix.shape[0]//90
        angle_ids = np.floor(np.arange(angles)*90/angles).astype(int)
        ids=(angle_ids[:,None]*detector_count+np.arange(detector_count)[None,:]).ravel()
        subsets[key] = matrix[ids], observations[dataset][snr][ids]
    return subsets[key]

verified_arrays = set()
max_abs_metric_diffs = {k:0. for k in ("nrmse","psnr_db","ssim","relative_residual")}
for rows in (baseline,regularized,classical,dip_individual,refined,refined_selected6000,neural_individual,latest_final,latest_best,fullview):
    for r in rows:
        path = r["array_path"]
        if path in verified_arrays:
            continue
        x=np.load(ROOT/path).reshape(-1)
        truth=truths[r["dataset"]]
        estimate=x.reshape(truth.shape,order="F")
        diff=estimate-truth
        mse=float(np.mean(diff**2))
        angles=r.get("angle_count",90)
        snr=r.get("snr_label_db",r.get("snr"))
        matrix,b=subset(r["dataset"],angles,snr)
        recomputed={"nrmse":float(np.linalg.norm(diff)/np.linalg.norm(truth)),"psnr_db":float(10*np.log10((truth.max()-truth.min())**2/mse)),"ssim":float(structural_similarity(truth,estimate,data_range=float(truth.max()-truth.min()))),"relative_residual":float(np.linalg.norm(matrix@x-b)/np.linalg.norm(b))}
        check(np.isfinite(x).all(),f"finite array {path}")
        for metric,value in recomputed.items():
            max_abs_metric_diffs[metric]=max(max_abs_metric_diffs[metric],abs(value-r[metric]))
            check(np.isclose(value,r[metric],rtol=1e-9,atol=1e-10),f"saved-array metric {metric} {path}")
        verified_arrays.add(path)

audits = {name:read_json(path) for name,path in {"extension":"outputs/extension/audit_summary.json","improvements":"outputs/improvements/audit_summary.json","min_angles":"outputs/min_angles/audit_summary.json","original90_comparison":"outputs/min_angles/fullview_audit/audit.json"}.items()}
metadata = {
"project_absolute_path":str(ROOT),
"audit_date":"2026-10-11",
"audit_scope":"Read existing docs/CSV/JSON and independently recompute selected saved vectors; no new optimization, training, test run, or replay",
"inputs":inputs,
"metric_conventions":{"vectorization":"column-major F","nrmse":"L2(image-truth)/L2(truth)","psnr":"10log10(GT_range^2/MSE)","ssim":"skimage structural_similarity withGT_range, default7x7window","relative_residual":"L2(A_subset*x-b_subset)/L2(b_subset), includes zero rows","clipping":"No clipping before metrics; plots alone useGT intensity range"},
"coverage":{"original_baseline_rows":len(baseline),"original_primary_LSQR_ART_fixed_rows":16,"original_regularization_selected":len(regularized),"original_regularization_candidates":len(reg_candidates),"sparse_classical_selected":len(classical),"sparse_classical_candidates":len(classical_candidates),"DIP_fits":48,"DIP_endpoints":len(dip_individual),"DIP_aggregate_rows":len(dip_aggregate),"refined_classical_final":len(refined),"heldout_DIP_fits":24,"latest_final":len(latest_final)},
"current_readonly_validation":{"status":"passed","assertions":len(checks),"unique_saved_arrays_recomputed":len(verified_arrays),"maximum_absolute_metric_difference":max_abs_metric_diffs,"scope":"Coverage, per-method oracle grid minima, 3seed mean/sampleStd, finite saved selected arrays and four metrics; no solver run"},
"tables":{"baseline_all":baseline,"baseline_primary":[r for r in baseline if r["method"] in ("LSQR","ART_fixed")],"original_regularization":regularized,"sparse_classical_main_all4SNR":classical,"DIP_fixed":[r for r in dip_aggregate if r["checkpoint"]=="fixed"],"DIP_oracle":[r for r in dip_aggregate if r["checkpoint"]=="oracle"],"DIP_individual":dip_individual,"refined_classical_final12000":refined,"refined_classical_selected6000":refined_selected6000,"heldout_DIP_aggregate":neural_aggregate,"latest_fixed_budget_final":latest_final,"latest_oracle_best_endpoint":latest_best,"original90_vs_best10":fullview},
"historical_saved_audits":audits,
"selection_warnings":["Original90 regularization, sparse classical and latest traditional parameters useGT NRMSE andGT range calibration.","DIP main fixed2000 uses noGT inloss, amplitude scaling or stopping; oracle columns are41sampled checkpoints only.","Original90 output, sparse extension90 rerun and later12000step TV_control are different stages; do not silently merge.","Latest final is12000TV/wave or160PnP at parameters selected earlier; best_available addsGT endpoint selection.","PnP cost includes nonnegativeTV12000 initialization,9candidate search andCG+BM3D steps.","Latest 4threadBM3D replay failed strict1e-6 tolerance; saved artifacts verified, not strict numerical replay.","All3-30angle saved outputs fail combinedNRMSE/SSIM threshold oforiginal90LS-TV forbothdatasets20/30dB.","Historical15angle threshold is notoriginal90quality; minimum tested angles is notglobal minimum.","Two truth images and one supplied noisy observation perSNR do not support broad generalization orclinical/dose claims."],
"main_table_recommendation":["Table1: original90 LSQR/ART_fixed across all4SNR; report NRMSE/PSNR/SSIM, LSQR stop code and iterations supplementary.","Table2: original90 six regularizers plus matching originalLSQR/ART_fixed,8data/SNRrows; mainNRMSE,PSNR/SSIM inappendix orparallelcompacttables.","Table3: sparse main all32conditions across90/45/30/15angles andall4SNR,5methods; source onlyextension/classical tokeep90rerun consistent.","Table4: DIP/DIP-TV fixed2000 mean±sampleStd for8conditions plussamecondition classicalTV; oracle separate diagnostic appendix.","Table5: later controlled TV/NN-TV/TGV atfixed12000 andheldoutDIP checkpoint comparison keptseparate fromoriginalbaselines.","Table6: latestTV/wave/PnP fixedbudget table covers20/30dB; commonLCgrid main, standardBM3Dprofile check separately.","Table7: original90LS-TV/ART-TV versusbest10 withendpoint labels andstrictcombinedthreshold table; stateoriginal90LS-TVquality notachieved."]
}
(OUT/"report_data.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"data_audit_checks.json").write_text(json.dumps({"checks":checks,**metadata["current_readonly_validation"]},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metadata["current_readonly_validation"],ensure_ascii=False))
print(str(OUT/"report_data.json"))
