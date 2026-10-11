"""独立复核新实验，生成可直接插入结果文档的表格与对照图。"""
import os
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
import csv,json,itertools
from pathlib import Path
import numpy as np
from run_baselines import ROOT,DATASETS,load_matrix,save_csv
from run_sparse_views import select_angles,read_json,write_json
from run_sparse_improvements import OUT,FRACTIONS,split_observed
from verify_extension import independent_metrics

METRICS=("nrmse","psnr_db","ssim","relative_residual")
def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def main():
    errors=[];checked=0;unique_arrays=set();model_checks=[]
    def check(condition,label):
        nonlocal checked
        checked+=1
        if not condition:errors.append(label)
    matrices={name:load_matrix(name,ROOT/"outputs/cache") for name in DATASETS}
    truths={name:np.loadtxt(ROOT/"Data"/name/"gt.csv",delimiter=",") for name in DATASETS}
    cache={}
    def case(row):
        key=(row["dataset"],int(row["angle_count"]),int(row["snr_label_db"]))
        if key not in cache:
            b=np.loadtxt(ROOT/"Data"/key[0]/f"b_hat_{key[2]}db.csv",delimiter=",").ravel()
            A,b,_,_=select_angles(matrices[key[0]],b,key[1]);cache[key]=(A,b,truths[key[0]])
        return cache[key]
    def metrics(row,path):
        A,b,truth=case(row);x=np.load(ROOT/path);unique_arrays.add(path)
        check(x.shape==(A.shape[1],) and np.isfinite(x).all(),"array shape/finite "+path)
        for metric,value in independent_metrics(x,truth,A,b).items():
            check(np.isclose(value,float(row[metric]),rtol=1e-9,atol=1e-10),"metric "+metric+" "+path)
        return x
    classical_dir=ROOT/"outputs/improvements_refined/classical"
    classical={};candidate_count=0
    files=sorted(classical_dir.glob("*.json"));check(len(files)==28,"28 classical cases")
    for path in files:
        payload=read_json(path);selected=payload["selected"];refined=payload["refinement"]
        key=(selected["dataset"],selected["angle_count"],selected["snr_label_db"],selected["method"])
        check(key not in classical,"classical uniqueness");classical[key]=payload
        candidates=payload["candidates"];expected=len(FRACTIONS)*(3 if selected["method"]=="TGV_nonnegative" else 1)
        check(len(candidates)==expected,"candidate count "+str(key));candidate_count+=len(candidates)
        actual_grid={(r["fraction"],r["tgv_ratio"]) for r in candidates}
        expected_grid=set(itertools.product(FRACTIONS,(.5,1.,2.) if selected["method"]=="TGV_nonnegative" else (0.,)))
        check(actual_grid==expected_grid,"parameter grid "+str(key))
        for row in candidates:
            x=metrics(row,row["array"])
            if row["nonnegative"]:check(x.min()>=0,"nonnegative "+row["array"])
        check(selected["nrmse"]==min(r["nrmse"] for r in candidates),"oracle selection "+str(key))
        check(any(r==selected for r in candidates),"selected record belongs to candidates")
        final={**selected,**refined};x=metrics(final,refined["array"])
        if selected["nonnegative"]:check(x.min()>=0,"refined nonnegative")
        check(refined["total_iterations"]==12000,"refinement budget")
    check(candidate_count==396,"396 candidates")
    expected_cases={(name,a,s,m) for name in DATASETS for a in (90,30,15) for s in (20,30)
                    for m in (("TV_control",) if a==90 else ("TV_control","TV_nonnegative","TGV_nonnegative"))}
    check(set(classical)==expected_cases,"complete classical condition grid")
    neural=[];neural_payloads=[]
    files=sorted((OUT/"neural").glob("*.json"));check(len(files)==24,"24 heldout fits")
    for path in files:
        payload=read_json(path);entries=payload["metrics"];neural_payloads.append(payload)
        check(len(payload["history"])==41,"41 heldout samples")
        validation=next(r for r in entries if r["checkpoint"]=="validation")
        A,b,truth=case(validation);train,val=split_observed(A,b,validation["angle_count"])
        check(payload["train_rows"]==train.tolist() and payload["validation_rows"]==val.tolist(),"saved split")
        check(np.intersect1d(train,val).size==0 and len(train)+len(val)==len(b),"disjoint complete split")
        best=min(payload["history"],key=lambda r:r["validation_relative_mse"])
        check(validation["iteration"]==best["iteration"],"validation selected step")
        check(payload["GT_used_for_training_or_checkpoint"] is False,"no GT checkpoint declaration")
        old_key=f"{validation['dataset']}_{validation['angle_count']}angles_{validation['snr_label_db']}db_DIP_TV_seed{validation['seed']}"
        old=read_json(ROOT/"outputs/extension/dip/records"/(old_key+".json"))
        check(payload["initialization_sha256"]==old["metrics"][0]["initialization_sha256"],"paired previous full-data initialization")
        for row in entries:
            x=metrics(row,row["array"]);neural.append(row)
            residual=A[val]@x-b[val];value=float(residual@residual/(b[val]@b[val]))
            point=best if row["checkpoint"]=="validation" else payload["history"][-1]
            check(np.isclose(value,point["validation_relative_mse"],rtol=3e-5,atol=1e-7),"independent heldout MSE "+row["array"])
        check((path.with_name(path.stem+"_validation.pt")).is_file(),"validation model exists")
    aggregate=csv_rows(OUT/"neural/aggregate_metrics.csv");check(len(aggregate)==16,"16 neural aggregates")
    expected_endpoints=set(itertools.product(DATASETS,(30,15),(20,30),(0,1,2),("validation","heldout_fixed")))
    actual_endpoints={(r["dataset"],r["angle_count"],r["snr_label_db"],r["seed"],r["checkpoint"]) for r in neural}
    check(len(neural)==48 and actual_endpoints==expected_endpoints,"complete neural condition grid")
    lookup={}
    for row in aggregate:
        key=(row["dataset"],int(row["angle_count"]),int(row["snr_label_db"]),row["checkpoint"]);lookup[key]=row
        group=[r for r in neural if (r["dataset"],r["angle_count"],r["snr_label_db"],r["checkpoint"])==key]
        check(sorted(r["seed"] for r in group)==[0,1,2],"three seeds")
        for metric in (*METRICS,"iteration"):
            values=[r[metric] for r in group]
            for suffix,value in (("mean",np.mean(values)),("std",np.std(values,ddof=1))):
                check(np.isclose(value,float(row[metric+"_"+suffix]),rtol=1e-10,atol=1e-11),"aggregate "+str(key)+metric+suffix)
    # 每个数据集恢复一个验证选中的模型，核对保存的输入、尺度、截点和输出。
    import torch
    from run_dip_extension import PriorNet,image_vector
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    for name in DATASETS:
        row=next(r for r in neural if (r["dataset"],r["angle_count"],r["snr_label_db"],r["seed"],r["checkpoint"])==(name,30,30,0,"validation"))
        model=OUT/"neural"/f"{name}_30angles_30db_seed0_validation.pt"
        payload=torch.load(model,map_location="cuda",weights_only=True);net=PriorNet().cuda();net.load_state_dict(payload["state_dict"]);net.eval()
        with torch.no_grad():x=image_vector(net(payload["noise"].cuda())*payload["scale"]).ravel().cpu().numpy()
        saved=np.load(ROOT/row["array"]);rel=float(np.linalg.norm(x-saved)/np.linalg.norm(saved))
        check(rel<2e-5,"restored validation model "+name);check(payload["iteration"]==row["iteration"],"restored step")
        model_checks.append({"dataset":name,"relative_error":rel})
    source_snapshot=OUT/"source_executed_v1.py"
    import hashlib
    check(hashlib.sha256(source_snapshot.read_bytes()).hexdigest()==read_json(OUT/"neural_manifest.json")["source_sha256"],"neural executed-source fingerprint")
    check(hashlib.sha256((ROOT/"run_sparse_improvements.py").read_bytes()).hexdigest()==read_json(ROOT/"outputs/improvements_refined/classical_manifest.json")["source_sha256"],"classical source fingerprint")
    summary={"status":"failed" if errors else "passed","checked_assertions":checked,"unique_arrays_recomputed":len(unique_arrays),
             "classical_selected":len(classical),"classical_candidates":candidate_count,"neural_fits":len(neural_payloads),"neural_endpoints":len(neural),
             "errors":errors,"model_restorations":model_checks}
    write_json(OUT/"audit_summary.json",summary)
    if errors:raise AssertionError(json.dumps(summary))
    old_classical=csv_rows(ROOT/"outputs/extension/classical/selected_metrics.csv")
    old_classical={(r["dataset"],int(r["angle_count"]),int(r["snr_label_db"]),r["method"]):r for r in old_classical}
    old_neural=csv_rows(ROOT/"outputs/extension/dip/aggregate_metrics.csv")
    old_neural={(r["dataset"],int(r["angle_count"]),int(r["snr_label_db"]),r["method"]):r for r in old_neural if r["checkpoint"]=="fixed"}
    lines=["## 12. 少角度改进实验：非负 TV、TGV 与投影验证截点","",
        "本节为新增独立实验，不覆盖第 8–9 节历史结果。两数据集、30／15 角、20／30 dB；传统方法另跑 90 角 TV 参考。","",
        "### 12.1 协议与评价边界","",
        "- 传统方法统一使用原始-对偶算法。TV 对照不加非负约束；非负 TV 与非负二阶 TGV 都施加 x≥0。TGV 比较非负 TV，尽量分离非负约束与高阶先验的影响。",
        "- 公共强度网格 9 值；TGV 的 alpha0/alpha1 比例取 0.5、1、2。全部候选运行 6000 步，按 GT NRMSE 选参后统一续迭代 6000 步。主表报告 12000 步结果，未按该终点重新搜索网格；6000 步与续迭代敏感性另表报告。GT 也用于强度尺度校准，传统结果仍属于 oracle。",
        "- DIP+TV 固定原网络、学习率 0.001、2000 步、每 50 步记录、种子 0/1/2。每 5 个已采集角度固定留出第 3 个完整角度块作验证，训练只用其余 80%；按验证投影相对 MSE 最小选择截点，GT 不参与训练／截点选择。不在选择后用全部观测重新拟合。",
        "- heldout fixed 与 validation 来自同一次训练，用于隔离截点效果；原 full-data fixed 使用全部观测，是历史对照，不能把两者差异全部归因于早停。三种子均值±为样本标准差。","",
        "### 12.2 传统改进：NRMSE 与全角度差距","",
        "下表前三个新方法均为 12000 步。每个数据集／SNR 的90角参考使用同求解器的 TV 对照；差距定义为少角度 NRMSE 减去该90角参考 NRMSE，不是两张重建图的直接距离。","",
        "| 数据集 | 角度 | SNR | 90角TV参考 | 原LS+TV | 新TV对照 | 非负TV | 非负TGV | 非负TV差距 | 非负TGV差距 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    report_rows=[]
    for name in DATASETS:
        for angles in (30,15):
            for snr in (20,30):
                ref=classical[(name,90,snr,"TV_control")]["refinement"]["nrmse"]
                old=float(old_classical[(name,angles,snr,"LS_TV")]["nrmse"])
                values=[classical[(name,angles,snr,m)]["refinement"]["nrmse"] for m in ("TV_control","TV_nonnegative","TGV_nonnegative")]
                lines.append(f"| {name} | {angles} | {snr} | {ref:.4f} | {old:.4f} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[1]-ref:+.4f} | {values[2]-ref:+.4f} |")
                report_rows.append({"dataset":name,"angles":angles,"snr":snr,"reference":ref,"old":old,"control":values[0],"nn":values[1],"tgv":values[2]})
    lines += ["","### 12.3 传统改进：PSNR／SSIM","","| 数据集 | 角度 | SNR | TV PSNR | 非负TV PSNR | TGV PSNR | TV SSIM | 非负TV SSIM | TGV SSIM |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in report_rows:
        results=[classical[(r["dataset"],r["angles"],r["snr"],m)]["refinement"] for m in ("TV_control","TV_nonnegative","TGV_nonnegative")]
        fields=[f"{x['psnr_db']:.2f}" for x in results]+[f"{x['ssim']:.4f}" for x in results]
        lines.append(f"| {r['dataset']} | {r['angles']} | {r['snr']} | "+" | ".join(fields)+" |")
    lines += ["","### 12.4 DIP+TV：验证截点与固定终点","","| 数据集 | 角度 | SNR | 原全部观测fixed NRMSE | 留出fixed NRMSE | 验证截点NRMSE | 验证截点PSNR | 验证截点SSIM | 选中步数范围 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    def fmt(r,k):return f"{float(r[k+'_mean']):.4f} ± {float(r[k+'_std']):.4f}"
    for r in report_rows:
        key=(r["dataset"],r["angles"],r["snr"]);v=lookup[(*key,"validation")];f=lookup[(*key,"heldout_fixed")];old=old_neural[(*key,"DIP_TV")]
        steps=[x["iteration"] for x in neural if (x["dataset"],x["angle_count"],x["snr_label_db"],x["checkpoint"])==(*key,"validation")]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(old,'nrmse')} | {fmt(f,'nrmse')} | {fmt(v,'nrmse')} | {fmt(v,'psnr_db')} | {fmt(v,'ssim')} | {min(steps)}–{max(steps)} |")
    lines += ["","DIP+TV 的其余两项指标完整对照（均值 ± 样本标准差）：","",
        "| 数据集 | 角度 | SNR | 原全部观测PSNR | 留出fixed PSNR | 验证PSNR | 原全部观测SSIM | 留出fixed SSIM | 验证SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in report_rows:
        key=(r["dataset"],r["angles"],r["snr"])
        groups=[old_neural[(*key,"DIP_TV")],lookup[(*key,"heldout_fixed")],lookup[(*key,"validation")]]
        fields=[fmt(g,m) for m in ("psnr_db","ssim") for g in groups]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | "+" | ".join(fields)+" |")
    lines += ["","### 12.5 迭代预算敏感性与选中参数","","| 数据集 | 角度 | SNR | 方法 | 强度比例 | TGV比例 | 6000步NRMSE | 12000步NRMSE | 末250步目标相对变化 |","| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    sensitivities=[]
    for key,payload in sorted(classical.items()):
        s=payload["selected"];f=payload["refinement"];history=f["history"]
        change=abs(history[-1]["objective_scaled"]-history[-2]["objective_scaled"])/max(abs(history[-2]["objective_scaled"]),1e-15)
        sensitivities.append(change)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {s['fraction']:g} | {s['tgv_ratio']:g} | {s['nrmse']:.4f} | {f['nrmse']:.4f} | {change:.2e} |")
    win_nn=sum(r["nn"]<r["control"] for r in report_rows);win_tgv=sum(r["tgv"]<r["nn"] for r in report_rows)
    validation_better=0;validation_better_full=0
    for r in report_rows:
        key=(r["dataset"],r["angles"],r["snr"])
        value=float(lookup[(*key,"validation")]["nrmse_mean"])
        validation_better+=value<float(lookup[(*key,"heldout_fixed")]["nrmse_mean"])
        validation_better_full+=value<float(old_neural[(*key,"DIP_TV")]["nrmse_mean"])
    lines += ["","### 12.6 当前结论与核验","",
        f"- 非负 TV 在 {win_nn}/8 个少角度条件的 NRMSE 低于同求解器 TV 对照；非负 TGV 在 {win_tgv}/8 个条件低于非负 TV。差异很小时应结合参数和预算敏感性判断，不能仅按末位小数宣称稳定优势。",
        f"- TGV 相对非负 TV 的 NRMSE 变化范围为 {min(r['tgv']-r['nn'] for r in report_rows):+.6f} 至 {max(r['tgv']-r['nn'] for r in report_rows):+.6f}（负值表示改善）；不能把极小数值差别当作明确的高阶先验收益。",
        f"- 投影验证截点在 {validation_better}/8 个条件的平均 NRMSE 低于同次留出训练的固定终点；相对原使用全部观测的 fixed 结果，在 {validation_better_full}/8 个条件更低。验证投影误差不是图像误差的保证，留出观测也减少了训练信息。",
        f"- 选中参数续迭代末250步目标相对变化最大为 {max(sensitivities):.3e}；这不是严格收敛证书。TGV 仅搜索三种高阶比例，不能证明最优高阶模型。",
        f"- 独立核验 {summary['checked_assertions']} 个断言通过，逐一复算 {summary['unique_arrays_recomputed']} 个数组的四项指标，并恢复两个验证截点模型。记录见 [改进实验核验](../outputs/improvements/audit_summary.json)。",
        "- 本次不引入外部去噪模型或参考图像，没有把90角观测用于少角度训练。不能由本次结果声称少角度普遍等价于全角度，也不能报告实际辐射剂量降低比例。","",
        "![少角度改进指标对照](../outputs/improvements/summary/comparison.png)","",
        "![SheppLogan64少角度改进图像](../outputs/improvements/summary/SheppLogan64_images.png)","",
        "![XCAT128少角度改进图像](../outputs/improvements/summary/XCAT128_images.png)",""]
    target=OUT/"summary";target.mkdir(parents=True,exist_ok=True)
    save_csv(target/"classical_final_metrics.csv",[
        {**{k:v for k,v in p["selected"].items() if k!="history"},
         **{k:v for k,v in p["refinement"].items() if k!="history"},
         "iterations":p["refinement"]["total_iterations"]} for p in classical.values()])
    (target/"results_fragment.md").write_text("\n".join(lines),encoding="utf-8")
    write_json(target/"comparison.json",{"rows":report_rows,"nonnegative_TV_wins":win_nn,"TGV_wins_vs_nonnegative_TV":win_tgv,
        "validation_wins_vs_heldout_fixed":validation_better,"validation_wins_vs_original_full_fixed":validation_better_full,
        "max_final_objective_change":max(sensitivities)})
    os.environ.setdefault("MPLCONFIGDIR",str(target/"matplotlib_config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for i,name in enumerate(DATASETS):
        selected=[r for r in report_rows if r["dataset"]==name];pos=np.arange(4)
        for j,metric in enumerate(("nrmse","ssim")):
            ax=axes[i,j]
            for index,method in enumerate(("TV_control","TV_nonnegative","TGV_nonnegative","validation")):
                means=[];std=[]
                for r in selected:
                    if method=="validation":
                        row=lookup[(name,r["angles"],r["snr"],method)];means.append(float(row[metric+"_mean"]));std.append(float(row[metric+"_std"]))
                    else:
                        row=classical[(name,r["angles"],r["snr"],method)]["refinement"];means.append(row[metric]);std.append(0)
                ax.bar(pos+(index-1.5)*.19,means,.19,yerr=std,label=method,capsize=2)
            ax.set_xticks(pos,[f"{r['angles']}v/{r['snr']}dB" for r in selected]);ax.set_title(name+" "+metric);ax.grid(axis="y",alpha=.2)
            if i==0 and j==0:ax.legend(fontsize=7)
    fig.savefig(target/"comparison.png",dpi=150);plt.close(fig)
    for name in DATASETS:
        truth=truths[name];fig,axes=plt.subplots(2,6,figsize=(15,6),layout="constrained")
        for i,angles in enumerate((30,15)):
            reference=classical[(name,90,30,"TV_control")]["refinement"]
            entries=[("Truth",truth),("90v TV",np.load(ROOT/reference["array"]).reshape(truth.shape,order="F"))]
            for method in ("TV_control","TV_nonnegative","TGV_nonnegative"):
                row=classical[(name,angles,30,method)]["refinement"]
                entries.append((method,np.load(ROOT/row["array"]).reshape(truth.shape,order="F")))
            row=next(r for r in neural if (r["dataset"],r["angle_count"],r["snr_label_db"],r["seed"],r["checkpoint"])==(name,angles,30,0,"validation"))
            entries.append(("DIP+TV val seed0",np.load(ROOT/row["array"]).reshape(truth.shape,order="F")))
            for ax,(title,image) in zip(axes[i],entries):
                label="Truth" if title=="Truth" else ("90v,30dB\nTV reference" if title=="90v TV" else f"{angles}v,30dB\n{title}")
                ax.imshow(image,cmap="gray",vmin=truth.min(),vmax=truth.max());ax.set_title(label,fontsize=8);ax.axis("off")
        fig.savefig(target/(name+"_images.png"),dpi=150,bbox_inches="tight",pad_inches=.15);plt.close(fig)
    print(json.dumps(summary),flush=True)
    print(json.dumps(read_json(target/"comparison.json")),flush=True)

if __name__=="__main__":main()
