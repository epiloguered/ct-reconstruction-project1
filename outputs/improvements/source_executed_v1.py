"""少角度改进实验：统一原始-对偶 TV/TGV 与无 GT 的投影验证截点。"""
from __future__ import annotations
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from run_baselines import ROOT, DATASETS, load_matrix, image_metrics, save_csv
from run_regularization import spectral_lipschitz
from run_sparse_views import select_angles, write_json, read_json

OUT = ROOT / "outputs" / "improvements"
FRACTIONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

# 前向差分最后一个边界为零；伴随必须与这个边界定义严格匹配。
def gradient(u):
    g = np.zeros((2, *u.shape))
    g[0, :-1] = u[1:] - u[:-1]
    g[1, :, :-1] = u[:, 1:] - u[:, :-1]
    return g

def adjoint(p):
    u = np.zeros(p.shape[1:])
    u[:-1] -= p[0, :-1]; u[1:] += p[0, :-1]
    u[:, :-1] -= p[1, :, :-1]; u[:, 1:] += p[1, :, :-1]
    return u

# E 是对称梯度；交叉项乘 sqrt(2)，使普通欧氏范数等于张量 Frobenius 范数。
def symmetric_gradient(v):
    g0, g1 = gradient(v[0]), gradient(v[1])
    return np.stack((g0[0], g1[1], (g0[1] + g1[0]) / np.sqrt(2)))

def symmetric_adjoint(q):
    return np.stack((adjoint(np.stack((q[0], q[2]/np.sqrt(2)))),
                     adjoint(np.stack((q[2]/np.sqrt(2), q[1])))))

def project_ball(p, radius):
    return p / np.maximum(1.0, np.sqrt(np.sum(p*p, axis=0)) / radius)

# 目标为 0.5||Bu-d||² + alpha1||Du-v||2,1 + alpha0||Ev||2,1。
# TV 省去 v、E 两项。B=A/sqrt(L)，u=x/amplitude；归一化只改善数值尺度。
# Condat 型原始-对偶迭代；非负约束在 primal 更新中施加，不用去噪后随意裁剪代替联合求解。
def primal_dual(A, b, shape, amplitude, lipschitz, fraction, ratio, nonnegative,
                iterations=3000, initial=None):
    B = (A / np.sqrt(lipschitz)).tocsr(); BT = B.T.tocsr()
    d = b / (amplitude*np.sqrt(lipschitz))
    u = np.zeros(shape) if initial is None else initial[0].copy()
    v = np.zeros((2,*shape)) if initial is None else initial[1].copy()
    p = np.zeros_like(v) if initial is None else initial[2].copy()
    q = np.zeros((3,*shape)) if initial is None else initial[3].copy()
    tau = sigma = 0.2
    history=[]
    for step in range(1,iterations+1):
        grad = (BT @ (B @ u.reshape(-1,order="F") - d)).reshape(shape,order="F")
        new_u = u - tau*(grad + adjoint(p))
        if nonnegative: new_u = np.maximum(new_u,0)
        new_v = v + tau*(p-symmetric_adjoint(q)) if ratio else v
        p = project_ball(p + sigma*(gradient(2*new_u-u) - (2*new_v-v) if ratio else gradient(2*new_u-u)), fraction)
        if ratio: q = project_ball(q + sigma*symmetric_gradient(2*new_v-v), ratio*fraction)
        u,v = new_u,new_v
        if step==1 or step%250==0 or step==iterations:
            residual=B@u.reshape(-1,order="F")-d
            regularization=fraction*np.linalg.norm(gradient(u)-v if ratio else gradient(u),axis=0).sum()
            if ratio:regularization += ratio*fraction*np.linalg.norm(symmetric_gradient(v),axis=0).sum()
            history.append({"iteration":step,"objective_scaled":float(0.5*(residual@residual)+regularization)})
    return amplitude*u.reshape(-1,order="F"),history,(u,v,p,q)

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def classical(args):
    target=args.output_dir/"classical"; target.mkdir(parents=True,exist_ok=True)
    for name,size in DATASETS.items():
        full=load_matrix(name,ROOT/"outputs/cache"); truth=np.loadtxt(ROOT/"Data"/name/"gt.csv",delimiter=",")
        amplitude=float(np.ptp(truth))
        for angles in args.angles:
            A,_,_,_=select_angles(full,np.zeros(full.shape[0]),angles); L=spectral_lipschitz(A)
            methods=("TV_control","TV_nonnegative","TGV_nonnegative") if angles<90 else ("TV_control",)
            for snr in args.snrs:
                full_b=np.loadtxt(ROOT/"Data"/name/f"b_hat_{snr}db.csv",delimiter=",").ravel()
                _,b,_,_=select_angles(full,full_b,angles)
                for method in methods:
                    key=f"{name}_{angles}angles_{snr}db_{method}"
                    record=target/(key+".json")
                    if record.exists():continue
                    candidates=[];best=None
                    for frac in FRACTIONS:
                        for ratio in ((0.5,1.,2.) if method=="TGV_nonnegative" else (0.,)):
                            start=time.perf_counter()
                            x,history,state=primal_dual(A,b,truth.shape,amplitude,L,frac,ratio,method!="TV_control",args.iterations)
                            row={"dataset":name,"angle_count":angles,"snr_label_db":snr,"method":method,"fraction":frac,"tgv_ratio":ratio,
                                 "lambda_raw":frac*amplitude*L,"iterations":args.iterations,"seconds":time.perf_counter()-start,
                                 "selection":"oracle minimum GT NRMSE","nonnegative":method!="TV_control",**image_metrics(x,truth,A,b)}
                            candidate_key=f"{key}_p{frac:g}_r{ratio:g}"
                            path=target/"candidates"/(candidate_key+".npy");path.parent.mkdir(exist_ok=True)
                            np.save(path,x);row["array"]=str(path.relative_to(ROOT))
                            row["history"]=history;candidates.append(row)
                            if best is None or row["nrmse"]<best[0]["nrmse"]:best=(row,x,state)
                    row,x,state=best
                    # 对选中参数继续迭代，检查预算敏感性；续迭代结果与原预算结果分别报告。
                    refined,trace,_=primal_dual(A,b,truth.shape,amplitude,L,row["fraction"],row["tgv_ratio"],method!="TV_control",args.refine_iterations,state)
                    refined_path=target/(key+"_refined.npy");np.save(refined_path,refined)
                    refinement={"array":str(refined_path.relative_to(ROOT)),"total_iterations":args.iterations+args.refine_iterations,
                                "history":trace,**image_metrics(refined,truth,A,b)}
                    write_json(record,{"selected":row,"candidates":candidates,"refinement":refinement})
                    print(f"[classical] {key} NRMSE={row['nrmse']:.5f} refined={refinement['nrmse']:.5f}",flush=True)
    rows=[read_json(p)["selected"] for p in sorted(target.glob("*.json"))]
    save_csv(target/"selected_metrics.csv",[{k:v for k,v in r.items() if k!="history"} for r in rows])

# 验证集从已采集角度中抽取完整角度块；不访问未采集投影。
def split_observed(A,b,angles):
    lines=A.shape[0]//angles
    # 固定每五角留一角：30 角留6角，15角留3角；全部种子使用相同划分。
    val_angles=np.arange(2,angles,5)
    val_rows=(val_angles[:,None]*lines+np.arange(lines)).ravel()
    train_rows=np.setdiff1d(np.arange(A.shape[0]),val_rows)
    return train_rows,val_rows

def neural(args):
    import torch
    from run_dip_extension import initialization, observation_scale, sparse_tensor, SparseProjection, image_vector, mean_tv
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    target=args.output_dir/"neural";target.mkdir(parents=True,exist_ok=True)
    for name,size in DATASETS.items():
        full=load_matrix(name,ROOT/"outputs/cache");truth=np.loadtxt(ROOT/"Data"/name/"gt.csv",delimiter=",")
        for angles in (a for a in args.angles if a<90):
            for snr in args.snrs:
                full_b=np.loadtxt(ROOT/"Data"/name/f"b_hat_{snr}db.csv",delimiter=",").ravel()
                A,b,_,_=select_angles(full,full_b,angles);train,val=split_observed(A,b,angles)
                for seed in (0,1,2):
                    key=f"{name}_{angles}angles_{snr}db_seed{seed}"
                    record=target/(key+".json")
                    if record.exists():continue
                    net,z,digest=initialization(seed,size,"cuda")
                    At=A[train];bt=b[train];Av=A[val];bv=b[val]
                    scale=observation_scale(At,bt)
                    tA=sparse_tensor(At,"cuda");tAT=sparse_tensor(At.T,"cuda")
                    tb=torch.as_tensor(bt,dtype=torch.float32,device="cuda").reshape(-1,1)
                    tV=sparse_tensor(Av,"cuda");vb=torch.as_tensor(bv,dtype=torch.float32,device="cuda").reshape(-1,1)
                    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3);beta=10.**(-snr/10)
                    best_value=float("inf");history=[];best_x=best_state=None;best_step=None
                    torch.cuda.synchronize();start=time.perf_counter()
                    for step in range(1,2001):
                        optimizer.zero_grad(set_to_none=True);scaled=net(z);image=scaled*scale
                        residual=SparseProjection.apply(image_vector(image),tA,tAT)-tb
                        loss=residual.square().sum()/(2*tb.square().sum())+beta*mean_tv(scaled)
                        if not torch.isfinite(loss):raise FloatingPointError(key)
                        loss.backward();optimizer.step()
                        if step==1 or step%50==0:
                            with torch.no_grad():
                                vec=image_vector(net(z)*scale)
                                validation=float((torch.sparse.mm(tV,vec)-vb).square().sum()/vb.square().sum())
                                x=vec.ravel().cpu().numpy().astype(np.float64)
                            history.append({"iteration":step,"validation_relative_mse":validation})
                            if validation<best_value:
                                best_value=validation;best_step=step;best_x=x.copy()
                                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    # 到此才用 GT 计算图像指标，GT 从未参与训练或验证截点选择。
                    entries=[]
                    for label,vec,iteration in (("validation",best_x,best_step),("heldout_fixed",x,2000)):
                        path=target/(key+"_"+label+".npy");np.save(path,vec)
                        entries.append({"dataset":name,"angle_count":angles,"snr_label_db":snr,"seed":seed,"checkpoint":label,
                                        "iteration":iteration,"array":str(path.relative_to(ROOT)),**image_metrics(vec,truth,A,b)})
                    torch.save({"state_dict":best_state,"noise":z.detach().cpu(),"scale":scale,"iteration":best_step},target/(key+"_validation.pt"))
                    write_json(record,{"metrics":entries,"history":history,"train_rows":train.tolist(),"validation_rows":val.tolist(),
                                       "initialization_sha256":digest,"scale":scale,"beta":beta,"seconds":elapsed,
                                       "GT_used_for_training_or_checkpoint":False})
                    print(f"[heldout] {key} step={best_step} NRMSE={entries[0]['nrmse']:.5f} fixed={entries[1]['nrmse']:.5f}",flush=True)
    rows=[r for p in sorted(target.glob("*.json")) for r in read_json(p)["metrics"]];save_csv(target/"individual_metrics.csv",rows)
    aggregate=[]
    for name in DATASETS:
        for angles in (a for a in args.angles if a<90):
            for snr in args.snrs:
                for checkpoint in ("validation","heldout_fixed"):
                    group=[r for r in rows if (r["dataset"],r["angle_count"],r["snr_label_db"],r["checkpoint"])==(name,angles,snr,checkpoint)]
                    if len(group)!=3:continue
                    row={"dataset":name,"angle_count":angles,"snr_label_db":snr,"checkpoint":checkpoint,"n_seeds":3}
                    for metric in ("nrmse","psnr_db","ssim","relative_residual","iteration"):
                        values=[r[metric] for r in group];row[metric+"_mean"]=float(np.mean(values));row[metric+"_std"]=float(np.std(values,ddof=1))
                    aggregate.append(row)
    save_csv(target/"aggregate_metrics.csv",aggregate)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("classical","neural"),required=True)
    parser.add_argument("--angles",type=int,nargs="+",default=[90,30,15])
    parser.add_argument("--snrs",type=int,nargs="+",default=[20,30])
    parser.add_argument("--iterations",type=int,default=6000)
    parser.add_argument("--refine-iterations",type=int,default=6000)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args();args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={"phase":args.phase,"angles":args.angles,"snrs":args.snrs,"iterations":args.iterations,"refine_iterations":args.refine_iterations,
              "fractions":FRACTIONS,"tgv_ratios":[0.5,1,2],"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    from run_baselines import matrix_source
    manifest["dependencies"]={filename:hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
        for filename in ("run_baselines.py","run_regularization.py","run_sparse_views.py","run_dip_extension.py")}
    manifest["inputs"]={}
    for name in DATASETS:
        for path_input in (matrix_source(name),ROOT/"Data"/name/"gt.csv",*(ROOT/"Data"/name/f"b_hat_{snr}db.csv" for snr in args.snrs)):
            stat=path_input.stat();item={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}
            if stat.st_size<20_000_000:item["sha256"]=hashlib.sha256(path_input.read_bytes()).hexdigest()
            manifest["inputs"][str(path_input.relative_to(ROOT))]=item
    manifest["neural_protocol"]={"iterations":2000,"sample_every":50,"learning_rate":0.001,"seeds":[0,1,2],
        "validation_angle_indices":"arange(2,angle_count,5)","GT_for_training_or_checkpoint":False,
        "checkpoint_selection":"minimum heldout relative projection MSE, earliest on tie",
        "no_full_data_refit":True}
    path=args.output_dir/(args.phase+"_manifest.json")
    if path.exists() and read_json(path)!=json.loads(json.dumps(manifest)):raise ValueError("Configuration/source changed: use new output directory")
    write_json(path,manifest)
    (classical if args.phase=="classical" else neural)(args)

if __name__=="__main__":main()
