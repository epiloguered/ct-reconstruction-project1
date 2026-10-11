"""非 DIP 的少角度实验：非负 TV、小波+TV、PnP-BM3D；保留所有候选。"""
from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
import pywt
from scipy.sparse.linalg import LinearOperator,cg
from run_baselines import ROOT,DATASETS,load_matrix,image_metrics,save_csv,matrix_source
from run_regularization import spectral_lipschitz
from run_sparse_views import select_angles,read_json,write_json
from run_sparse_improvements import gradient,adjoint,project_ball,primal_dual,FRACTIONS

OUT=ROOT/'outputs/min_angles'
ANGLES=(30,15,10,6,3)
SNRS=(20,30)
WAVE_RATIOS=(.1,1.,10.)
PNP_RHOS=(.003,.03,.3)
PNP_SIGMAS=(.01,.03,.08)

class Wavelet:
    """周期边界正交 db2；只惩罚细节系数，保留最低频近似项。"""
    def __init__(self,shape):
        self.shape=shape
        coeff=pywt.wavedec2(np.zeros(shape),'db2',mode='periodization',level=3)
        arr,self.slices=pywt.coeffs_to_array(coeff)
        self.mask=np.ones_like(arr);self.mask[self.slices[0]]=0
    def forward(self,u):
        return pywt.coeffs_to_array(pywt.wavedec2(u,'db2',mode='periodization',level=3))[0]
    def transpose(self,c):
        return pywt.waverec2(pywt.array_to_coeffs(c,self.slices,output_format='wavedec2'),
                             'db2',mode='periodization')[:self.shape[0],:self.shape[1]]

def wavelet_tv(A,b,shape,amplitude,L,p,ratio,iterations,initial=None):
    """归一化目标 .5||Bu-d||²+p TV(u)+ratio*p||W_detail u||1，u>=0。"""
    B=(A/np.sqrt(L)).tocsr();BT=B.T.tocsr();d=b/(amplitude*np.sqrt(L));W=Wavelet(shape)
    if initial is None:u=np.zeros(shape);v=np.zeros((2,*shape));q=np.zeros(shape)
    else:u,v,q=(a.copy() for a in initial)
    tau,sigma=1.5,.005;trace=[]
    for step in range(1,iterations+1):
        g=(BT@(B@u.reshape(-1,order='F')-d)).reshape(shape,order='F')
        new=np.maximum(u-tau*(g+adjoint(v)+W.transpose(q)),0)
        bar=2*new-u;v=project_ball(v+sigma*gradient(bar),p)
        q=np.clip(q+sigma*W.forward(bar),-ratio*p,ratio*p)*W.mask
        u=new
        if step==1 or step%250==0 or step==iterations:
            residual=B@u.reshape(-1,order='F')-d
            objective=.5*(residual@residual)+p*np.linalg.norm(gradient(u),axis=0).sum()+ratio*p*np.abs(W.forward(u)*W.mask).sum()
            trace.append({'iteration':step,'objective_scaled':float(objective)})
    return amplitude*u.reshape(-1,order='F'),trace,(u,v,q)

def bm3d_denoise(image,sigma,profile='lc'):
    """非负限制在去噪输出施加；输入不按 GT 范围裁剪。"""
    import bm3d
    from bm3d.profiles import BM3DProfile,BM3DProfileLC
    config=BM3DProfileLC() if profile=='lc' else BM3DProfile();config.num_threads=4
    return np.maximum(bm3d.bm3d(image,sigma,profile=config),0)

def pnp_admm(A,b,shape,amplitude,L,rho,sigma,iterations,start,initial=None,profile='lc'):
    """ADMM 数据子问题用 CG；BM3D替代先验近端，不能称为显式凸目标最小化。"""
    B=(A/np.sqrt(L)).tocsr();BT=B.T.tocsr();d=b/(amplitude*np.sqrt(L));rhs0=BT@d
    if initial is None:z=start/amplitude;dual=np.zeros_like(z);x=z.copy()
    else:x,z,dual=(a.copy() for a in initial)
    op=LinearOperator((A.shape[1],A.shape[1]),matvec=lambda t:BT@(B@t)+rho*t,dtype=np.float64)
    diagonal=np.asarray(B.power(2).sum(axis=0)).ravel()+rho
    pre=LinearOperator(op.shape,matvec=lambda t:t/diagonal,dtype=np.float64)
    trace=[];nonconverged=0
    for step in range(1,iterations+1):
        x,info=cg(op,rhs0+rho*(z-dual),x0=x,M=pre,rtol=1e-6,atol=0,maxiter=200)
        if info<0:raise RuntimeError('CG breakdown')
        nonconverged+=info>0
        old=z.copy();z=bm3d_denoise((x+dual).reshape(shape,order='F'),sigma,profile).reshape(-1,order='F')
        dual+=x-z
        if not np.isfinite(z).all():raise FloatingPointError('PnP nonfinite')
        if step==1 or step%10==0 or step==iterations:
            trace.append({'iteration':step,'primal_relative':float(np.linalg.norm(x-z)/max(np.linalg.norm(z),1e-15)),
                          'dual_relative':float(rho*np.linalg.norm(z-old)/max(np.linalg.norm(z),1e-15)),
                          'cg_not_converged_count':int(nonconverged)})
    return amplitude*z,trace,(x,z,dual)

def stamp(args):
    deps=['run_min_angles.py','run_baselines.py','run_regularization.py','run_sparse_views.py','run_sparse_improvements.py']
    record={'phase':args.phase,'datasets':args.datasets,'angles':args.angles,'snrs':args.snrs,
            'tv_grid':FRACTIONS,'wave_ratios':WAVE_RATIOS,'wave_tv_multipliers':[.3,1.,3.],
            'pnp_rhos':PNP_RHOS,'pnp_sigmas':PNP_SIGMAS,'grid_steps':args.grid_steps,'refine_steps':args.refine_steps,
            'pnp_grid_steps':args.pnp_grid_steps,'pnp_refine_steps':args.pnp_refine_steps,
            'selection':'oracle minimum GT NRMSE at grid endpoint; continue selected state without reselection',
            'pnp_profile':'BM3DProfileLC, 4 threads, all stages; nonnegative output',
            'wavelet':'orthogonal db2, periodization, level3; approximation not penalized',
            'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in deps},
            'packages':{f:importlib.metadata.version(f) for f in ('numpy','scipy','PyWavelets','bm3d','bm4d')},'inputs':{}}
    for name in args.datasets:
        for p in (matrix_source(name),ROOT/'Data'/name/'gt.csv',*(ROOT/'Data'/name/f'b_hat_{s}db.csv' for s in args.snrs)):
            stat=p.stat();item={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            if p.stat().st_size<20_000_000:item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            record['inputs'][str(p.relative_to(ROOT))]=item
    path=args.output_dir/f"manifest_{args.phase}_{'_'.join(args.datasets)}.json"
    if path.exists() and read_json(path)!=json.loads(json.dumps(record)):raise ValueError('Changed configuration/source: use a new directory')
    write_json(path,record)

def run(args):
    target=args.output_dir/args.phase;target.mkdir(parents=True,exist_ok=True)
    for name in args.datasets:
        full=load_matrix(name,ROOT/'outputs/cache');truth=np.loadtxt(ROOT/'Data'/name/'gt.csv',delimiter=',');s=float(np.ptp(truth))
        for angles in args.angles:
            A,_,angle_ids,row_ids=select_angles(full,np.zeros(full.shape[0]),angles);L=spectral_lipschitz(A)
            for snr in args.snrs:
                key=f'{name}_{angles}angles_{snr}db';path=target/(key+'.json')
                if path.exists():continue
                full_b=np.loadtxt(ROOT/'Data'/name/f'b_hat_{snr}db.csv',delimiter=',').ravel();b=full_b[row_ids]
                control=None
                if args.phase!='tv':
                    control=read_json(args.output_dir/'tv'/(key+'.json'))
                if args.phase=='tv':grid=[(p,0.) for p in FRACTIONS]
                elif args.phase=='wave':grid=[(control['selected']['p']*mult,r) for mult in (.3,1.,3.) for r in WAVE_RATIOS]
                else:grid=[(rho,sigma) for rho in PNP_RHOS for sigma in PNP_SIGMAS]
                start=np.load(ROOT/control['refined']['array']) if control else None
                candidates=[];best=None;tic=time.perf_counter()
                for i,(p,r) in enumerate(grid):
                    cpath=target/'candidates'/f'{key}_{i:02d}.npy';cpath.parent.mkdir(exist_ok=True)
                    if args.phase=='tv':
                        x,trace,state=primal_dual(A,b,truth.shape,s,L,p,0.,True,args.grid_steps)
                    elif args.phase=='wave':
                        x,trace,state=wavelet_tv(A,b,truth.shape,s,L,p,r,args.grid_steps)
                    else:x,trace,state=pnp_admm(A,b,truth.shape,s,L,p,r,args.pnp_grid_steps,start)
                    np.save(cpath,x)
                    row={'dataset':name,'angle_count':angles,'snr_label_db':snr,'method':args.phase,'p':p,'ratio':r,
                         'array':str(cpath.relative_to(ROOT)),**image_metrics(x,truth,A,b),'history':trace}
                    candidates.append(row)
                    if best is None or row['nrmse']<best[0]['nrmse']:best=(row,state)
                selected,state=best;p,r=selected['p'],selected['ratio']
                if args.phase=='tv':x,trace,_=primal_dual(A,b,truth.shape,s,L,p,0.,True,args.refine_steps,state)
                elif args.phase=='wave':x,trace,_=wavelet_tv(A,b,truth.shape,s,L,p,r,args.refine_steps,state)
                else:x,trace,_=pnp_admm(A,b,truth.shape,s,L,p,r,args.pnp_refine_steps,start,state)
                refined_path=target/(key+'_refined.npy');np.save(refined_path,x)
                refined={'array':str(refined_path.relative_to(ROOT)),**image_metrics(x,truth,A,b),'history':trace,
                         'total_iterations':args.pnp_grid_steps+args.pnp_refine_steps if args.phase=='pnp' else args.grid_steps+args.refine_steps}
                write_json(path,{'selected':selected,'candidates':candidates,'refined':refined,'amplitude':s,'L':L,
                                'angle_ids':angle_ids.tolist(),'row_ids':row_ids.tolist(),
                                'actual_subset_snr_db':float(20*np.log10(np.linalg.norm(A@truth.reshape(-1,order='F'))/np.linalg.norm(b-A@truth.reshape(-1,order='F')))),
                                'seconds':time.perf_counter()-tic,'GT_used_for_scale_and_selection':True,
                                'warm_start':control['refined']['array'] if args.phase=='pnp' else 'zero'})
                print(f'[{args.phase}] {key} selected={selected["nrmse"]:.5f} refined={refined["nrmse"]:.5f} p={p:g} ratio={r:g}',flush=True)
        records=[]
        for path in sorted(target.glob(name+'_*.json')):
            d=read_json(path);records.append({**{k:v for k,v in d['selected'].items() if k!='history'},
                                             **{k:v for k,v in d['refined'].items() if k!='history'}})
        save_csv(target/(name+'_final_metrics.csv'),records)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('tv','wave','pnp'),required=True)
    parser.add_argument('--datasets',nargs='+',choices=tuple(DATASETS),default=list(DATASETS))
    parser.add_argument('--angles',nargs='+',type=int,default=list(ANGLES));parser.add_argument('--snrs',nargs='+',type=int,default=list(SNRS))
    parser.add_argument('--grid-steps',type=int,default=6000);parser.add_argument('--refine-steps',type=int,default=6000)
    parser.add_argument('--pnp-grid-steps',type=int,default=40);parser.add_argument('--pnp-refine-steps',type=int,default=120)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    stamp(args);run(args)
if __name__=='__main__':main()
