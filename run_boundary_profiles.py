"""将质量较好的常规BM3D配置继续应用到7/8/9角边界；沿用各自LC参数。"""
import hashlib,time
import numpy as np
from run_min_angles import ROOT,OUT,load_matrix,select_angles,read_json,write_json,image_metrics,pnp_admm
def main():
    name='SheppLogan64';source=ROOT/'outputs/min_angles_boundary/pnp';target=OUT/'profile_boundary';records=target/'records';records.mkdir(parents=True,exist_ok=True)
    m={'dataset':name,'angles':[9,8,7],'snr':30,'profile':'BM3DProfile (np), 4 threads, all stages','grid_steps':40,'refine_steps':120,
       'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('run_boundary_profiles.py','run_min_angles.py','run_baselines.py','run_sparse_views.py')},
       'lc_record_hashes':{a:hashlib.sha256((source/f'{name}_{a}angles_30db.json').read_bytes()).hexdigest() for a in (9,8,7)}}
    import json
    manifest=target/'manifest.json'
    if manifest.exists() and read_json(manifest)!=json.loads(json.dumps(m)):raise ValueError('Changed source/input')
    write_json(manifest,m)
    full=load_matrix(name,ROOT/'outputs/cache');truth=np.loadtxt(ROOT/'Data'/name/'gt.csv',delimiter=',');bfull=np.loadtxt(ROOT/'Data'/name/'b_hat_30db.csv',delimiter=',').ravel()
    for a in (9,8,7):
        key=f'{name}_{a}angles_30db';path=records/(key+'.json')
        if path.exists():continue
        lc=read_json(source/(key+'.json'));s=lc['selected'];A,b,_,_=select_angles(full,bfull,a);start=np.load(ROOT/lc['warm_start']);tic=time.perf_counter()
        x,h,state=pnp_admm(A,b,truth.shape,lc['amplitude'],lc['L'],s['p'],s['ratio'],40,start,profile='np')
        p=records/(key+'_40.npy');np.save(p,x)
        row={**{k:v for k,v in s.items() if k not in ('history','array')},'method':'pnp_np','array':str(p.relative_to(ROOT)),**image_metrics(x,truth,A,b),'history':h}
        x,h,_=pnp_admm(A,b,truth.shape,lc['amplitude'],lc['L'],s['p'],s['ratio'],120,start,state,profile='np')
        p=records/(key+'_160.npy');np.save(p,x);f={'array':str(p.relative_to(ROOT)),**image_metrics(x,truth,A,b),'history':h,'total_iterations':160}
        write_json(path,{**{k:v for k,v in lc.items() if k not in ('selected','refined','candidates','seconds')},'selected':row,'candidates':[row],'refined':f,
                         'seconds':time.perf_counter()-tic,'parameter_selection':'LC parameters transferred; np profile not retuned'})
        print(f'[boundary np] {a} angles grid={row["nrmse"]:.5f} final={f["nrmse"]:.5f}',flush=True)
if __name__=='__main__':main()
