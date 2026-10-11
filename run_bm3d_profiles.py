"""在决定角度边界的30dB案例检查常规BM3D配置，参数沿用LC选中值。"""
import argparse,hashlib,time
import numpy as np
from run_min_angles import ROOT,OUT,DATASETS,load_matrix,select_angles,read_json,write_json,image_metrics,pnp_admm

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=tuple(DATASETS),required=True)
    args=parser.parse_args();name=args.dataset;target=OUT/'profile_check';records=target/'records';records.mkdir(parents=True,exist_ok=True)
    source_files=['run_bm3d_profiles.py','run_min_angles.py','run_baselines.py','run_sparse_views.py']
    manifest={'dataset':name,'angles':[15,10],'snr':30,'profile':'BM3DProfile (np), 4 threads, all stages',
              'grid_steps':40,'refine_steps':120,'parameters':'transferred from LC grid selection; not reoptimized for np',
              'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},
              'lc_record_hashes':{a:hashlib.sha256((OUT/'pnp'/f'{name}_{a}angles_30db.json').read_bytes()).hexdigest() for a in (15,10)}}
    path=target/f'manifest_{name}.json'
    import json
    if path.exists() and read_json(path)!=json.loads(json.dumps(manifest)):raise ValueError('Changed source/input: do not overwrite old profile check')
    write_json(path,manifest)
    full=load_matrix(name,ROOT/'outputs/cache');truth=np.loadtxt(ROOT/'Data'/name/'gt.csv',delimiter=',')
    bfull=np.loadtxt(ROOT/'Data'/name/'b_hat_30db.csv',delimiter=',').ravel()
    for a in (15,10):
        key=f'{name}_{a}angles_30db';path=records/(key+'.json')
        if path.exists():continue
        lc=read_json(OUT/'pnp'/(key+'.json'));s=lc['selected'];A,b,_,_=select_angles(full,bfull,a);start=np.load(ROOT/lc['warm_start']);tic=time.perf_counter()
        x,trace,state=pnp_admm(A,b,truth.shape,lc['amplitude'],lc['L'],s['p'],s['ratio'],40,start,profile='np')
        grid_path=records/(key+'_40.npy');np.save(grid_path,x)
        row={**{k:v for k,v in s.items() if k not in ('history','array')},'method':'pnp_np','array':str(grid_path.relative_to(ROOT)),
             **image_metrics(x,truth,A,b),'history':trace}
        x,trace,_=pnp_admm(A,b,truth.shape,lc['amplitude'],lc['L'],s['p'],s['ratio'],120,start,state,profile='np')
        final_path=records/(key+'_160.npy');np.save(final_path,x)
        final={'array':str(final_path.relative_to(ROOT)),**image_metrics(x,truth,A,b),'history':trace,'total_iterations':160}
        result={**{k:v for k,v in lc.items() if k not in ('selected','refined','candidates','seconds')},
                'selected':row,'candidates':[row],'refined':final,'seconds':time.perf_counter()-tic,
                'parameter_selection':'LC parameters transferred; np profile not retuned'}
        write_json(path,result);print(f'[np] {key} grid={row["nrmse"]:.5f} final={final["nrmse"]:.5f}',flush=True)
if __name__=='__main__':main()
