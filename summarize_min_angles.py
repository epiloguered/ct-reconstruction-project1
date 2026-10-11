"""独立核验更少角度实验，导出质量—角度权衡、历史对照和完整三指标。"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
from run_baselines import ROOT,DATASETS,load_matrix,save_csv
from run_sparse_views import read_json,write_json
from run_min_angles import OUT,ANGLES,SNRS,FRACTIONS,WAVE_RATIOS,PNP_RHOS,PNP_SIGMAS,pnp_admm
from verify_extension import independent_metrics

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    errors=[];checked=0;arrays=set();records={};rows=[];cache={}
    def check(condition,label):
        nonlocal checked
        checked+=1
        if not condition:errors.append(label)
    full={n:load_matrix(n,ROOT/'outputs/cache') for n in DATASETS}
    truths={n:np.loadtxt(ROOT/'Data'/n/'gt.csv',delimiter=',') for n in DATASETS}
    def case(name,a,s):
        key=name,a,s
        if key not in cache:
            m=full[name];detectors=m.shape[0]//90
            angle_ids=np.floor(np.arange(a)*90/a).astype(int)
            row_ids=np.array([j*detectors+i for j in angle_ids for i in range(detectors)])
            b=np.loadtxt(ROOT/'Data'/name/f'b_hat_{s}db.csv',delimiter=',').ravel()
            cache[key]=m[row_ids].tocsr(),b[row_ids],truths[name],angle_ids,row_ids
        return cache[key]
    def verify(row,path):
        A,b,truth,_,_=case(row['dataset'],row['angle_count'],row['snr_label_db'])
        x=np.load(ROOT/path);arrays.add(path)
        check(x.shape==(A.shape[1],) and np.isfinite(x).all(),'shape finite '+path)
        check(x.min()>=0,'nonnegative '+path)
        for k,v in independent_metrics(x,truth,A,b).items():
            check(np.isclose(v,row[k],rtol=1e-9,atol=1e-10),'independent metric '+k+' '+path)
        return x
    boundary=ROOT/'outputs/min_angles_boundary'
    manifests=sorted(OUT.glob('manifest_*.json'))+sorted(boundary.glob('manifest_*.json'))
    check(len(manifests)==7,'seven phase/dataset manifests including boundary refinement')
    for path in manifests:
        m=read_json(path)
        is_boundary=path.parent==boundary
        check(m['angles']==([9,8,7] if is_boundary else list(ANGLES)) and m['snrs']==([30] if is_boundary else list(SNRS)),'manifest condition grid')
        check((m['grid_steps'],m['refine_steps'],m['pnp_grid_steps'],m['pnp_refine_steps'])==(6000,6000,40,120),'manifest budgets')
        for f,digest in m['source_hashes'].items():
            check(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest,'source '+f)
        for f,item in m['inputs'].items():
            p=ROOT/f;stat=p.stat()
            check(stat.st_size==item['size'] and stat.st_mtime_ns==item['mtime_ns'],'input metadata '+f)
            if 'sha256' in item:check(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'input hash '+f)
    profile_manifests=sorted((OUT/'profile_check').glob('manifest_*.json'))+[OUT/'profile_boundary/manifest.json']
    check(len(profile_manifests)==3,'three normal-profile manifests')
    for path in profile_manifests:
        m=read_json(path);is_boundary=path.parent==OUT/'profile_boundary'
        check(m['angles']==([9,8,7] if is_boundary else [15,10]) and m['snr']==30,'normal profile condition grid')
        for f,digest in m['source_hashes'].items():check(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest,'profile source '+f)
        for angle,digest in m['lc_record_hashes'].items():
            lc_source=boundary/'pnp' if is_boundary else OUT/'pnp'
            check(hashlib.sha256((lc_source/f"{m['dataset']}_{angle}angles_30db.json").read_bytes()).hexdigest()==digest,'LC parameter source fingerprint')
    candidate_count=0
    for phase in ('tv','wave','pnp','pnp_np'):
        files=sorted((OUT/'profile_check/records').glob('*.json'))+sorted((OUT/'profile_boundary/records').glob('*.json')) if phase=='pnp_np' else sorted((OUT/phase).glob('*.json'))+sorted((boundary/phase).glob('*.json'))
        check(len(files)==(7 if phase=='pnp_np' else 23),'case count '+phase)
        for path in files:
            data=read_json(path);s=data['selected'];f=data['refined'];key=s['dataset'],s['angle_count'],s['snr_label_db'],phase
            check(key not in records,'unique case');records[key]=data
            A,b,truth,angles,ids=case(*key[:3])
            check(data['angle_ids']==angles.tolist() and data['row_ids']==ids.tolist(),'independent angle rows '+str(key))
            check(data['GT_used_for_scale_and_selection'] is True,'oracle declaration')
            check(data['amplitude']==np.ptp(truth),'GT amplitude')
            signal=A@truth.reshape(-1,order='F');snr=20*np.log10(np.linalg.norm(signal)/np.linalg.norm(b-signal))
            check(np.isclose(snr,data['actual_subset_snr_db'],rtol=1e-12),'actual subset SNR')
            candidates=data['candidates'];check(len(candidates)==(1 if phase=='pnp_np' else 9),'candidate count');candidate_count+=len(candidates)
            if phase=='tv':grid={(p,0.) for p in FRACTIONS}
            elif phase=='wave':
                control=records[(*key[:3],'tv')];grid={(control['selected']['p']*mult,r) for mult in (.3,1.,3.) for r in WAVE_RATIOS}
            elif phase=='pnp':grid=set(itertools.product(PNP_RHOS,PNP_SIGMAS))
            else:
                lc=records[(*key[:3],'pnp')]['selected'];grid={(lc['p'],lc['ratio'])}
            check({(r['p'],r['ratio']) for r in candidates}==grid,'candidate parameter grid '+str(key))
            for r in candidates:verify(r,r['array'])
            check(s['nrmse']==min(r['nrmse'] for r in candidates),'oracle selected at grid endpoint')
            check(any(s==r for r in candidates),'selected belongs to candidates')
            verify({**s,**f},f['array'])
            check(f['total_iterations']==(160 if phase.startswith('pnp') else 12000),'final iteration budget')
            if phase.startswith('pnp'):
                check(data['warm_start']==records[(*key[:3],'tv')]['refined']['array'],'TV warm start provenance')
                check(all(r['history'][-1]['cg_not_converged_count']==0 for r in candidates) and f['history'][-1]['cg_not_converged_count']==0,'all CG subproblems converged')
            else:
                check(all(len(r['history'])==25 for r in candidates) and len(f['history'])==25,'objective samples')
            rows.append({**{k:v for k,v in s.items() if k!='history'},**{k:v for k,v in f.items() if k!='history'},
                         'grid_nrmse':s['nrmse'],'actual_subset_snr_db':data['actual_subset_snr_db'],'seconds':data['seconds']})
    expected=set(itertools.product(DATASETS,ANGLES,SNRS,('tv','wave','pnp')))
    expected|=set(itertools.product(('SheppLogan64',),(9,8,7),(30,),('tv','wave','pnp')))
    expected|=set(itertools.product(DATASETS,(15,10),(30,),('pnp_np',)))
    expected|=set(itertools.product(('SheppLogan64',),(9,8,7),(30,),('pnp_np',)))
    check(set(records)==expected,'complete 76-case grid');check(candidate_count==628,'628 candidates')
    # 多线程BM3D不保证逐位一致：保留严格比较结果，同时量化图像指标变化。
    replay=[];replay_dir=OUT/'replay_checks';replay_dir.mkdir(exist_ok=True)
    if not errors:
        for name,angle,snr,method,budget in (('SheppLogan64',10,30,'pnp_np',160),('XCAT128',15,30,'pnp',160),('XCAT128',10,20,'pnp',40)):
            data=records[(name,angle,snr,method)];s=data['selected'];A,b,truth,_,_=case(name,angle,snr)
            start=np.load(ROOT/data['warm_start'])
            original=data['refined'] if budget==160 else s;saved=np.load(ROOT/original['array']);repeats=[]
            for trial in (1,2):
                x,h,_=pnp_admm(A,b,truth.shape,data['amplitude'],data['L'],s['p'],s['ratio'],budget,start,profile='np' if method=='pnp_np' else 'lc')
                check(np.isfinite(x).all() and x.min()>=0,'replay finite/nonnegative')
                check(h[-1]['cg_not_converged_count']==0,'replay CG convergence')
                rel=float(np.linalg.norm(x-saved)/np.linalg.norm(saved));path=replay_dir/f'{name}_{angle}angles_{snr}db_{method}_{budget}_repeat{trial}.npy';np.save(path,x)
                repeats.append({'relative_error':rel,'strict_match_1e_minus6':bool(rel<1e-6),'array':str(path.relative_to(ROOT)),**independent_metrics(x,truth,A,b)})
            replay.append({'dataset':name,'angles':angle,'snr':snr,'method':method,'iterations':budget,
                           'original':{k:original[k] for k in ('nrmse','psnr_db','ssim')},'repeats':repeats})
    variation=any(not trial['strict_match_1e_minus6'] for r in replay for trial in r['repeats'])
    audit={'status':'failed' if errors else ('passed_with_replay_variation' if variation else 'passed'),
           'artifact_verification':'failed' if errors else 'passed','strict_replay_passed':not variation,
           'assertions':checked,'unique_arrays':len(arrays),'selected_cases':len(records),'candidates':candidate_count,
           'pnp_replays':replay,'errors':errors,'warnings':['4-thread BM3D: strict replay tolerance 1e-6 not met; reported metrics belong to saved arrays'] if variation else []}
    write_json(OUT/'audit_summary.json',audit)
    if errors:raise AssertionError(json.dumps(audit))
    target=OUT/'summary';target.mkdir(exist_ok=True);save_csv(target/'final_metrics.csv',rows)
    lookup={(r['dataset'],r['angle_count'],r['snr_label_db'],r['method']):r for r in rows}
    # 质量—角度分析另外选已保存的两个预算终点中较好者，明确这是额外GT选择。
    best_rows=[];available_rows=[]
    for r in rows:
        data=records[(r['dataset'],r['angle_count'],r['snr_label_db'],r['method'])]
        use_grid=data['selected']['nrmse']<=data['refined']['nrmse']
        chosen=data['selected'] if use_grid else data['refined']
        best_rows.append({**r,**{k:v for k,v in chosen.items() if k!='history'},
                          'endpoint':'grid' if use_grid else 'refined',
                          'total_iterations':(40 if r['method'].startswith('pnp') else 6000) if use_grid else data['refined']['total_iterations']})
        for candidate in data['candidates']:
            available_rows.append({**{k:v for k,v in candidate.items() if k!='history'},
                                   'total_iterations':40 if r['method'].startswith('pnp') else 6000,'endpoint':'grid'})
        available_rows.append({**r,'endpoint':'refined'})
    save_csv(target/'best_available_metrics.csv',best_rows)
    best_lookup={(r['dataset'],r['angle_count'],r['snr_label_db'],r['method']):r for r in best_rows}
    # 旧15角基准：比较已有传统oracle参数和网络三种子fixed均值，不使用网络GT oracle截点。
    historical=[]
    for r in read_csv(ROOT/'outputs/extension/classical/selected_metrics.csv'):
        if int(r['angle_count'])==15 and int(r['snr_label_db']) in SNRS:
            historical.append({'dataset':r['dataset'],'snr':int(r['snr_label_db']),'method':r['method'],
                               **{k:float(r[k]) for k in ('nrmse','psnr_db','ssim')}})
    for r in read_csv(ROOT/'outputs/improvements/summary/classical_final_metrics.csv'):
        if int(r['angle_count'])==15:
            historical.append({'dataset':r['dataset'],'snr':int(r['snr_label_db']),'method':r['method'],
                               **{k:float(r[k]) for k in ('nrmse','psnr_db','ssim')}})
    for r in read_csv(ROOT/'outputs/extension/dip/aggregate_metrics.csv'):
        if int(r['angle_count'])==15 and r['checkpoint']=='fixed':
            historical.append({'dataset':r['dataset'],'snr':int(r['snr_label_db']),'method':r['method']+'_fixed_mean',
                               **{k:float(r[k+'_mean']) for k in ('nrmse','psnr_db','ssim')}})
    lines=['## 13. 非 DIP 的更少角度实验：小波＋TV 与 PnP-BM3D','',
        '**参照更正：本脚本的历史15角门槛仅为历史分析，不代表达到用户要求的原90角正则化基线。原90角LS+TV／ART+TV的正确比较由audit_fullview_comparison.py生成，见docs/基线实验结果.md第14节。**','',
        '目标是观察图像质量与角度数的权衡。本轮统一比较30、15、10、6、3角，20／30 dB；3角是极端探索条件，不预设图像仍可用。发现SheppLogan64／30dB的10角已超过历史15角质量后，再补测7／8／9角，三个方法使用相同网格和预算。','',
        '### 13.1 协议与限制','',
        '- 非负TV作为公共对照：9个强度候选，各6000步，GT NRMSE选参后续6000步。',
        '- 小波＋TV：db2正交小波、周期边界、3层，只惩罚细节系数。TV强度围绕该条件TV选中参数乘0.3／1／3，小波/TV比例0.1／1／10；9个候选，各6000步，选中后续6000步。这是局部网格，没有证明全局最佳参数。',
        '- PnP-BM3D：以该条件12000步非负TV为起点；rho为0.003／0.03／0.3，sigma为0.01／0.03／0.08。9个候选，各40次ADMM迭代，按GT NRMSE选中后再续120次。主表报告160次的选中结果，没有在160次重搜整个网格。BM3D采用官方LC低复杂度配置、4线程和两阶段去噪，输出限制非负；每步数据子问题CG相对容差1e-6、最多200次。',
        '- 所有新方法使用GT强度范围归一化并用GT选参，属于oracle研究比较；没有监督训练或DIP。PnP需要先运行TV，其成本不能仅按160次迭代计算。去噪步骤不保证对应某个显式凸正则项，有限预算不构成收敛保证。',
        '- 每个角度数均使用floor(arange(k)*90/k)选择完整探测器行块；不同角度数的集合不一定嵌套，因此曲线不必单调。标签SNR沿用原数据；实际子集SNR保存在CSV与JSON。只有两个GT及每条件一个给定噪声观测，没有统计泛化保证。','',
        '- 第13.2节按统一续迭代预算报告；第13.3–13.4节及质量曲线为追求最好质量，另从已保存的网格选中终点和续迭代终点中按GT NRMSE取较好者，并标明步数。这仍是oracle截点选择，不是未知真值自动早停；没有保存或搜索每一步的最佳图像。','',
        '- 对两数据集15／10角、30dB另检查BM3D常规np配置；发现SheppLogan64明显改善后，也应用到该数据集7／8／9角、30dB。rho、sigma沿用LC选中值，没有针对np重新调参；保存40／160两个终点，作为额外配置对照纳入质量门槛搜索。np没有覆盖所有条件，不能宣称全面优于LC。','',
        '### 13.2 完整三项图像指标','',
        '| 数据集 | 角度 | SNR | TV NRMSE | 小波TV NRMSE | PnP NRMSE | TV PSNR | 小波TV PSNR | PnP PSNR | TV SSIM | 小波TV SSIM | PnP SSIM |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    conditions=sorted({(r['dataset'],r['angle_count'],r['snr_label_db']) for r in rows},key=lambda r:(r[0],-r[1],r[2]))
    for name,a,s in conditions:
        group=[lookup[(name,a,s,m)] for m in ('tv','wave','pnp')]
        fields=[f"{r[k]:.4f}" for k in ('nrmse','psnr_db','ssim') for r in group]
        lines.append(f'| {name} | {a} | {s} | '+' | '.join(fields)+' |')
    lines+=['','常规BM3D配置补充检查（同一组LC选中参数，未重新调参）：','',
            '| 数据集 | 角度 | LC 40步NRMSE | np 40步NRMSE | LC 160步NRMSE | np 160步NRMSE | np 40步SSIM | np 160步SSIM |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,a in sorted((key[0],key[1]) for key in records if key[3]=='pnp_np'):
        lc=records[(name,a,30,'pnp')];npdata=records[(name,a,30,'pnp_np')]
        lines.append(f"| {name} | {a} | {lc['selected']['nrmse']:.4f} | {npdata['selected']['nrmse']:.4f} | {lc['refined']['nrmse']:.4f} | {npdata['refined']['nrmse']:.4f} | {npdata['selected']['ssim']:.4f} | {npdata['refined']['ssim']:.4f} |")
    lines+=['','### 13.3 在已有15角质量标准下，可以减少到多少角？','',
        '这里把“接近原有质量”明确为：NRMSE不高于本轮之前该条件15角最优主结果，且SSIM不低于同一基准。在全部保存候选和续迭代终点中搜索，两条件同时满足，取最少测试角度，再取NRMSE最小者；这是额外GT辅助选择，不是临床可接受阈值。历史网络为三种子均值，新传统方法是给定观测的单次结果，不构成同等统计精度的比较。','',
        '| 数据集 | SNR | 原15角基准 | 基准NRMSE | 基准SSIM | 最少测试角度 | 新方法 | 步数 | 新NRMSE | 新SSIM |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    quality=[]
    for name,s in itertools.product(DATASETS,SNRS):
        ref=min((r for r in historical if (r['dataset'],r['snr'])==(name,s)),key=lambda r:r['nrmse'])
        feasible=[r for r in available_rows if (r['dataset'],r['snr_label_db'])==(name,s) and r['nrmse']<=ref['nrmse'] and r['ssim']>=ref['ssim']]
        if feasible:
            best=min(feasible,key=lambda r:(r['angle_count'],r['nrmse']))
            lines.append(f"| {name} | {s} | {ref['method']} | {ref['nrmse']:.4f} | {ref['ssim']:.4f} | {best['angle_count']} | {best['method']} | {best['total_iterations']} | {best['nrmse']:.4f} | {best['ssim']:.4f} |")
        else:
            best=None;lines.append(f"| {name} | {s} | {ref['method']} | {ref['nrmse']:.4f} | {ref['ssim']:.4f} | 未满足 | — | — | — | — |")
        quality.append({'dataset':name,'snr':s,'reference':ref,'smallest_feasible':best})
    lines+=['','### 13.4 角度与NRMSE的非支配结果','',
        '若另一个结果使用不更多的角度且NRMSE不更高，至少一项严格更好，则当前点被支配。以下前沿只考虑角度和NRMSE，SSIM单独列出；不代表三指标同时最优，也不代表未测试角度不存在更优点。','',
        '| 数据集 | SNR | 角度 | 方法 | 步数 | NRMSE | PSNR | SSIM |','| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    fronts=[]
    for name,s in itertools.product(DATASETS,SNRS):
        group=[r for r in best_rows if (r['dataset'],r['snr_label_db'])==(name,s)]
        for r in sorted(group,key=lambda r:(r['angle_count'],r['nrmse'])):
            dominated=any(q['angle_count']<=r['angle_count'] and q['nrmse']<=r['nrmse'] and (q['angle_count']<r['angle_count'] or q['nrmse']<r['nrmse']) for q in group)
            if not dominated:
                fronts.append(r);lines.append(f"| {name} | {s} | {r['angle_count']} | {r['method']} | {r['total_iterations']} | {r['nrmse']:.4f} | {r['psnr_db']:.2f} | {r['ssim']:.4f} |")
    lines+=['','### 13.5 参数与预算敏感性','',
        '| 数据集 | 角度 | SNR | 方法 | p或rho | 小波比例或sigma | 网格终点NRMSE | 续迭代后NRMSE |','| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in sorted(rows,key=lambda r:(r['dataset'],-r['angle_count'],r['snr_label_db'],r['method'])):
        lines.append(f"| {r['dataset']} | {r['angle_count']} | {r['snr_label_db']} | {r['method']} | {r['p']:g} | {r['ratio']:g} | {r['grid_nrmse']:.4f} | {r['nrmse']:.4f} |")
    wins={m:sum(lookup[(n,a,s,m)]['nrmse']<lookup[(n,a,s,'tv')]['nrmse'] for n,a,s in itertools.product(DATASETS,ANGLES,SNRS)) for m in ('wave','pnp')}
    degradation=[r for r in rows if r['nrmse']>r['grid_nrmse']+1e-3]
    lines+=['','### 13.6 总结与核验','',
        f"- 按第13.2节续迭代终点，小波＋TV在{wins['wave']}/20个条件的NRMSE低于非负TV；PnP-BM3D在{wins['pnp']}/20个条件更低。具体推荐应同时看角度、噪声及SSIM。",
        f'- 有{len(degradation)}个选中结果在续迭代后NRMSE上升超过0.001，详见预算表；不能把迭代更多等同于图像更好。',
        f"- 产物核验{audit['assertions']}个断言通过，复算{audit['unique_arrays']}个数组的四项指标；三个代表条件各重跑两次。4线程BM3D的严格重跑检查未达到1e-6相对误差，因此审计状态为{audit['status']}，不能说数值完全复现。最初严格检查失败报告也已保留。记录见[审计文件](../outputs/min_angles/audit_summary.json)。",
        '- “最少角度”限定为本轮测试集合；不声称少角度等价于90角，也未建立实际辐射剂量或临床质量标准。','',
        '![质量与角度权衡](../outputs/min_angles/summary/quality_angles.png)','',
        '![SheppLogan64图像对照](../outputs/min_angles/summary/SheppLogan64_images.png)','',
        '![XCAT128图像对照](../outputs/min_angles/summary/XCAT128_images.png)','']
    lines+=['','### 13.7 多线程重跑变化','',
        'BM3D安装包profiles.py明确提示，线程数不等于1时，聚合中的浮点精度和加法顺序会导致输出差异。本次完整PnP重跑观察到差异；不能据此保证其他条件变化幅度相同。主表仍是保存数组的精确指标，下表单独展示重跑，不将其混为不同噪声样本或算法随机种子的统计。','',
        '| 条件 | 步数 | 保存NRMSE | 重跑1 | 重跑2 | 保存SSIM | 重跑1 SSIM | 重跑2 SSIM | 最大图像相对差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in replay:
        first,second=r['repeats'];old=r['original']
        lines.append(f"| {r['dataset']}／{r['angles']}角／{r['snr']}dB／{r['method']} | {r['iterations']} | {old['nrmse']:.6f} | {first['nrmse']:.6f} | {second['nrmse']:.6f} | {old['ssim']:.6f} | {first['ssim']:.6f} | {second['ssim']:.6f} | {max(first['relative_error'],second['relative_error']):.3e} |")
    (target/'results_fragment.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(target/'tradeoff.json',{'wins_vs_TV':wins,'quality_thresholds':quality,'pareto_front':fronts,'refinement_degradation_over_0_001':len(degradation)})
    os.environ.setdefault('MPLCONFIGDIR',str(target/'matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'tv':'Nonnegative TV','wave':'Wavelet+TV','pnp':'PnP-BM3D'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,name in enumerate(DATASETS):
        for j,s in enumerate(SNRS):
            ax=axes[i,j]
            tested=sorted({r['angle_count'] for r in best_rows if (r['dataset'],r['snr_label_db'])==(name,s)})
            for m in labels:
                values=[best_lookup[(name,a,s,m)]['nrmse'] for a in tested]
                ax.plot(tested,values,'o-',label=labels[m])
            extra=sorted((r for r in best_rows if (r['dataset'],r['snr_label_db'],r['method'])==(name,s,'pnp_np')),key=lambda r:r['angle_count'])
            if extra:ax.plot([r['angle_count'] for r in extra],[r['nrmse'] for r in extra],'x--',label='PnP-BM3D np (checked cases)')
            ax.set_title(f'{name}, {s} dB');ax.set_xlabel('Number of angles');ax.set_ylabel('NRMSE (lower is better)');ax.set_xticks(tested);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(target/'quality_angles.png',dpi=150);plt.close(fig)
    from plot_min_angle_images import render_images
    render_images(best_rows,target)
    print(json.dumps(audit),flush=True)
    print(json.dumps({'wins':wins,'minimum_angles':[{k:v for k,v in r.items() if k!='smallest_feasible'}|{'smallest_feasible':None if r['smallest_feasible'] is None else {k:r['smallest_feasible'][k] for k in ('angle_count','method','nrmse','ssim')}} for r in quality]}),flush=True)
if __name__=='__main__':main()
