"""从既有实验产物生成课程报告表格和科学图像，不重新求解、不覆盖原实验。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/final'
FIG = ROOT/'report/figures'


def read(path):
    return list(csv.DictReader((ROOT/path).open(encoding='utf-8-sig', newline='')))


def save(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    sources=[]; regular=[]
    for name in ('SheppLogan64','XCAT128'):
        for folder in (f'outputs/{name}', f'outputs/regularized/{name}'):
            path=f'{folder}/'+('metrics.csv' if 'regularized' not in folder else 'selected_metrics.csv')
            sources.append(path)
            for r in read(path):
                if 'regularized' not in folder and r['method'] not in ('LSQR','ART_fixed'):
                    continue
                regular.append({**r, 'angle_count':90, 'source_csv':path,
                    'selection':'GT NRMSE parameter selection; fixed iterations' if 'regularized' in folder else 'fixed stopping configuration'})
    assert len(regular)==64
    save('baseline_regularization.csv',regular)
    for source, filename in (
        ('outputs/extension/classical/selected_metrics.csv','sparse_classical.csv'),
        ('outputs/extension/dip/aggregate_metrics.csv','dip_aggregate.csv'),
        ('outputs/improvements/summary/classical_final_metrics.csv','tv_tgv_controls.csv'),
        ('outputs/min_angles/summary/final_metrics.csv','non_dip_fixed.csv'),
        ('outputs/min_angles/fullview_audit/comparison.csv','original90_vs_best10.csv'),
        ('outputs/min_angles/fullview_audit/region_diagnostics.csv','region_diagnostics.csv')):
        sources.append(source);save(filename,[{**r,'source_csv':source} for r in read(source)])
    sources += ['outputs/min_angles/fullview_audit/audit.json','outputs/min_angles/audit_summary.json',
                'outputs/min_angles/environment_runtime.json','docs/实验实施记录.md','docs/基线实验结果.md']
    manifest={'generated_for':'2026-10-11 course report',
              'policy':'Copy existing numerical records; do not reoptimize or replace saved arrays. Fixed endpoints are primary; GT-selected endpoints are separate.',
              'sources':[{ 'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources]}
    (OUT/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    # 基础与正则化可视化都使用GT灰度范围；指标继续来自未经裁剪的数组。
    for figure,methods,titles in (
        ('baseline_images',('GT','LSQR','ART_fixed'),('Ground truth','LSQR / 90 views','ART / 90 views')),
        ('regularization_images',('GT','LS_L2','LS_L1','LS_TV'),('Ground truth','LS + L2','LS + L1','LS + TV'))):
        fig, axes=plt.subplots(2,len(methods),figsize=(10,6.5))
        for i,name in enumerate(('SheppLogan64','XCAT128')):
            gt=np.loadtxt(ROOT/'Data'/name/'gt.csv',delimiter=',')
            for j,method in enumerate(methods):
                if method=='GT':x=gt
                elif method in ('LSQR','ART_fixed'):
                    path=ROOT/'outputs'/name/'reconstructions'/f'{name}_30db_{method}.npy'
                    x=np.load(path).reshape(gt.shape,order='F')
                else:
                    x=np.load(ROOT/'outputs/regularized'/name/'reconstructions'/f'{name}_30db_{method}.npy').reshape(gt.shape,order='F')
                axes[i,j].imshow(x,cmap='gray',vmin=gt.min(),vmax=gt.max());axes[i,j].axis('off')
                axes[i,j].set_title(titles[j],fontsize=13)
            axes[i,0].text(-.12,.5,name,transform=axes[i,0].transAxes,rotation=90,va='center',ha='center')
        fig.tight_layout();fig.savefig(FIG/(figure+'.png'),dpi=210);plt.close(fig)
    fixed=read('outputs/min_angles/summary/final_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7.8))
    for i,name in enumerate(('SheppLogan64','XCAT128')):
        for j,snr in enumerate((20,30)):
            ax=axes[i,j]
            for method,label,color in (('tv','NN-TV','#24608c'),('wave','Wavelet + TV','#66994e'),('pnp','PnP-BM3D LC','#c66b2b')):
                rr=sorted([r for r in fixed if r['dataset']==name and int(r['snr_label_db'])==snr and r['method']==method and int(r['angle_count']) in (3,6,10,15,30)],key=lambda r:int(r['angle_count']))
                ax.plot([int(r['angle_count']) for r in rr],[float(r['nrmse']) for r in rr],'o-',label=label,color=color)
            for method,style,col in (('LS_TV','--','#555555'),('ART_TV',':','#999999')):
                r=next(r for r in regular if r['dataset']==name and int(r['snr_label_db'])==snr and r['method']==method)
                ax.axhline(float(r['nrmse']),ls=style,color=col,label='90-view '+method.replace('_','+'))
            if snr==30:
                rr=sorted([r for r in fixed if r['dataset']==name and int(r['snr_label_db'])==snr and r['method']=='pnp_np'],key=lambda r:int(r['angle_count']))
                ax.plot([int(r['angle_count']) for r in rr],[float(r['nrmse']) for r in rr],'x--',color='#8d4b8a',label='PnP standard profile')
            ax.set_title(f'{name} / {snr} dB');ax.set_xlabel('Number of projection angles');ax.set_ylabel('NRMSE (lower is better)')
            ax.set_xticks([3,6,10,15,30]);ax.grid(alpha=.18)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=10)
    fig.subplots_adjust(bottom=.19,hspace=.45,wspace=.25)
    fig.savefig(FIG/'quality_angles_fixed.png',dpi=200);plt.close(fig)
    # 展示10角高SSIM案例的细节误差，不对图像做后期增强。
    name='SheppLogan64';gt=np.loadtxt(ROOT/'Data'/name/'gt.csv',delimiter=',')
    paths=[None,ROOT/'outputs/regularized'/name/'reconstructions'/f'{name}_30db_LS_TV.npy',
           ROOT/'outputs/min_angles/tv'/f'{name}_10angles_30db_refined.npy',
           ROOT/'outputs/min_angles/profile_check/records'/f'{name}_10angles_30db_160.npy']
    fig,axes=plt.subplots(2,4,figsize=(10,5.4))
    for j,(path,title) in enumerate(zip(paths,('Ground truth','90-view LS+TV','10-view NN-TV','10-view PnP standard'))):
        x=gt if path is None else np.load(path).reshape(gt.shape,order='F')
        axes[0,j].imshow(x,cmap='gray',vmin=gt.min(),vmax=gt.max());axes[0,j].set_title(title,fontsize=12)
        axes[1,j].imshow(np.abs(x-gt)/np.ptp(gt),cmap='magma',vmin=0,vmax=.3)
        axes[1,j].set_title('Absolute error / GT range',fontsize=10)
        for ax in axes[:,j]:ax.axis('off')
    fig.tight_layout();fig.savefig(FIG/'ssim_error_case.png',dpi=220);plt.close(fig)
    print('Report assets ready: 64 core rows, 7 CSV summaries, 4 figures; original experiments unchanged.')


if __name__=='__main__':main()
