"""仅重画已核验图像，不重复求解或重跑昂贵的PnP审计。"""
import os
from pathlib import Path
import numpy as np
from run_baselines import ROOT,DATASETS
from run_min_angles import OUT,ANGLES

def render_images(rows,target):
    os.environ.setdefault('MPLCONFIGDIR',str(target/'matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(r['dataset'],int(r['angle_count']),int(r['snr_label_db']),r['method']):r for r in rows}
    labels={'tv':'Nonnegative TV','wave':'Wavelet+TV','pnp':'PnP-BM3D LC'}
    for name in DATASETS:
        truth=np.loadtxt(ROOT/'Data'/name/'gt.csv',delimiter=',')
        # 显式行间距为两行标题留空间，避免自动布局让相邻行盖住文字。
        fig,axes=plt.subplots(5,4,figsize=(12,16))
        fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.96,wspace=.04,hspace=.28)
        for i,a in enumerate(ANGLES):
            entries=[('Truth',truth)]
            for m in labels:
                chosen=lookup[(name,a,30,m)]
                if m=='pnp' and (name,a,30,'pnp_np') in lookup:
                    chosen=min((chosen,lookup[(name,a,30,'pnp_np')]),key=lambda r:float(r['nrmse']))
                title=('PnP-BM3D np' if chosen['method']=='pnp_np' else labels[m])+f" ({chosen['total_iterations']} steps)"
                entries.append((f'{a} angles, 30dB\n'+title,np.load(ROOT/chosen['array']).reshape(truth.shape,order='F')))
            for ax,(title,image) in zip(axes[i],entries):
                ax.imshow(image,cmap='gray',vmin=truth.min(),vmax=truth.max());ax.set_title(title,fontsize=9,pad=7);ax.axis('off')
        fig.savefig(target/(name+'_images.png'),dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)

if __name__=='__main__':
    import csv
    with (OUT/'summary/best_available_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    render_images(rows,OUT/'summary')
