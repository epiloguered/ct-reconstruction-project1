"""用原90角LS+TV/ART+TV重新审计少角度结果；只读重建数组，不重新选参求解。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_baselines import ROOT, load_matrix, save_csv
from run_sparse_views import select_angles

OUT = ROOT / 'outputs/min_angles/fullview_audit'


def metrics(x, truth, A, b):
    # 独立按定义计算；不裁剪、不调整亮度、不使用显示后的图像。
    estimate = x.reshape(truth.shape, order='F')
    error = estimate - truth
    return dict(nrmse=float(np.sqrt(np.sum(error**2) / np.sum(truth**2))),
                psnr_db=float(10*np.log10(np.ptp(truth)**2 / np.mean(error**2))),
                ssim=float(structural_similarity(truth, estimate, data_range=np.ptp(truth))),
                relative_residual=float(np.linalg.norm(A@x-b)/np.linalg.norm(b)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = 0
    fingerprints = {}

    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(label)

    cases = []
    for folder in ('outputs/min_angles/tv', 'outputs/min_angles/wave', 'outputs/min_angles/pnp',
                   'outputs/min_angles_boundary/tv', 'outputs/min_angles_boundary/wave',
                   'outputs/min_angles_boundary/pnp', 'outputs/min_angles/profile_check/records',
                   'outputs/min_angles/profile_boundary/records'):
        for path in sorted((ROOT/folder).glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if 'selected' in data:
                cases.append(data)
    check(len(cases) == 76, '76 method/condition records')
    baseline, available, ten, regions, input_rows = [], [], [], [], []
    display = {}
    for name in ('SheppLogan64', 'XCAT128'):
        truth = np.loadtxt(ROOT/'Data'/name/'gt.csv', delimiter=',')
        full = load_matrix(name, ROOT/'outputs/cache')
        saved = list(csv.DictReader((ROOT/'outputs/regularized'/name/'selected_metrics.csv').open(encoding='utf-8')))
        for snr in (20, 30):
            bp = ROOT/'Data'/name/f'b_hat_{snr}db.csv'
            bfull = np.loadtxt(bp, delimiter=',').ravel()
            for path in (bp, ROOT/'Data'/name/'gt.csv'):
                fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            reference = full@truth.reshape(-1, order='F')
            full_snr = float(20*np.log10(np.linalg.norm(reference)/np.linalg.norm(bfull-reference)))
            for row in saved:
                if int(row['snr_label_db']) != snr or row['method'] not in ('LS_TV', 'ART_TV'):
                    continue
                path = ROOT/'outputs/regularized'/name/'reconstructions'/f'{name}_{snr}db_{row["method"]}.npy'
                x = np.load(path)
                m = metrics(x, truth, full, bfull)
                for key, value in m.items():
                    check(np.isclose(value, float(row[key]), atol=1e-10, rtol=1e-9), 'baseline '+key)
                baseline.append(dict(dataset=name, snr=snr, angle_count=90, method=row['method'],
                                     array=str(path.relative_to(ROOT)), iterations=int(row['iterations']), endpoint='fixed', **m))
                if snr == 30:
                    display[name, row['method']] = x.reshape(truth.shape, order='F')
            for data in cases:
                s = data['selected']
                if s['dataset'] != name or int(s['snr_label_db']) != snr:
                    continue
                k = int(s['angle_count'])
                A, b, ids, rows = select_angles(full, bfull, k)
                check(np.array_equal(rows, data['row_ids']), 'same full detector blocks')
                check(np.array_equal(ids, data['angle_ids']), 'same angle IDs')
                check(data['amplitude'] == float(np.ptp(truth)), 'GT amplitude scaling')
                if k == 10 and s['method'] == 'tv':
                    input_rows.append(dict(dataset=name, snr=snr, unknowns=full.shape[1],
                                           full_measurements=full.shape[0], sparse_measurements=A.shape[0],
                                           full_actual_snr_db=full_snr, subset_actual_snr_db=data['actual_subset_snr_db']))
                for endpoint, row in [('grid', r) for r in data['candidates']] + [('refined', data['refined'])]:
                    path = ROOT/row['array']; x = np.load(path)
                    check(x.shape == (truth.size,) and np.isfinite(x).all(), 'finite vector in original resolution')
                    m = metrics(x, truth, A, b)
                    for key, value in m.items():
                        check(np.isclose(value, float(row[key]), atol=1e-10, rtol=1e-9), 'sparse '+key)
                    available.append(dict(dataset=name, snr=snr, angle_count=k, method=s['method'],
                                          endpoint=endpoint, array=row['array'],
                                          iterations=(160 if s['method'].startswith('pnp') else 12000) if endpoint == 'refined'
                                          else (40 if s['method'].startswith('pnp') else 6000), **m))
                if snr == 30 and k == 10 and s['method'] == 'tv':
                    display[name, 'TV10'] = np.load(ROOT/data['refined']['array']).reshape(truth.shape, order='F')
            choices = [r for r in available if r['dataset'] == name and r['snr'] == snr and r['angle_count'] == 10]
            best = min(choices, key=lambda r:r['nrmse'])
            ten.append(best)
            if snr == 30:
                pnp_best = min((r for r in choices if r['method'].startswith('pnp')), key=lambda r:r['nrmse'])
                display[name, 'PnP10'] = np.load(ROOT/pnp_best['array']).reshape(truth.shape, order='F')
                display[name, 'GT'] = truth
                # 固定的诊断区域：真值为零的背景，以及真值相邻像素发生变化的边缘。
                edge = np.zeros(truth.shape, dtype=bool)
                dy = np.abs(np.diff(truth, axis=0)) > 0.05*np.ptp(truth)
                dx = np.abs(np.diff(truth, axis=1)) > 0.05*np.ptp(truth)
                edge[:-1] |= dy; edge[1:] |= dy; edge[:,:-1] |= dx; edge[:,1:] |= dx
                background = np.abs(truth) <= 1e-12
                for method in ('LS_TV', 'ART_TV', 'TV10', 'PnP10'):
                    estimate = display[name, method]; error = (estimate-truth)**2
                    _, smap = structural_similarity(truth, estimate, data_range=np.ptp(truth), full=True)
                    interior = np.zeros(truth.shape, bool); interior[3:-3,3:-3] = True
                    bg = background & interior; fg = ~background & interior
                    regions.append(dict(dataset=name, method=method,
                                        background_fraction=float(background.mean()), edge_fraction=float(edge.mean()),
                                        background_rmse=float(np.sqrt(error[background].mean())),
                                        foreground_rmse=float(np.sqrt(error[~background].mean())),
                                        edge_rmse=float(np.sqrt(error[edge].mean())),
                                        background_ssim_map_mean=float(smap[bg].mean()),
                                        foreground_ssim_map_mean=float(smap[fg].mean())))
    check(len(available) == 704, '704 sparse arrays')
    check(len(baseline) == 8, '8 original 90-angle TV baselines')
    thresholds = []
    for ref in baseline:
        # 同时达到NRMSE、SSIM；同一真值下NRMSE达标等价于PSNR达标。
        choices = [r for r in available if r['dataset'] == ref['dataset'] and r['snr'] == ref['snr']
                   and r['nrmse'] <= ref['nrmse'] and r['ssim'] >= ref['ssim']]
        best = min(choices, key=lambda r:(r['angle_count'], r['nrmse'])) if choices else None
        thresholds.append(dict(reference=ref, least_tested_match=best))
    save_csv(OUT/'comparison.csv', baseline+ten)
    save_csv(OUT/'region_diagnostics.csv', regions)
    save_csv(OUT/'input_comparison.csv', input_rows)
    report = dict(status='passed', assertions=checks, sparse_arrays=704, baseline_arrays=8,
                  input_sha256=fingerprints, thresholds=thresholds,
                  limitation='Saved-array verification, not new optimization or a single-factor controlled trial. Original baseline has no historical input hash manifest.')
    (OUT/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['### 与原90角正则化基线的三指标对照', '',
             '10角列选取该条件全部已保存候选和续迭代终点中NRMSE最小者，包含GT选参、配置及终点选择；不是无真值自动选参的结果。', '',
             '| 数据集 | SNR | 角度／方法 | NRMSE↓ | PSNR↑（dB） | SSIM↑ |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for name in ('SheppLogan64', 'XCAT128'):
        for snr in (20,30):
            for r in [r for r in baseline+ten if r['dataset']==name and r['snr']==snr]:
                lines.append(f'| {name} | {snr} | {r["angle_count"]}角／{r["method"]} | {r["nrmse"]:.6f} | {r["psnr_db"]:.4f} | {r["ssim"]:.6f} |')
    lines += ['', '### 同时达到原90角NRMSE与SSIM的最少已测角度', '',
              '“未达到”仅指本次保存的3–30角结果，没有测试全部31–89角，也没有给误差门槛设置容差。', '',
              '| 数据集 | SNR | 90角基准 | 最少已测角度 | 方法 | NRMSE | SSIM |',
              '| --- | ---: | --- | ---: | --- | ---: | ---: |']
    for t in thresholds:
        r=t['reference']; b=t['least_tested_match']
        tail = f'{b["angle_count"]} | {b["method"]}／{b["endpoint"]} | {b["nrmse"]:.6f} | {b["ssim"]:.6f}' if b else '未达到 | — | — | —'
        lines.append(f'| {r["dataset"]} | {r["snr"]} | {r["method"]} | {tail} |')
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(4,5,figsize=(15,12))
    methods=('GT','LS_TV','ART_TV','TV10','PnP10')
    for i,name in enumerate(('SheppLogan64','XCAT128')):
        gt=display[name,'GT']; scale=np.ptp(gt)
        for j,method in enumerate(methods):
            image=display[name,method]
            axes[2*i,j].imshow(image,cmap='gray',vmin=gt.min(),vmax=gt.max())
            label={'GT':'Ground truth','LS_TV':'90-view LS+TV','ART_TV':'90-view ART+TV','TV10':'10-view NN-TV','PnP10':'10-view PnP-BM3D'}[method]
            n=np.linalg.norm(image-gt)/np.linalg.norm(gt)
            ss=structural_similarity(gt,image,data_range=scale)
            axes[2*i,j].set_title(f'{label}\nNRMSE {n:.4f} / SSIM {ss:.4f}',fontsize=10)
            axes[2*i+1,j].imshow(np.abs(image-gt)/scale,cmap='magma',vmin=0,vmax=.3)
            axes[2*i+1,j].set_title('Absolute error / GT range (0 to 0.3)',fontsize=9)
            axes[2*i,j].axis('off');axes[2*i+1,j].axis('off')
        axes[2*i,0].text(-.12,.5,name+' / 30 dB',transform=axes[2*i,0].transAxes,rotation=90,ha='center',va='center')
    fig.tight_layout();fig.savefig(OUT/'fullview_error_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(status='passed',assertions=checks,sparse_arrays=704,baseline_arrays=8),ensure_ascii=False))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
