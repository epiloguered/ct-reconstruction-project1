"""Compare saved classical oracle results with three-seed fixed DIP endpoints."""
import os
import csv
from pathlib import Path
import numpy as np
from run_baselines import ROOT


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_tables(output, classical, neural, individual):
    cases = [(name, angles, snr) for name in ("SheppLogan64", "XCAT128")
             for angles in (30, 15) for snr in (20, 30)]
    c = {(r["dataset"], int(r["angle_count"]), int(r["snr_label_db"]), r["method"]): r for r in classical}
    n = {(r["dataset"], int(r["angle_count"]), int(r["snr_label_db"]), r["method"], r["checkpoint"]): r for r in neural}
    def summary(row, metric):
        return f"{float(row[metric+'_mean']):.4f} ± {float(row[metric+'_std']):.4f}"
    lines = ["## 9. DIP 与 DIP+TV 扩展结果", "",
             "两数据集、30／15 角、20／30 dB、两个网络方法、三个随机种子，共完成 48 次 2000 步拟合。主结果为固定终点；另外报告每 50 步采样的 GT oracle 截点。下表的 ± 为三个种子的**样本标准差**，不是置信区间。传统 TV 参数按 GT NRMSE 选择，而 DIP+TV 的 β 预先固定为 20 dB 下 0.01、30 dB 下 0.001；两类选参规则和可行域不同，应结合这些设置理解对照。", "",
             "### 9.1 固定第 2000 步的 NRMSE", "",
             "| 数据集 | 角度数 | SNR | LS+TV（oracle参数） | ART+TV（oracle参数） | DIP（均值±std） | DIP+TV（均值±std） | DIP+TV相对LS+TV的NRMSE降低 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    improvements = []
    for name,a,s in cases:
        tv = float(c[(name,a,s,"LS_TV")]["nrmse"])
        ai = float(n[(name,a,s,"DIP_TV","fixed")]["nrmse_mean"])
        pct = 100*(tv-ai)/tv
        improvements.append({"dataset":name,"angle_count":a,"snr_label_db":s,"percent":pct})
        lines.append(f"| {name} | {a} | {s} dB | {tv:.4f} | {float(c[(name,a,s,'ART_TV')]['nrmse']):.4f} | {summary(n[(name,a,s,'DIP','fixed')],'nrmse')} | {summary(n[(name,a,s,'DIP_TV','fixed')],'nrmse')} | {pct:+.1f}% |")
    lines += ["", "最后一列为 `(LS_TV NRMSE − DIP_TV平均NRMSE)/LS_TV NRMSE`；负值表示 DIP+TV 误差更高。表格没有按最佳种子选取成绩。", "",
              "### 9.2 同一终点的 SSIM", "",
              "| 数据集 | 角度数 | SNR | LS+TV | ART+TV | DIP（均值±std） | DIP+TV（均值±std） |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,a,s in cases:
        lines.append(f"| {name} | {a} | {s} dB | {float(c[(name,a,s,'LS_TV')]['ssim']):.4f} | {float(c[(name,a,s,'ART_TV')]['ssim']):.4f} | {summary(n[(name,a,s,'DIP','fixed')],'ssim')} | {summary(n[(name,a,s,'DIP_TV','fixed')],'ssim')} |")
    lines += ["", "PSNR、相对投影残差、每个种子的结果及耗时均保存在 `outputs/extension/dip/individual_metrics.csv` 和 `aggregate_metrics.csv`。PSNR 的汇总是各次 PSNR 的均值，不是把平均 MSE 再转换成 PSNR。", "",
              "### 9.3 真值辅助的 oracle 截点诊断", "",
              "本表的 GT 最小 NRMSE 截点不能直接用于未知真实图像的自动早停；它与同一次固定预算轨迹对应，并非另一次训练。步数范围为三个种子的最优采样截点最小／最大值。", "",
              "| 数据集 | 角度数 | SNR | DIP oracle NRMSE | DIP截点范围 | DIP+TV oracle NRMSE | DIP+TV截点范围 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,a,s in cases:
        steps = {}
        for method in ("DIP","DIP_TV"):
            chosen = [int(r["iteration"]) for r in individual if r["dataset"]==name and int(r["angle_count"])==a and int(r["snr_label_db"])==s and r["method"]==method and r["checkpoint"]=="oracle"]
            steps[method] = f"{min(chosen)}–{max(chosen)}"
        lines.append(f"| {name} | {a} | {s} dB | {summary(n[(name,a,s,'DIP','oracle')],'nrmse')} | {steps['DIP']} | {summary(n[(name,a,s,'DIP_TV','oracle')],'nrmse')} | {steps['DIP_TV']} |")
    fixed_rows=[r for r in individual if r["checkpoint"]=="fixed"]
    oracle_rows=[r for r in individual if r["checkpoint"]=="oracle"]
    elapsed=[float(r["runtime_seconds"]) for r in fixed_rows]
    counts={m:sum(int(r["iteration"])==2000 for r in oracle_rows if r["method"]==m) for m in ("DIP","DIP_TV")}
    lines += ["", f"48 次拟合的累计训练与评价计时为 **{sum(elapsed):.2f} 秒**，单次 {min(elapsed):.2f}–{max(elapsed):.2f} 秒，不含网络初始化、矩阵构造与保存绘图；全部使用 RTX 5070。DIP 的 24 次训练中 {counts['DIP']} 次最优采样截点在预算终点，DIP+TV 为 {counts['DIP_TV']} 次。落在终点的记录说明当前预算内尚未观察到更优的早期截点，不能证明继续训练一定改善，也不能宣称网络优化已严格收敛。", "",
              "### 9.4 共同条件对照图", "",
              "柱状图使用全部三种子的均值及样本标准差；图像示例预先固定展示 seed 0，不选最优种子。所有显示图共用本数据集的 GT 灰度范围，指标使用原始向量。", "",
              "![SheppLogan64传统TV与DIP的指标对照](../outputs/extension/summary/SheppLogan64_classical_vs_DIP.png)", "",
              "![XCAT128传统TV与DIP的指标对照](../outputs/extension/summary/XCAT128_classical_vs_DIP.png)", "",
              "[SheppLogan64全部共同条件图像](../outputs/extension/summary/SheppLogan64_shared_cases.png)；[XCAT128全部共同条件图像](../outputs/extension/summary/XCAT128_shared_cases.png)。", "",
              "[SheppLogan64三种子误差轨迹](../outputs/extension/dip/figures/SheppLogan64_DIP_history.png)；[XCAT128三种子误差轨迹](../outputs/extension/dip/figures/XCAT128_DIP_history.png)。", ""]
    (output / "dip_results_fragment.md").write_text("\n".join(lines),encoding="utf-8")
    import json
    (output / "comparison_summary.json").write_text(json.dumps({"DIP_TV_percent_nrmse_reduction_vs_LS_TV":improvements,"fit_seconds_total":sum(elapsed),"oracle_at_budget_counts":counts},indent=2),encoding="utf-8")


def main():
    output = ROOT / "outputs" / "extension" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classical = read_csv(ROOT / "outputs" / "extension" / "classical" / "selected_metrics.csv")
    neural = read_csv(ROOT / "outputs" / "extension" / "dip" / "aggregate_metrics.csv")
    individual = read_csv(ROOT / "outputs" / "extension" / "dip" / "individual_metrics.csv")
    if len(neural) != 32 or any(int(r["n_seeds"]) != 3 for r in neural):
        raise ValueError("The full three-seed experiment must be complete before plotting")
    write_tables(output, classical, neural, individual)
    cases = [(30, 20), (30, 30), (15, 20), (15, 30)]
    c_lookup = {(r["dataset"], int(r["angle_count"]), int(r["snr_label_db"]), r["method"]): r for r in classical}
    n_lookup = {(r["dataset"], int(r["angle_count"]), int(r["snr_label_db"]), r["method"], r["checkpoint"]): r for r in neural}
    labels = ("LS+TV (oracle)", "ART+TV (oracle)", "DIP (fixed)", "DIP+TV (fixed)")
    methods = ("LS_TV", "ART_TV", "DIP", "DIP_TV")
    colors = ("#486f9f", "#78a6c4", "#e09f58", "#3d937c")
    for name in ("SheppLogan64", "XCAT128"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        x_axis = np.arange(len(cases))
        width = 0.19
        for axis, metric in zip(axes, ("nrmse", "ssim")):
            for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
                means, deviations = [], []
                for angles, snr in cases:
                    if i < 2:
                        means.append(float(c_lookup[(name, angles, snr, method)][metric]))
                        deviations.append(0)
                    else:
                        row = n_lookup[(name, angles, snr, method, "fixed")]
                        means.append(float(row[f"{metric}_mean"]))
                        deviations.append(float(row[f"{metric}_std"]))
                axis.bar(x_axis + (i - 1.5) * width, means, width, yerr=deviations,
                         capsize=3, color=color, label=label)
            axis.set_xticks(x_axis, [f"{a} views\n{s} dB" for a,s in cases])
            axis.set_ylabel("NRMSE (lower is better)" if metric == "nrmse" else "SSIM (higher is better)")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{name}: classical truth-tuned priors vs DIP 2000-step endpoints\nError bars: sample standard deviation of all three DIP seeds", fontsize=11)
        fig.savefig(output / f"{name}_classical_vs_DIP.png", dpi=180)
        fig.savefig(output / f"{name}_classical_vs_DIP.pdf")
        plt.close(fig)
        truth = np.loadtxt(ROOT / "Data" / name / "gt.csv", delimiter=",")
        fig, axes = plt.subplots(4, 5, figsize=(14, 11), layout="constrained")
        for row_index, (angles, snr) in enumerate(cases):
            panels = [("Truth", truth, None)]
            for method in ("LS_TV", "ART_TV"):
                row = c_lookup[(name, angles, snr, method)]
                estimate = np.load(ROOT / "outputs" / "extension" / "classical" / row["selected_array"])
                panels.append((method + " (oracle)", estimate.reshape(truth.shape, order="F"), row))
            for method in ("DIP", "DIP_TV"):
                row = next(r for r in individual if r["dataset"] == name and int(r["angle_count"]) == angles and int(r["snr_label_db"]) == snr and r["method"] == method and r["checkpoint"] == "fixed" and int(r["seed"]) == 0)
                estimate = np.load(ROOT / row["reconstruction_path"])
                panels.append((method + " (seed 0, fixed)", estimate.reshape(truth.shape, order="F"), row))
            for col, (title, estimate, row) in enumerate(panels):
                axis = axes[row_index, col]
                axis.imshow(estimate, cmap="gray", vmin=truth.min(), vmax=truth.max())
                subtitle = f"{angles} views | {snr} dB" if row is None else f"N={float(row['nrmse']):.3f}, S={float(row['ssim']):.3f}"
                axis.set_title(f"{title}\n{subtitle}", fontsize=9)
                axis.axis("off")
        fig.suptitle(f"{name}: identical display ranges; illustration uses preselected seed 0", fontsize=12)
        fig.savefig(output / f"{name}_shared_cases.png", dpi=160)
        plt.close(fig)
    print(f"[summary plots] {output}")


if __name__ == "__main__":
    main()
