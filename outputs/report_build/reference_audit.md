# CT 课程实验报告：指标与方法引用核查

核查日期：2026-10-11。范围是本地源码、已有实施与结果文档、相关论文原文/作者或机构官方页面、实现库官方文档；未修改主报告、未运行新重建实验。条目结构见同目录 `references.json`。

**编号约定：本核查文件的 [n] 是 JSON 的 `id`。主报告已有编排使用 `report_number` 映射，不要直接混用。** `key` 是稳定关键词，可用于脚本匹配。15 篇与当前方法/指标直接相关的论文另附 3 篇可选历史或体模背景，以及 5 项实现文档；最终正文只保留实际引用项。

## 1. 三项图像指标的准确表述

实际入口为 `run_baselines.py:image_metrics()`，各阶段共用。设真值为 g，重建为 x̂，像素数为 N，R=max(g)−min(g)，并要求 ||g||F>0、R>0。

- **NRMSE**：`||x̂−g||F / ||g||F`。这是本项目明确采用的常见相对欧氏/Frobenius 范数归一化；不是某篇论文独创，亦不是文献唯一的 NRMSE 定义。源码用 NumPy 直接计算，与官方文档的 euclidean 形式相同 [19]。不能写成按真值强度范围归一化的 RMSE。
- **PSNR**：`10 log10(R²/MSE)`，其中 `MSE=||x̂−g||F²/N`。MSE/PSNR 关系可引 [16]；把动态范围固定为 **GTmax−GTmin** 是本项目协议，不能归给原文。不同方法沿用同一真值范围；指标函数不裁剪重建、不按重建自身范围重新缩放。零 MSE 在代码中返回正无穷。
- **SSIM**：概念出处为 Wang 等 [15]；实现是 `structural_similarity(truth, estimate, data_range=R)` [19]。本地 `.conda` 实际版本为 **scikit-image 0.26.0**，已读取函数签名与源码。

SSIM 必须披露的差别如下：

| 设置 | 项目实际默认值 | Wang2004 §III-C 的示例设置 |
|---|---|---|
| 局部窗口 | 7×7，均匀权重 | 11×11，高斯权重，sigma=1.5 |
| 协方差归一化 | 样本协方差，N−1 | 式(14)–(16)的归一化高斯加权统计；库匹配选项为总体协方差 |
| K1、K2 | 0.01、0.03 | 0.01、0.03 |
| 强度范围 | 显式 GTmax−GTmin | 原文 L 为允许像素强度的动态范围 |

官方文档给出的匹配参数是 `gaussian_weights=True, sigma=1.5, use_sample_covariance=False` 并显式传 `data_range` [19]。这些参数**尚未用于历史结果**；本次仅准确描述现状，不能把已有 SSIM 表写成按上述匹配设置重算过。

推荐正文句式：

> 本实验将相对真值的 Frobenius 范数误差定义为 NRMSE，并以其选参；PSNR 采用真值最大值与最小值之差作为动态范围，公式依据 MSE/PSNR 的通常关系 [16]。SSIM 概念参考 Wang 等 [15]，实际计算采用 scikit-image 0.26.0 的默认均匀窗口实现 [19]，其窗口与协方差设置不同于原文示例。PSNR 与 SSIM 均对同一选中结果计算。

同一真值下有：

`PSNR = −20 log10(NRMSE) + 20 log10(R√N / ||g||F)`。

因此 NRMSE 与 PSNR 排名必然相反，不能当作两项独立的图像质量证据；SSIM 补充局部结构差异，也不能证明全部小结构已准确恢复。这里前者是由项目公式推导的数学事实，后者属于评价边界。


## 2. 方法出处与本项目实现的对应关系

| 项目位置 | 论文/实现引用 | 可支持的陈述 | 必须注明的项目选择或边界 |
|---|---|---|---|
| `run_baselines.py:run_dataset()` | LSQR [1]；SciPy [21] | 用 LSQR 求稀疏最小二乘 | 调用 SciPy；报告容差、迭代上限和停止码；数值停止不等于图像准确 |
| `cyclic_art()`、`art_with_prior()` | ART [2]；历史渊源可补 [3] | 逐行 Kaczmarz 型投影重建 | 循环顺序、非负投影、零行跳过；ART加先验是项目组合流程，不能称精确求解正则化LS |
| `run_regularization.py:proximal()` | TV/ROF [4]；Chambolle [5]；库 [20] | TV图像先验与TV去噪近端 | 实际 `eps=1e-4,max_num_iter=80`，近端数值近似 |
| `run_regularization.py:fista()` | FISTA [6] | 梯度、近端与动量外推 | 数值估计L、有限预算、近似TV近端；不能无条件套用理论速率 |
| `run_sparse_views.py:select_angles()` | 少视角TV背景 [7] | TV用于不充分CT投影的动机 | 项目抽取课程A/b完整行块；几何及求解器不同；非原论文算法/成绩复现 |
| `run_dip_extension.py:PriorNet/fit_case()` | DIP [8] | 固定随机输入及随机网络结构先验 | 自建四尺度网络、CT投影损失、softplus、TV项及2000步预算是项目适配；训练与尺度不使用GT |
| `run_sparse_improvements.py:primal_dual()` | TGV [9]；Condat [11]；CP [10]作背景 | 二阶TGV与原始—对偶分裂 | 实际显式计算光滑数据梯度，更直接对应Condat的Algorithm3.1；不要写成逐式复现CP原算法 |
| `run_min_angles.py:Wavelet/wavelet_tv()` | 小波收缩相关理论 [12]；Condat [11]；接口 [22] | 小波细节稀疏先验 | db2、3层、周期边界、不罚最低频、TV和非负联合目标均为项目选择；不是VisuShrink复现 |
| `run_min_angles.py:pnp_admm()` | PnP [13]、BM3D [14]、包 [23] | ADMM数据/先验分离，以BM3D更新先验 | CG数据更新、TV起点、rho/sigma独立网格和非负输出；不保证显式凸目标或任意去噪器收敛 |

**Condat 的核对证据**：作者 PDF Algorithm 3.1 式(9)采用“原始近端 + 光滑项梯度 → 对外推原始变量作对偶近端”的顺序；源码 `new_u`、`2*new_u-u`、`project_ball` 与该结构对应 [11]。论文 §3.1.1 说明在没有光滑项时退化为 Chambolle–Pock 类算法；本项目有二次光滑数据项。这一对应属于源码与论文结构核对，不意味着已验证论文定理的全部数值条件。

**BM3D 的核对证据**：本项目实际安装 `bm3d==4.0.3`，官方说明该包同时基于 Dabov 等2007与 Mäkinen/Azzari/Foi2020改进，并通过 bm4d 提供接口 [23]。报告可用 [14]说明原始思想、[23]说明实际实现；不要称直接运行2007原版程序。LC/常规 profile、4线程和非负输出需保留。关于多线程重跑变化，证据应来自本地 `bm3d/profiles.py` 与已有审计日志，不能归给2007论文。

**体模来源边界**：[17]、[18]只能用来解释 Shepp–Logan、XCAT 的名称与历史。本项目文件没有足以确认生成脚本、切片/参数和直接提供者的溯源信息；只能写“课程提供的 SheppLogan64/XCAT128 数据”。不得声称这些CSV由论文原作者提供、与原刊体模完全一致，或已复现4D XCAT仿真流程。

## 3. 主报告编号映射

| 正文编号 report_number | 核查 id | key | 简记 |
|---|---|---|---|
| 1 | 1 | `lsqr1982` | Paige 1982 |
| 2 | 2 | `art1970` | Gordon 1970 |
| 3 | 4 | `rof1992` | Rudin 1992 |
| 4 | 6 | `fista2009` | Beck 2009 |
| 5 | 16 | `wang_bovik2009` | Wang 2009 |
| 6 | 15 | `ssim2004` | Wang 2004 |
| 7 | 7 | `sidky2006` | Sidky 2006 |
| 8 | 13 | `pnp2013` | Venkatakrishnan 2013 |
| 9 | 14 | `bm3d2007` | Dabov 2007 |
| 10 | 8 | `dip2018` | Ulyanov 2018 |
| 11 | 9 | `tgv2010` | Bredies 2010 |
| 12 | 12 | `donoho1995` | Donoho 1995 |
| 13 | 11 | `condat2013` | Condat 2013 |
| 14 | 19 | `skimage_metrics026` | developers 2026 |
| 15 | 23 | `bm3d403` | Mäkinen 2024 |
| 16 | 5 | `chambolle2004` | Chambolle 2004 |

Chambolle–Pock2011条目 `chambolle_pock2011`、Kaczmarz1937、两项体模背景及额外接口文档暂不占用主报告编号。若正文展开它们，应重新生成连续编号并同步全文。

## 4. 书目信息与可追溯链接

- [1] Christopher C. Paige; Michael A. Saunders. *LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares*. ACM Transactions on Mathematical Software, 8(1):43–71, 1982. DOI: [10.1145/355984.355989](https://doi.org/10.1145/355984.355989). [核查来源](https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf).

- [2] Richard Gordon; Robert Bender; Gábor T. Herman. *Algebraic reconstruction techniques (ART) for three-dimensional electron microscopy and x-ray photography*. Journal of Theoretical Biology, 29(3):471–481, 1970. DOI: [10.1016/0022-5193(70)90109-8](https://doi.org/10.1016/0022-5193(70)90109-8). [核查来源](https://doi.org/10.1016/0022-5193(70)90109-8).

- [3] Stefan Kaczmarz. *Angenäherte Auflösung von Systemen linearer Gleichungen*. Bulletin International de l’Académie Polonaise des Sciences et des Lettres, Classe des Sciences Mathématiques et Naturelles, Série A:355–357, 1937. [核查来源](https://faculty.sites.iastate.edu/esweber/files/inline-files/kaczmarz_english_translation_1937.pdf).

- [4] Leonid I. Rudin; Stanley Osher; Emad Fatemi. *Nonlinear total variation based noise removal algorithms*. Physica D: Nonlinear Phenomena, 60(1–4):259–268, 1992. DOI: [10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F). [核查来源](https://www.sciencedirect.com/science/article/pii/016727899290242F).

- [5] Antonin Chambolle. *An Algorithm for Total Variation Minimization and Applications*. Journal of Mathematical Imaging and Vision, 20:89–97, 2004. DOI: [10.1023/B:JMIV.0000011325.36760.1e](https://doi.org/10.1023/B:JMIV.0000011325.36760.1e). [核查来源](https://link.springer.com/article/10.1023/B:JMIV.0000011325.36760.1e).

- [6] Amir Beck; Marc Teboulle. *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*. SIAM Journal on Imaging Sciences, 2(1):183–202, 2009. DOI: [10.1137/080716542](https://doi.org/10.1137/080716542). [核查来源](https://epubs.siam.org/doi/10.1137/080716542).

- [7] Emil Y. Sidky; Chien-Min Kao; Xiaochuan Pan. *Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT*. Journal of X-Ray Science and Technology, 14(2):119–139, 2006. DOI: [10.3233/XST-2006-00155](https://doi.org/10.3233/XST-2006-00155). [核查来源](https://journals.sagepub.com/doi/10.3233/XST-2006-00155).

- [8] Dmitry Ulyanov; Andrea Vedaldi; Victor Lempitsky. *Deep Image Prior*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR):9446–9454, 2018. [核查来源](https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html).

- [9] Kristian Bredies; Karl Kunisch; Thomas Pock. *Total Generalized Variation*. SIAM Journal on Imaging Sciences, 3(3):492–526, 2010. DOI: [10.1137/090769521](https://doi.org/10.1137/090769521). [核查来源](https://epubs.siam.org/doi/10.1137/090769521).

- [10] Antonin Chambolle; Thomas Pock. *A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging*. Journal of Mathematical Imaging and Vision, 40:120–145, 2011. DOI: [10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1). [核查来源](https://link.springer.com/article/10.1007/s10851-010-0251-1).

- [11] Laurent Condat. *A Primal–Dual Splitting Method for Convex Optimization Involving Lipschitzian, Proximable and Linear Composite Terms*. Journal of Optimization Theory and Applications, 158(2):460–479, 2013. DOI: [10.1007/s10957-012-0245-9](https://doi.org/10.1007/s10957-012-0245-9). [核查来源](https://lcondat.github.io/publis/Condat-optim-JOTA-2013.pdf).

- [12] David L. Donoho. *De-Noising by Soft-Thresholding*. IEEE Transactions on Information Theory, 41(3):613–627, 1995. DOI: [10.1109/18.382009](https://doi.org/10.1109/18.382009). [核查来源](https://doi.org/10.1109/18.382009).

- [13] Singanallur V. Venkatakrishnan; Charles A. Bouman; Brendt Wohlberg. *Plug-and-Play Priors for Model Based Reconstruction*. 2013 IEEE Global Conference on Signal and Information Processing (GlobalSIP):945–948, 2013. DOI: [10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048). [核查来源](https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/GlobalSIP2013a.pdf).

- [14] Kostadin Dabov; Alessandro Foi; Vladimir Katkovnik; Karen Egiazarian. *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering*. IEEE Transactions on Image Processing, 16(8):2080–2095, 2007. DOI: [10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238). [核查来源](https://webpages.tuni.fi/foi/GCF-BM3D/index.html).

- [15] Zhou Wang; Alan C. Bovik; Hamid R. Sheikh; Eero P. Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE Transactions on Image Processing, 13(4):600–612, 2004. DOI: [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861). [核查来源](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf).

- [16] Zhou Wang; Alan C. Bovik. *Mean Squared Error: Love It or Leave It? A New Look at Signal Fidelity Measures*. IEEE Signal Processing Magazine, 26(1):98–117, 2009. DOI: [10.1109/MSP.2008.930649](https://doi.org/10.1109/MSP.2008.930649). [核查来源](https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf).

- [17] Lawrence A. Shepp; Benjamin F. Logan. *The Fourier Reconstruction of a Head Section*. IEEE Transactions on Nuclear Science, 21(3):21–43, 1974. DOI: [10.1109/TNS.1974.6499235](https://doi.org/10.1109/TNS.1974.6499235). [核查来源](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fourier-reconstruction-of-a-head-section.pdf).

- [18] W. P. Segars; G. Sturgeon; S. Mendonca; Jason Grimes; B. M. W. Tsui. *4D XCAT Phantom for Multimodality Imaging Research*. Medical Physics, 37(9):4902–4915, 2010. DOI: [10.1118/1.3480985](https://doi.org/10.1118/1.3480985). [核查来源](https://pmc.ncbi.nlm.nih.gov/articles/PMC2941518/).

- [19] scikit-image developers. *skimage.metrics API documentation, version 0.26*. Official software documentation, 2026. [核查来源](https://scikit-image.org/docs/0.26.x/api/skimage.metrics.html).

- [20] scikit-image developers. *skimage.restoration.denoise_tv_chambolle API documentation, version 0.26*. Official software documentation, 2026. [核查来源](https://scikit-image.org/docs/0.26.x/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle).

- [21] SciPy developers. *scipy.sparse.linalg.lsqr API documentation*. Official software documentation, 2026. [核查来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html).

- [22] PyWavelets developers. *Handling DWT Coefficients*. Official software documentation, 2026. [核查来源](https://pywavelets.readthedocs.io/en/latest/ref/dwt-coefficient-handling.html).

- [23] Ymir Mäkinen; Lucio Azzari; Alessandro Foi. *bm3d 4.0.3: Python wrapper for BM3D denoising*. Official PyPI package description, 2024. [核查来源](https://pypi.org/project/bm3d/4.0.3/).

## 5. 已核查与未核查

已核查：指标入口公式；本地SSIM版本、默认窗口/协方差及源码；主要算法调用和新原始—对偶更新结构；论文作者、标题、年份、刊物/会议、页码、DOI或官方链接；官方BM3D包说明。正文出处只采用论文原文、作者/机构官方页面、出版社页面和官方实现文档。

获取限制：ART1970出版社正文返回403，书目信息经官方PubMed核对；LSQR的ACM页返回403，但Stanford作者维护的原文PDF可读取；Donoho1995的IEEE页要求浏览器验证，原文PDF已从大学托管副本读取。Kaczmarz1937链接为大学托管的译文与原文扫描合集，原刊DOI未查得。没有据这些访问限制虚构原文细节。

未做：历史指标重算、算法数值复现、追加实验、数据生成来源确认。引用正确只能支持方法来源与解释，不能替代本项目产物、参数协议和验证记录。

