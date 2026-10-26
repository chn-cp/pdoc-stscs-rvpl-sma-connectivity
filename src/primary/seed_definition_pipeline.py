#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP 4（终版）—— 仅对已确认的阳性种子做发表级最终分析
  · 只跑 FINAL_SEEDS（默认 R_VPL、R_CL），不再浪费算力在阴性种子上。
  · 5000 次置换；一次跑完【主分析(全被试)】+【敏感性(保留帧>=90)】两套。
  · 每个协变量模型(M0->M4)输出【所有】显著簇（size/peakT/坐标/FWE），供论文结果表。
  · 单尾 p<0.001 定簇（responder>non），Freedman-Lane 置换簇水平 FWE。
说明：跨种子多重比较的“族”仍是最初检验的 8 个种子（Bonferroni 阈值 0.05/8≈0.0063），
      不能因为 6 个阴性就把族缩成 2 个——报告时用 8 作分母。
依赖：nibabel numpy pandas scipy
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats, ndimage

OUTPUT_DIR = './restricted_inputs/MRI/harmonized_output/'
SPM_ROOT   = './restricted_inputs/MRI/SPM_SecondLevel_Analysis/'
QC_CSV     = os.path.join(SPM_ROOT, "subject_master_QC.csv")
GROUP_MASK = os.path.join(OUTPUT_DIR, "group_explicit_mask_90pct.nii.gz")
RESULT_DIR = os.path.join(SPM_ROOT, "PY_final_results")
os.makedirs(RESULT_DIR, exist_ok=True)

FINAL_SEEDS = ["Seed_R_VPL", "Seed_R_CL", "Seed_L_CL"]   # 2个确认阳性 + L_CL(M0=0.0599近界值,5000次精判)
COVAR_MODELS = [
    ("M0_nocovar",       []),
    ("M1_+age",          ["age"]),
    ("M2_+duration",     ["age", "duration_days"]),
    ("M3_+crs",          ["age", "duration_days", "crs_pre"]),
    ("M4_+motion(full)", ["age", "duration_days", "crs_pre", "mean_fd"]),
]
ALL_COVARS = ["age", "duration_days", "crs_pre", "mean_fd"]
SUBJECT_SETS = [("main", 0), ("sensitivity", 90)]   # 主分析 + 敏感性(保留帧>=90)
N_SEEDS_FAMILY = 8            # 跨种子校正的族大小(最初检验的种子数)
CDT_P = 0.001; N_PERM = 5000; FWE_ALPHA = 0.05; RNG_SEED = 2024

rng = np.random.default_rng(RNG_SEED)
struct = np.ones((3, 3, 3))
mimg = nib.load(GROUP_MASK); mask3d = mimg.get_fdata() > 0; affine = mimg.affine


def load_vec(sub, seed):
    p = os.path.join(OUTPUT_DIR, sub, f"{sub}_{seed}_FCz.nii.gz")
    return nib.load(p).get_fdata()[mask3d] if os.path.exists(p) else None


def clusters(tvec, thr):
    vol = np.zeros(mask3d.shape, np.float32); vol[mask3d] = tvec
    lab, n = ndimage.label(vol >= thr, structure=struct)          # 单尾正向
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    return lab, n, sizes, vol


def glm_t(Y, X, XtXinv, c, df, var_c):
    beta = XtXinv @ (X.T @ Y); resid = Y - X @ beta
    sigma2 = (resid ** 2).sum(0) / df
    return (c @ beta) / np.sqrt(sigma2 * var_c + 1e-30)


def run_model(Y, g, meta, covars):
    n = Y.shape[0]
    Zc = [np.ones(n)] + [meta[cv].values.astype(float) - meta[cv].values.astype(float).mean() for cv in covars]
    Z = np.column_stack(Zc); X = np.column_stack([g, Z])
    p = X.shape[1]; df = n - p
    XtXinv = np.linalg.inv(X.T @ X); c = np.zeros(p); c[0] = 1.0
    var_c = float(c @ XtXinv @ c); t_thr = stats.t.ppf(1 - CDT_P, df)
    t_obs = glm_t(Y, X, XtXinv, c, df, var_c)
    lab, nlab, sizes, vol = clusters(t_obs, t_thr)
    Zpinv = np.linalg.pinv(Z); fit_z = Z @ (Zpinv @ Y); Rz = Y - fit_z
    null_max = np.empty(N_PERM)
    for i in range(N_PERM):
        Ystar = fit_z + Rz[rng.permutation(n), :]
        _, npl, sz, _ = clusters(glm_t(Ystar, X, XtXinv, c, df, var_c), t_thr)
        null_max[i] = sz.max() if npl > 0 else 0
    fwec = float(np.percentile(null_max, 95))
    hits = []
    for k in range(1, nlab + 1):
        size = int(sizes[k - 1]); fwe = (1 + np.sum(null_max >= size)) / (N_PERM + 1)
        cl = (lab == k); pk = np.array(np.unravel_index(np.argmax(vol * cl), mask3d.shape))
        pt = float(vol[tuple(pk)]); mm = (affine @ np.append(pk, 1))[:3]
        hits.append(dict(size=size, peak_t=round(pt, 2),
                         x=round(mm[0], 1), y=round(mm[1], 1), z=round(mm[2], 1),
                         fwe=round(float(fwe), 4), df=df))
    hits.sort(key=lambda d: d['fwe'])
    return hits, fwec


qc = pd.read_csv(QC_CSV); qc['Subject'] = qc['Subject'].astype(str).str.strip()
kept0 = qc[qc['exclude_reason'].astype(str).str.strip().isin(["", "nan"])].copy()

rows = []
for set_name, min_ret in SUBJECT_SETS:
    print(f"\n{'#'*70}\n# {set_name}  (保留帧>= {min_ret})\n{'#'*70}")
    for seed in FINAL_SEEDS:
        subs, Y = [], []
        for _, r in kept0.iterrows():
            v = load_vec(r['Subject'], seed)
            if v is None or (not np.all(np.isfinite(v))) or any(pd.isna(r[c]) for c in ALL_COVARS):
                continue
            if min_ret > 0 and ('n_retained' in r.index) and pd.notna(r['n_retained']) and float(r['n_retained']) < min_ret:
                continue
            subs.append(r); Y.append(v)
        meta = pd.DataFrame(subs).reset_index(drop=True)
        Y = np.asarray(Y, np.float64); g = meta['group'].values.astype(float)
        print(f"\n=== {seed}  n={len(meta)} (resp {int(g.sum())}/non {int((g==0).sum())}) ===")
        for mname, covars in COVAR_MODELS:
            hits, fwec = run_model(Y, g, meta, covars)
            sig = [h for h in hits if h['fwe'] < FWE_ALPHA]
            print(f"  {mname:20s} df={hits[0]['df'] if hits else '-'}  FWEc={fwec:.0f}vx  显著簇={len(sig)}")
            for h in sig:
                bonf = "跨8种子Bonf✓" if h['fwe'] < FWE_ALPHA / N_SEEDS_FAMILY else "跨8种子Bonf✗"
                print(f"       size={h['size']:4d} peakT={h['peak_t']:+.2f} [{h['x']:.0f},{h['y']:.0f},{h['z']:.0f}] "
                      f"FWE={h['fwe']}  ({bonf})")
                rows.append(dict(subject_set=set_name, seed=seed, model=mname, n=len(meta),
                                 fwec=round(fwec, 1), **h,
                                 bonf8_sig=(h['fwe'] < FWE_ALPHA / N_SEEDS_FAMILY)))

pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, f"final_significant_clusters_nperm{N_PERM}.csv"),
                          index=False, encoding="utf-8-sig")
print(f"\n结果已存：{RESULT_DIR}")
print(f"判读：主分析与敏感性都显著 -> 稳健；FWE<{FWE_ALPHA/N_SEEDS_FAMILY:.4f} 者跨8种子Bonferroni仍成立。")