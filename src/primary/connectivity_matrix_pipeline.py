#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FC 矩阵 步骤1（患者侧）—— 30 节点(22皮层 + 8皮层下) FC 矩阵 + 预后边筛选
· 从患者损伤感知清洗数据取 30 个 ROI 均值时间序列（节点>30%损毁 -> 该被试该节点 NaN）；
· FC = Pearson -> Fisher-z，存每人 30x30 矩阵(损伤节点行/列为 NaN)；
· 逐边(pairwise-complete)：responder/non(Welch t) + 与 ΔCRS(Spearman)，跨边 FDR；
· 重点看 皮层下(VPL/CL...)×皮层 的边是否复现 seed 发现(跨方法收敛)。
输出：每人 FC 矩阵(.npy) + node_info + edge_stats.csv
依赖：nibabel nilearn numpy pandas scipy
"""
import os, inspect
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from nilearn.image import resample_to_img, new_img_like

OUTPUT_DIR = './restricted_inputs/MRI/harmonized_output/'
ROI_DIR    = './restricted_inputs/MRI/ROI/'
SPM_ROOT   = './restricted_inputs/MRI/SPM_SecondLevel_Analysis/'
QC_CSV     = os.path.join(SPM_ROOT, "subject_master_QC.csv")
DELTACRS   = './restricted_inputs/MRI/deltaCRS_imaging_cohort.csv'
GROUP_MASK = os.path.join(OUTPUT_DIR, "group_explicit_mask_90pct.nii.gz")
WORK = './restricted_inputs/MRI/FCmat_work'; MATDIR = os.path.join(WORK, "patient_mats")
os.makedirs(MATDIR, exist_ok=True)
SPHERE_MM, MIN_NODE_VOX, LESION_THRESH = 6, 3, 0.30

CORTICAL = [
 ("Sensorimotor","L.M1",-39,-26,51),("Sensorimotor","R.M1",38,-26,51),("Sensorimotor","SMA",0,-21,51),
 ("Visual","L.V1",-13,-85,6),("Visual","R.V1",8,-82,6),("Visual","L.V4",-30,-89,20),("Visual","R.V4",30,-89,20),
 ("Auditory","L.A1",-62,-30,12),("Auditory","R.A1",59,-27,15),("Auditory","MCC",0,-7,43),
 ("DMN","aMPFC",-1,54,27),("DMN","PCC",0,-52,27),("DMN","L.LatP",-46,-66,30),("DMN","R.LatP",49,-63,33),
 ("ExecuContr","DMPFC",0,27,46),("ExecuContr","L.PFC",-44,45,0),("ExecuContr","R.PFC",44,45,0),
 ("ExecuContr","L.SPC",-50,-51,45),("ExecuContr","R.SPC",50,-51,45),
 ("Salience","L.AIns",-40,18,-12),("Salience","R.AIns",42,10,-12),("Salience","dACC",0,18,30)]
NUCLEI = {"L_CL":["L_CL.nii.gz"],"R_CL":["R_CL.nii.gz"],
          "L_VPL":["L_VPLa.nii.gz","L_VPLp.nii.gz"],"R_VPL":["R_VPLa.nii.gz","R_VPLp.nii.gz"],
          "L_PTg":["L_PTg.nii.gz"],"R_PTg":["R_PTg.nii.gz"],"L_mRt":["L_mRt.nii.gz"],"R_mRt":["R_mRt.nii.gz"]}

_RS = inspect.signature(resample_to_img).parameters
def resample(s, t, i):
    kw = {}
    if 'force_resample' in _RS: kw['force_resample'] = True
    if 'copy_header' in _RS: kw['copy_header'] = True
    return resample_to_img(s, t, interpolation=i, **kw)

ref = nib.load(GROUP_MASK); shape = ref.shape; aff = ref.affine
invA = np.linalg.inv(aff); vs = np.sqrt((aff[:3, :3]**2).sum(0))
gi, gj, gk = np.ogrid[:shape[0], :shape[1], :shape[2]]
parc = np.zeros(shape, np.int32); names, nets = [], []; lab = 0
for net, nm, x, y, z in CORTICAL:
    ci, cj, ck = (invA @ np.array([x, y, z, 1]))[:3]
    m = (((gi-ci)*vs[0])**2 + ((gj-cj)*vs[1])**2 + ((gk-ck)*vs[2])**2 <= SPHERE_MM**2) & (parc == 0)
    if m.sum() >= MIN_NODE_VOX: lab += 1; parc[m] = lab; names.append(nm); nets.append(net)
for nm, files in NUCLEI.items():
    first = nib.load(os.path.join(ROI_DIR, files[0])); comb = np.zeros(first.shape, np.float32)
    for f in files:
        p = os.path.join(ROI_DIR, f)
        if os.path.exists(p): comb += (nib.load(p).get_fdata() > 0.5)
    roi = resample(nib.Nifti1Image((comb > 0).astype(np.float32), first.affine, first.header), ref, 'linear').get_fdata() >= 0.3
    m = roi & (parc == 0)
    if m.sum() >= MIN_NODE_VOX: lab += 1; parc[m] = lab; names.append(nm); nets.append("Subcortical")
N = lab; pf = parc.ravel()
pd.DataFrame({'idx': range(N), 'name': names, 'network': nets}).to_csv(os.path.join(WORK, "node_info.csv"),
                                                                       index=False, encoding='utf-8-sig')
print(f"节点 N={N}: {names}")

qc = pd.read_csv(QC_CSV); qc['Subject'] = qc['Subject'].astype(str).str.strip()
kept = qc[qc['exclude_reason'].astype(str).str.strip().isin(["", "nan"])].copy()
dcm = pd.read_csv(DELTACRS)[['Subject', 'delta_crs']]
info = kept.merge(dcm, on='Subject', how='left')
order = []; MATS = []
for _, r in info.iterrows():
    sub = r['Subject']
    f = os.path.join(OUTPUT_DIR, sub, f"{sub}_space-MNI2mm_desc-cleaned_unsmoothed.nii.gz")
    fl = os.path.join(OUTPUT_DIR, sub, f"{sub}_space-MNI2mm_desc-lesion_mask.nii.gz")
    if not os.path.exists(f): continue
    d = nib.load(f).get_fdata(); T = d.shape[-1]; d2 = d.reshape(-1, T)
    les = nib.load(fl).get_fdata().ravel() > 0.5 if os.path.exists(fl) else np.zeros(pf.shape, bool)
    ts = np.full((N, T), np.nan)
    for k in range(1, N+1):
        node = pf == k; ntot = node.sum()
        if ntot and (node & les).sum()/ntot <= LESION_THRESH and (node & ~les).sum() >= MIN_NODE_VOX:
            ts[k-1] = d2[node & ~les].mean(0)
    Z = np.full((N, N), np.nan)
    good = np.where(~np.isnan(ts).any(1))[0]
    if len(good) > 1:
        tg = ts[good]; tz = (tg - tg.mean(1, keepdims=True)) / (tg.std(1, keepdims=True) + 1e-8)
        z = np.arctanh(np.clip((tz @ tz.T) / T, -0.999, 0.999)); np.fill_diagonal(z, 0)
        Z[np.ix_(good, good)] = z
    np.save(os.path.join(MATDIR, f"{sub}.npy"), Z); order.append(sub); MATS.append(Z)
    print(f"  {sub}: 完好节点 {len(good)}/{N}")
MATS = np.array(MATS)
pd.Series(order).to_csv(os.path.join(WORK, "patient_order.csv"), index=False, header=['Subject'])

grp = np.array([int(info.set_index('Subject').loc[s, 'group']) for s in order])
dc = np.array([float(info.set_index('Subject').loc[s, 'delta_crs']) for s in order])
def fdr(p):
    p = np.asarray(p, float); m = len(p)
    if m == 0: return p
    o = np.argsort(p); q = np.empty(m); pr = 1
    for r in range(m-1, -1, -1):
        i = o[r]; pr = min(pr, p[i]*m/(r+1)); q[i] = pr
    return q
rows = []
for i in range(N):
    for j in range(i+1, N):
        e = MATS[:, i, j]; ok = np.isfinite(e); r1 = ok & (grp == 1); r0 = ok & (grp == 0)
        if ok.sum() < 15 or r1.sum() < 4 or r0.sum() < 4: continue
        _, prn = stats.ttest_ind(e[r1], e[r0], equal_var=False)
        okd = ok & np.isfinite(dc); rho, pdc = stats.spearmanr(e[okd], dc[okd]) if okd.sum() > 5 else (np.nan, np.nan)
        subc = ('Subcortical' in (nets[i], nets[j]))
        rows.append(dict(nodeA=names[i], nodeB=names[j], subcortical_edge=subc, n=int(ok.sum()),
                         resp_mean=round(np.nanmean(e[r1]), 3), non_mean=round(np.nanmean(e[r0]), 3),
                         p_resp_vs_non=prn, rho_deltaCRS=rho, p_deltaCRS=pdc))
res = pd.DataFrame(rows)
res['q_resp'] = fdr(res['p_resp_vs_non'].values); res['q_dcrs'] = fdr(res['p_deltaCRS'].fillna(1).values)
res.to_csv(os.path.join(WORK, "edge_stats.csv"), index=False, encoding='utf-8-sig')
print(f"
共 {len(res)} 条可检验边。FDR q<0.05：")
sig = res[(res['q_resp'] < 0.05) | (res['q_dcrs'] < 0.05)]
print(sig.to_string(index=False) if len(sig) else "（FDR后无；见未校正候选）")
print("
候选边(未校正 p<0.05，双路径第1阶段)：")
cand = res[(res['p_resp_vs_non'] < 0.05) | (res['p_deltaCRS'] < 0.05)].sort_values('p_resp_vs_non')
print(cand[['nodeA', 'nodeB', 'subcortical_edge', 'n', 'p_resp_vs_non', 'rho_deltaCRS', 'p_deltaCRS']].head(25).to_string(index=False))
print(f"
矩阵存 {MATDIR}；edge_stats.csv 已存。候选边下一步进贝叶斯 pd 验证(FCMAT_2)。")
