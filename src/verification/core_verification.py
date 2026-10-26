from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

OUT = Path(os.environ.get('RVPL_SMA_VERIFICATION_OUTPUT', Path.cwd() / 'outputs' / 'verification'))
ROOT = Path(os.environ.get('RVPL_SMA_PROJECT_ROOT', Path.cwd()))
LOCAL = ROOT / 'restricted_inputs' / 'archive'
LEGACY = ROOT / 'restricted_inputs' / 'legacy_analysis'
HIST = LEGACY / 'focused_edge_validation'
HARM = LOCAL / 'harmonized_output/harmonized_output'
PUBLIC_PRIMARY = Path(__file__).resolve().parents[1] / 'primary'
SEED = 20261020
N_PERM = 5000
FEATURE = 'R_VPL_SMA_z_local26_z'
for name in ['00_inputs', '01_ordered_state', '02_spatial_convergence',
             '03_anatomical_competition', '04_longitudinal_optional',
             '05_master_results', 'figures', 'tables', 'logs', 'scripts']:
    (OUT / name).mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
BLUE, DARK, GREY = '#2166AC', '#263441', '#858C94'


def emit(message):
    print(message, flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                   default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')


def savefig(fig, name):
    fig.savefig(OUT / 'figures' / (name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT / 'figures' / (name+'.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


def read_csv_encoding(path):
    for enc in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeError:
            pass
    raise RuntimeError(f'Unknown CSV encoding: {path}')


def prepare_inputs():
    records = []
    def copy_file(src, target):
        target = OUT / '00_inputs' / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        records.append({'source': str(src), 'snapshot': str(target.relative_to(OUT)),
                        'sha256': sha(target), 'bytes': target.stat().st_size})
    for f in HIST.glob('*'):
        if f.is_file() and f.suffix in {'.csv', '.json', '.md', '.sha256'}:
            copy_file(f, 'historical/'+f.name)
    for f in ['clinical_models.py', 'focused_edge_validation.py',
              'seed_definition_pipeline.py', 'connectivity_matrix_pipeline.py']:
        copy_file(PUBLIC_PRIMARY/f, 'source_scripts/'+f)
    for f in ['focused_edge_targeted_sensitivity.csv', 'thalamic_seed_family_unified_p.csv',
              'cortical_target_family_unified_p.csv', 'clinical_baseline_nested_cv_predictions.csv',
              'clinical_baseline_nested_cv_repeats.csv']:
        copy_file(ROOT/'work/peer_review_audit/results'/f, 'peer_review/'+f)
    for zipname, predicate in [
        ('FCmat_work.zip', lambda n: '/patient_mats/' in n or '/HC_mats/' in n or n.endswith(('patient_order.csv','HC_order.csv','node_info.csv'))),
        ('SPM_SecondLevel_Analysis.zip', lambda n: '/PY_final_results/' in n or n.endswith('subject_master_QC.csv')),
    ]:
        zpath = LOCAL/zipname
        with ZipFile(zpath) as z:
            for item in z.infolist():
                if item.is_dir() or not predicate(item.filename):
                    continue
                dst = OUT/'00_inputs'/item.filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(item.filename))
                records.append({'source': str(zpath)+'::'+item.filename,
                                'snapshot': str(dst.relative_to(OUT)), 'sha256': sha(dst), 'bytes': item.file_size})
    clinical = read_csv_encoding(LOCAL/'information.csv')
    # Explicit deidentification whitelist. No names, clinical IDs, dates or free text.
    cols = ['Subject','age','etiology_group','days_to_surgery','days_to_fmri',
            'CRS_pre','CRS_1mo','CRS_3mo','CRS_6mo',
            'diagnosis_pre','diagnosis_1mo','diagnosis_3mo','diagnosis_6mo','died']
    clinical = clinical[cols].copy()
    clinical.to_csv(OUT/'00_inputs/clinical_followup_deidentified.csv', index=False)
    records.append({'source': str(LOCAL/'information.csv')+' [deidentified whitelist]',
                    'snapshot': '00_inputs/clinical_followup_deidentified.csv',
                    'sha256': sha(OUT/'00_inputs/clinical_followup_deidentified.csv')})
    lesion = pd.read_csv(LEGACY/'outputs/reanalysis/results/core_patient_level.csv')
    lesion[['Subject','lesion_vol_ml']].to_csv(OUT/'00_inputs/lesion_volume_deidentified.csv', index=False)
    records.append({'source': str(LEGACY/'outputs/reanalysis/results/core_patient_level.csv')+' [whitelist]',
                    'snapshot':'00_inputs/lesion_volume_deidentified.csv',
                    'sha256':sha(OUT/'00_inputs/lesion_volume_deidentified.csv')})
    save_json(OUT/'00_inputs/INPUT_MANIFEST.json', records)
    return clinical


def cohort(clinical):
    d = pd.read_csv(OUT/'00_inputs/historical/RVPL_SMA_analysis_dataset_deidentified.csv')
    mats = OUT/'00_inputs/FCmat_work'
    nodes = pd.read_csv(mats/'node_info.csv').name.tolist()
    i,j = nodes.index('R_VPL'),nodes.index('SMA')
    patients = {p.stem:np.load(p) for p in (mats/'patient_mats').glob('*.npy')}
    hc = np.array([np.load(p)[i,j] for p in (mats/'HC_mats').glob('*.npy')])
    assert len(hc)==26 and np.isfinite(hc).all()
    mu,sd = hc.mean(), hc.std(ddof=1)
    d['RVPL_SMA_raw'] = d.Subject.map({s:m[i,j] for s,m in patients.items()})
    assert np.allclose((d.RVPL_SMA_raw-mu)/sd,d[FEATURE],equal_nan=True,atol=1e-10)
    # Verify clinical records rather than relying only on a derivative feature file.
    check = d.merge(clinical, on='Subject',suffixes=('','_source'),validate='one_to_one')
    for c in ['age','days_to_surgery','CRS_pre','CRS_6mo','diagnosis_6mo']:
        assert np.allclose(check[c],check[c+'_source'],equal_nan=True), c
    extra = [c for c in clinical if c not in d or c=='Subject']
    d = d.merge(clinical[extra],on='Subject',validate='one_to_one')
    d = d.merge(pd.read_csv(OUT/'00_inputs/lesion_volume_deidentified.csv'),on='Subject',validate='one_to_one')
    d['baseline_MCSminus']=(d.diagnosis_pre>=2).astype(int)
    d['log1p_lesion_vol_ml']=np.log1p(d.lesion_vol_ml)
    d['RVPL_SMA_z']=d[FEATURE]
    d['final_state_code']=d.diagnosis_6mo.astype(int)-1
    d['final_state']=d.final_state_code.map({0:'VS/UWS',1:'MCS-',2:'MCS+',3:'EMCS'})
    d.to_csv(OUT/'00_inputs/master_imaging_cohort_n30.csv',index=False)
    focus=d[d.RVPL_SMA_z.notna()].copy()
    assert len(focus)==26 and focus.favourable6.sum()==12
    assert focus.final_state.notna().all() and (focus.died==0).all()
    focus.to_csv(OUT/'00_inputs/master_focused_cohort_n26.csv',index=False)
    keys={'source_n':len(d),'focused_n':len(focus),'events':int(focus.favourable6.sum()),
          'HC_n':len(hc),'HC_mean':mu,'HC_sample_sd':sd,
          'strict_focused_n':int(focus.strict_available.sum()),
          'missing_focused_subjects':d.loc[d.RVPL_SMA_z.isna(),'Subject'].tolist(),
          'raw_vs_sealed_max_error':float(np.nanmax(np.abs((d.RVPL_SMA_raw-mu)/sd-d[FEATURE])))}
    save_json(OUT/'logs/cohort_qc.json',keys)
    emit('Cohort and raw-FC reconstruction passed: '+str(keys))
    return d,focus,nodes,patients


def jt(x,g):
    # Sum pairwise concordances across strictly ordered groups; half for tied x.
    a,b=np.triu_indices(len(x),1)
    sign=np.sign((x[b]-x[a])*(g[b]-g[a]))
    valid=g[a]!=g[b]
    return float(np.sum((sign[valid]+1)/2))


def ordered_state(d):
    cols=['Subject','final_state','final_state_code','RVPL_SMA_raw','RVPL_SMA_z',
          'CRS_pre','days_to_surgery','age','etiology_group']
    d[cols].rename(columns={'Subject':'patient_id','CRS_pre':'baseline_CRSR',
                             'days_to_surgery':'timing_days'}).to_csv(
        OUT/'01_ordered_state/ordered_state_analysis.csv',index=False)
    x=d.RVPL_SMA_raw.to_numpy(); g=d.final_state_code.to_numpy()
    rx=stats.rankdata(x); rg=stats.rankdata(g)
    rx=(rx-rx.mean())/np.linalg.norm(rx-rx.mean())
    rg=(rg-rg.mean())/np.linalg.norm(rg-rg.mean())
    rho=float(rx@rg); observed_jt=jt(x,g)
    rng=np.random.default_rng(SEED)
    null=[]
    for b in range(N_PERM):
        ix=rng.permutation(len(d))
        null.append({'iteration':b,'spearman_rho':float(rx@rg[ix]),'JT':jt(x,g[ix])})
    null=pd.DataFrame(null); null.to_csv(OUT/'01_ordered_state/ordered_state_permutation_null.csv',index=False)
    p=(1+int((null.spearman_rho.abs()>=abs(rho)-1e-12).sum()))/(N_PERM+1)
    jtp=(1+int((null.JT>=observed_jt-1e-12).sum()))/(N_PERM+1)
    group=[]
    for code in range(4):
        a=d[d.final_state_code==code]
        row={'state_code':code,'state':{0:'VS/UWS',1:'MCS-',2:'MCS+',3:'EMCS'}[code],'n':len(a)}
        for var in ['RVPL_SMA_raw','RVPL_SMA_z']:
            row.update({var+'_median':a[var].median(),var+'_q25':a[var].quantile(.25),var+'_q75':a[var].quantile(.75)})
        group.append(row)
    group=pd.DataFrame(group); group.to_csv(OUT/'01_ordered_state/ordered_state_group_summary.csv',index=False)
    result={'n':len(d),'spearman_rho':rho,'spearman_permutation_p_two_sided':p,
            'spearman_asymptotic_p':float(stats.spearmanr(x,g).pvalue),
            'JT_statistic':observed_jt,'JT_permutation_p_increasing_one_sided':jtp,
            'permutations':N_PERM,'seed':SEED,'category_n':group.n.tolist(),
            'medians_nondecreasing':bool(np.all(np.diff(group.RVPL_SMA_raw_median)>=0)),
            'ordinal_model_decision':'omit: at least one category n<3' if group.n.min()<3 else 'eligible: inspect separately before using',
            'multiplicity':'exploratory correlated trend tests; unadjusted, not independent confirmation'}
    save_json(OUT/'01_ordered_state/ordered_state_summary.json',result)
    fig,ax=plt.subplots(figsize=(7.6,5.0))
    rng=np.random.default_rng(SEED+1)
    for code,row in group.iterrows():
        vals=d.loc[d.final_state_code==code,'RVPL_SMA_raw'].to_numpy()
        ax.scatter(code+rng.uniform(-.15,.15,len(vals)),vals,s=42,c=BLUE,alpha=.8,edgecolor='white',linewidth=.4,zorder=3)
        ax.plot([code,code],[row.RVPL_SMA_raw_q25,row.RVPL_SMA_raw_q75],c=DARK,lw=3,zorder=4)
        ax.plot([code-.16,code+.16],[row.RVPL_SMA_raw_median]*2,c=DARK,lw=2,zorder=4)
    ax.set(xticks=range(4),xticklabels=[f'{r.state}\n(n={r.n})' for r in group.itertuples()],
           ylabel='Pretreatment R-VPL-SMA connectivity (Fisher z)',xlabel='Recorded behavioural state at 6 months')
    ax.set_title('Ordered-state association',loc='left',fontweight='bold',pad=34)
    ax.text(0,1.025,f'Spearman rho = {rho:.3f}; two-sided permutation P = {p:.4f}',transform=ax.transAxes,fontsize=10)
    ax.grid(axis='y',alpha=.12); ax.set_axisbelow(True)
    sec=ax.secondary_yaxis('right',functions=(lambda y:(y-.25962071544696086)/.2518158930022524,
                                             lambda z:z*.2518158930022524+.25962071544696086))
    sec.set_ylabel('Local healthy-reference SD units',labelpad=10)
    fig.subplots_adjust(right=.84,bottom=.17,top=.82)
    savefig(fig,'Fig_A1_ordered_state_gradient')
    (OUT/'01_ordered_state/ORDERED_STATE_REPORT.md').write_text(
        '# Four-category state association\n\n'+group.to_markdown(index=False)+'\n\n'+
        f'Spearman rho={rho:.6f}, two-sided permutation P={p:.6f}; JT={observed_jt:.3f}, one-sided increasing-trend permutation P={jtp:.6f}. '
        f'{N_PERM} permutations, seed={SEED}.\n\n{result["ordinal_model_decision"]}. '
        f'Medians strictly nondecreasing: {result["medians_nondecreasing"]}. A significant global rank association is not proof of strictly monotonic group medians or adjacent-state differences.\n',encoding='utf-8')
    emit('A1 '+json.dumps(result))


def spatial():
    base=OUT/'02_spatial_convergence'
    maskpath=HARM/'group_explicit_mask_90pct.nii.gz'
    mimg=nib.load(maskpath); mask=mimg.get_fdata()>0
    aff=mimg.affine; shape=mask.shape
    world=nib.affines.apply_affine(aff,np.indices(shape).reshape(3,-1).T).reshape(*shape,3)
    center=np.array([0.,-21.,51.]); sphere=np.linalg.norm(world-center,axis=-1)<=6
    nib.save(nib.Nifti1Image(sphere.astype('uint8'),aff),base/'SMA_fixed_sphere_6mm.nii.gz')
    shutil.copy2(maskpath,OUT/'00_inputs/group_explicit_mask_90pct.nii.gz')
    qc=pd.read_csv(OUT/'00_inputs/SPM_SecondLevel_Analysis/subject_master_QC.csv')
    archived=pd.read_csv(OUT/'00_inputs/SPM_SecondLevel_Analysis/PY_final_results/final_significant_clusters_nperm5000.csv')
    archived=archived[archived.seed=='Seed_R_VPL'].copy()
    kept=qc[qc.exclude_reason.fillna('').str.strip().eq('')].copy()
    allcov=['age','duration_days','crs_pre','mean_fd']
    vols=[]; rr=[]; source=[]
    for row in kept.to_dict('records'):
        path=HARM/row['Subject']/(row['Subject']+'_Seed_R_VPL_FCz.nii.gz')
        if not path.exists() or any(pd.isna(row[c]) for c in allcov):continue
        img=nib.load(path)
        assert img.shape==shape and np.allclose(img.affine,aff),path
        vec=img.get_fdata()[mask]
        if not np.isfinite(vec).all():continue
        vols.append(vec);rr.append(row)
        source.append({'source':str(path),'sha256':sha(path)})
    meta=pd.DataFrame(rr);Yall=np.array(vols)
    meta.to_csv(base/'spatial_subjects_deidentified.csv',index=False)
    save_json(base/'SPATIAL_INPUT_HASHES.json',source+[{'source':str(maskpath),'sha256':sha(maskpath)}])
    clinical=pd.read_csv(OUT/'00_inputs/clinical_followup_deidentified.csv')
    timing=meta[['Subject','duration_days']].merge(clinical[['Subject','days_to_fmri','days_to_surgery','diagnosis_6mo']],on='Subject',validate='one_to_one')
    timing['duration_equals_days_to_fmri']=np.isclose(timing.duration_days,timing.days_to_fmri,equal_nan=False)
    timing.to_csv(base/'spatial_timing_definition_audit.csv',index=False)
    assert np.array_equal(meta.group.to_numpy(int),(timing.diagnosis_6mo>=3).to_numpy(int))
    models=['M0_nocovar','M1_+age','M2_+duration','M3_+crs','M4_+motion(full)']
    structures=np.ones((3,3,3));rows=[];matches=[];allpeaks=[]
    for sset,minframes in [('main',0),('sensitivity',90)]:
        ix=meta.n_retained>=minframes; md=meta.loc[ix].copy(); Y=Yall[ix]
        count=np.zeros(shape,dtype='uint8')
        for imodel,model in enumerate(models):
            cov=allcov[:imodel];n=len(md)
            Z=np.column_stack([np.ones(n)]+[md[c].to_numpy(float)-md[c].mean() for c in cov])
            X=np.column_stack([md.group.to_numpy(float),Z]); df=n-X.shape[1]
            inv=np.linalg.inv(X.T@X); beta=inv@(X.T@Y);res=Y-X@beta
            t=beta[0]/np.sqrt((res**2).sum(0)/df*inv[0,0]+1e-30)
            vol=np.zeros(shape,dtype='float32');vol[mask]=t
            thr=stats.t.ppf(.999,df)
            lab,nlab=ndimage.label(vol>=thr,structure=structures)
            sizes=np.bincount(lab.ravel()); accepted=np.zeros(shape,dtype=bool)
            hits=archived[(archived.subject_set==sset)&(archived.model==model)]
            assert len(hits)>0,(sset,model)
            for hit in hits.to_dict('records'):
                assert hit['n']==n and hit['df']==df,(sset,model,n,df,hit)
                peak=np.array([hit['x'],hit['y'],hit['z']])
                ijk=np.rint(nib.affines.apply_affine(np.linalg.inv(aff),peak)).astype(int)
                k=int(lab[tuple(ijk)]);assert k>0,(sset,model,peak)
                cl=lab==k;pk=np.array(np.unravel_index(np.argmax(np.where(cl,vol,-np.inf)),shape))
                pkmm=nib.affines.apply_affine(aff,pk); pval=float(vol[tuple(pk)])
                ok=bool(int(sizes[k])==hit['size'] and np.allclose(pkmm,peak,atol=.01) and abs(pval-hit['peak_t'])<=.0051)
                matches.append({'subject_set':sset,'model':model,'archived_size':hit['size'],'reconstructed_size':int(sizes[k]),
                                'archived_peak_t':hit['peak_t'],'reconstructed_peak_t':pval,
                                'peak_xyz_match':bool(np.allclose(pkmm,peak,atol=.01)),'all_match':ok})
                if not ok: raise RuntimeError(f'Archived cluster reconstruction mismatch {sset} {model}: {matches[-1]}')
                accepted|=cl
                overlap=int((sphere&cl).sum())
                medial=bool(abs(peak[0])<=20 and -35<=peak[1]<=10 and peak[2]>=40)
                allpeaks.append({**hit,'cluster_label':k,'center_to_peak_mm':float(np.linalg.norm(peak-center)),
                                 'sphere_overlap_voxels':overlap,'coordinate_envelope_medial_dorsal':medial})
            outname=f'RVPL_{sset}_M{imodel}'
            nib.save(nib.Nifti1Image(np.where(accepted,vol,0).astype('float32'),aff),base/(outname+'_significant_t.nii.gz'))
            count+=accepted.astype('uint8')
            coords=world[accepted];dist=float(np.linalg.norm(coords-center,axis=1).min())
            hrows=[r for r in allpeaks if r['subject_set']==sset and r['model']==model]
            med=[r for r in hrows if r['coordinate_envelope_medial_dorsal']]
            nearest_peak=min(r['center_to_peak_mm'] for r in hrows)
            overlaps=int((sphere&accepted).sum())
            rows.append({'subject_set':sset,'model':model,'n':n,'df':df,'covariates':'+'.join(cov) or 'none',
                         'CDT_one_sided_p':.001,'CDT_t':float(thr),'cluster_FWE_alpha':.05,
                         'archived_permutations':5000,'significant_clusters':len(hits),
                         'significant_voxels':int(accepted.sum()),'SMA_sphere_voxels':int(sphere.sum()),
                         'SMA_voxels_in_group_mask':int((sphere&mask).sum()),'overlap_voxels':overlaps,
                         'SMA_coverage_fraction':overlaps/sphere.sum(),'significant_union_SMA_fraction':overlaps/accepted.sum(),
                         'center_to_nearest_significant_voxel_mm':dist,'sphere_surface_to_voxel_center_gap_mm':max(dist-6,0),
                         'center_to_nearest_cluster_peak_mm':nearest_peak,
                         'center_to_nearest_medial_dorsal_peak_mm':min(r['center_to_peak_mm'] for r in med) if med else np.nan,
                         'has_medial_dorsal_peak_by_coordinate_envelope':bool(med),
                         'all_clusters_cross_8_seed_bonferroni':bool(hits.bonf8_sig.all())})
            emit(f'A2 reconstructed {sset} {model}: n={n}, significant voxels={accepted.sum()}, SMA overlap={overlaps}, center distance={dist:.2f} mm')
        name='RVPL_model_consistency_count' if sset=='main' else 'RVPL_strict_model_consistency_count'
        nib.save(nib.Nifti1Image(count,aff),base/(name+'.nii.gz'))
    pd.DataFrame(rows).to_csv(base/'seed2voxel_sma_convergence_metrics.csv',index=False)
    pd.DataFrame(matches).to_csv(base/'archived_cluster_reconstruction_checks.csv',index=False)
    pk=pd.DataFrame(allpeaks)
    pk['cluster_SMA_fraction']=pk.sphere_overlap_voxels/pk['size']
    pk.to_csv(base/'seed2voxel_cluster_peak_distances.csv',index=False)
    emit('A2 all archived cluster size, peak, T, n and df matches passed.')


if __name__=='__main__':
    clinical=prepare_inputs()
    data,focus,nodes,patients=cohort(clinical)
    ordered_state(focus)
    spatial()
