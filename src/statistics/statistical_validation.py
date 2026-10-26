"""Prespecified statistical validation and sensitivity analyses for the focused R_VPL-SMA candidate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='10'
from pathlib import Path
import sys, json, hashlib, ctypes, time, shutil, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(os.environ.get('RVPL_SMA_PROJECT_ROOT', Path.cwd()))
V=Path(os.environ.get('RVPL_SMA_DISCOVERY_DIR', ROOT/'restricted_inputs'/'discovery_analysis'))
W=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RVPL_SMA_VALIDATION_OUTPUT', ROOT/'outputs'/'statistical_validation'))
A=OUT/'analysis_outputs'
PRIOR_SUBMISSION=Path(os.environ.get('RVPL_SMA_PRIOR_SUBMISSION_DIR', ROOT/'restricted_inputs'/'prior_submission'))
SUPPLEMENT_INPUTS=Path(os.environ.get('RVPL_SMA_SUPPLEMENT_INPUTS_DIR', ROOT/'restricted_inputs'/'supplement_inputs'))
A.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'primary'))
import clinical_models as core
libdirs=[]
_rtools=os.environ.get('RTOOLS_BIN')
if _rtools and hasattr(os, 'add_dll_directory'):
    libdirs=[os.add_dll_directory(_rtools)]
lib=ctypes.CDLL(str(W/'firth_accelerator.dll'))
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.fit_one.argtypes=[D,D,ctypes.c_int,ctypes.c_int,D,D,I]
lib.permute.argtypes=[D,D,D,D,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,D,I]
lib.calibrate.argtypes=[D,D,D,D,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,D]
B=5000; S=2000; SEED=20261023
fcol='R_VPL_SMA_z_local26_z'
d=pd.read_csv(V/'00_inputs/master_imaging_cohort_n30.csv')
names=pd.read_csv(V/'00_inputs/FCmat_work/node_info.csv').name.tolist()
P=np.array([np.load(V/'00_inputs/FCmat_work/patient_mats'/(s+'.npy')) for s in d.Subject])
H=np.array([np.load(p) for p in sorted((V/'00_inputs/FCmat_work/HC_mats').glob('*.npy'))])
ij=np.triu_indices(30,1); edges=P[:,ij[0],ij[1]]; g=d.favourable6.to_numpy(int)
focus=next(k for k,(i,j) in enumerate(zip(*ij)) if {names[i],names[j]}=={'R_VPL','SMA'})

def emit(s): print(s,flush=True)
def js(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def z(x): return (x-x.mean())/x.std(ddof=1)
def bh(p): return stats.false_discovery_control(np.asarray(p,float))
def fastfit(X,y):
    X=np.ascontiguousarray(X,dtype=float);y=np.ascontiguousarray(y,dtype=float)
    b=np.zeros(X.shape[1]);s=b.copy();ok=np.zeros(1,np.int32)
    lib.fit_one(X,y,len(y),X.shape[1],b,s,ok)
    return b,s,int(ok[0])
def build(feature,covs=['CRS_pre','log1p_days'],strict=False):
    dd=d.copy();dd['_feature']=feature
    if strict: dd=dd[dd.strict27_sensitivity_cohort]
    dd=dd.replace([np.inf,-np.inf],np.nan).dropna(subset=['_feature','favourable6']+covs)
    Z=np.column_stack([np.ones(len(dd))]+[dd[c].to_numpy(float) if c=='TBI' else z(dd[c].to_numpy(float)) for c in covs])
    x=dd._feature.to_numpy(float);y=dd.favourable6.to_numpy(float)
    return dd,np.ascontiguousarray(Z),x,y
def infer(Z,x,y):
    b,s,ok=fastfit(np.column_stack([Z,x]),y); beta=b[-1];se=s[-1]
    return dict(n=len(y),events=int(y.sum()),beta=beta,se=se,OR=np.exp(beta),OR95_low=np.exp(beta-1.96*se),OR95_high=np.exp(beta+1.96*se),wald_p=2*stats.norm.sf(abs(beta/se)),converged=ok==1)
def residuals(Z,x):
    fitted=Z@np.linalg.lstsq(Z,x,rcond=None)[0]
    return np.ascontiguousarray(fitted),np.ascontiguousarray(x-fitted)
def uniform(label,Z,x,y,seed):
    path=A/(label+'_permutation.npz')
    r=infer(Z,x,y);fitted,resid=residuals(Z,x)
    if path.exists():
        p=np.load(path);null=p['beta'];status=p['status']
    else:
        null=np.zeros(B);status=np.zeros(B,np.int32)
        lib.permute(Z,fitted,resid,np.ascontiguousarray(y),len(y),Z.shape[1],B,seed,null,status)
        np.savez_compressed(path,beta=null,status=status)
    extreme=int((abs(null)>=abs(r['beta'])).sum())
    ci=stats.binomtest(extreme,B).proportion_ci()
    r.update(permutation_p=(extreme+1)/(B+1),permutations=B,permutation_extreme=extreme,MC95_low=ci.low,MC95_high=ci.high,permutation_nonconverged=int((status!=1).sum()),seed=seed)
    emit(label+': '+json.dumps(r))
    return r

def truthlock():
    manifest=[]
    for p in (PRIOR_SUBMISSION).iterdir():
        if p.is_file(): manifest.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    js(OUT/'PRIOR_SUBMISSION_MANIFEST.json',manifest)
    shutil.copy2(W/'ANALYSIS_SPECIFICATION.md',OUT/'ANALYSIS_SPECIFICATION.md')
    sc=pd.read_csv(PRIOR_SUBMISSION/'Supplementary_Data_1.csv')
    new=[]
    for k,(i,j) in enumerate(zip(*ij)):
        x=edges[:,k];ok=np.isfinite(x);a=x[ok&(g==1)];b=x[ok&(g==0)]
        t=stats.ttest_ind(a,b,equal_var=False)
        new.append(dict(node_A=names[i],node_B=names[j],n_available=int(ok.sum()),n_MCSplus_EMCS=len(a),n_below_MCSplus=len(b),difference_Fisher_z=a.mean()-b.mean(),Welch_t=t.statistic,Welch_df=t.df,Welch_P_two_sided=t.pvalue,edge_index=k))
    screen=pd.DataFrame(new);screen['BH_q_across_435']=bh(screen.Welch_P_two_sided)
    screen=screen.sort_values('Welch_P_two_sided').reset_index(drop=True);screen['rank']=np.arange(1,436)
    old={frozenset([r.node_A,r.node_B]):r for r in sc.itertuples()}
    assert all(np.isclose(r.Welch_P_two_sided,old[frozenset([r.node_A,r.node_B])].Welch_P_two_sided,rtol=1e-9,atol=1e-12) for r in screen.itertuples())
    screen.to_csv(A/'discovery_reproduced_with_Welch_statistics.csv',index=False)
    _,Z,x,y=build(d[fcol].to_numpy())
    r=infer(Z,x,y);oldr=pd.read_csv(V/'05_master_results/robustness_verified.csv').query("model=='main_base'")
    if len(oldr)==0: oldr=pd.read_csv(V/'05_master_results/robustness_verified.csv').iloc[:1]
    assert np.isclose(r['OR'],oldr.iloc[0].OR,rtol=1e-7)
    rng=np.random.default_rng(SEED);checks=[]
    for k in range(40):
        xx=x.copy() if k==0 else rng.permutation(x)
        yy=y.copy() if k<20 else rng.permutation(y)
        X=np.column_stack([Z,xx]);b,s,ok=fastfit(X,yy);ref=core.firth_fit(X,yy)
        checks.append(dict(case=k,beta_max_abs_diff=float(max(abs(b-ref['beta']))),se_max_abs_diff=float(max(abs(s-ref['se']))),accelerator_converged=ok,python_converged=ref['converged']))
    cc=pd.DataFrame(checks);cc.to_csv(A/'accelerator_vs_archived_python.csv',index=False)
    assert cc.beta_max_abs_diff.max()<1e-6 and cc.se_max_abs_diff.max()<1e-6
    js(A/'TRUTH_LOCK.json',dict(discovery_edges_verified=435,focus=r,accelerator_max_beta_difference=cc.beta_max_abs_diff.max(),sources_unchanged=True))
    emit('TRUTH LOCK PASSED: 435 Welch comparisons and primary OR reproduced; accelerator independently checked.')

def families():
    jobs={}; memberships=[]
    for family,labels in [('seed_to_SMA',names[22:]),('RVPL_to_cortex',names[:22])]:
        for label in labels:
            a,b=(names.index(label),names.index('SMA')) if family=='seed_to_SMA' else (names.index('R_VPL'),names.index(label))
            key='edge_'+str(min(a,b))+'_'+str(max(a,b));memberships.append((family,label,key))
            if key not in jobs:
                ref=H[:,a,b];ref=ref[np.isfinite(ref)];xx=(P[:,a,b]-ref.mean())/ref.std(ddof=1)
                dd,Z,x,y=build(xx)
                jobs[key]=dict(result=uniform(key,Z,x,y,SEED+min(a,b)*1000+max(a,b)),Z=Z,x=x,y=y,dd=dd)
    rows=[]
    for family,label,key in memberships:
        rows.append(dict(family=family,label=label,**jobs[key]['result']))
    f=pd.DataFrame(rows)
    for family in f.family.unique():
        ix=f.family==family
        f.loc[ix,'permutation_BH_q']=bh(f.loc[ix,'permutation_p'])
        f.loc[ix,'wald_BH_q']=bh(f.loc[ix,'wald_p'])
        f.loc[ix,'rank_permutation']=f.loc[ix,'permutation_p'].rank(method='min')
        f.loc[ix,'rank_wald']=f.loc[ix,'wald_p'].rank(method='min')
    f.to_csv(A/'families_uniform5000_Wald_and_permutation.csv',index=False)
    _,Z,x,y=build(d[fcol].to_numpy(),strict=True)
    strict=uniform('primary_strict',Z,x,y,SEED+99000)
    js(A/'strict_uniform5000.json',strict)
    return jobs

def calibration(label,Z,x,y):
    path=A/('calibration_'+label+'.csv')
    if path.exists() and len(pd.read_csv(path))==S: return
    b,_,ok=fastfit(Z,y);prob=1/(1+np.exp(-(Z@b)))
    fitted,resid=residuals(Z,x);rng=np.random.default_rng(SEED+len(y)*20)
    rows=[]
    for start in range(0,S,100):
        ys=[]
        while len(ys)<100:
            sim=(rng.random((1000,len(y)))<prob).astype(float)
            ys.extend(sim[sim.sum(1)==y.sum()][:100-len(ys)])
        ys=np.ascontiguousarray(np.array(ys));out=np.zeros((100,5))
        t=time.time();lib.calibrate(Z,fitted,resid,ys,len(y),Z.shape[1],100,B,SEED+start*4321+len(y),out)
        for k,r in enumerate(out):rows.append(dict(sim=start+k,permutation_p=r[0],beta=r[1],se=r[2],wald_p=2*stats.norm.sf(abs(r[1]/r[2])),permutation_nonconverged=int(r[3]),observed_converged=int(r[4]),events=int(y.sum()),n=len(y)))
        pd.DataFrame(rows).to_csv(path,index=False)
        emit('CALIBRATION '+label+' '+str(start+100)+'/'+str(S)+'; batch seconds '+str(round(time.time()-t,1)))
    result=[];df=pd.DataFrame(rows)
    for method in ['permutation_p','wald_p']:
        for alpha in [.05,.01]:
            count=int((df[method]<alpha).sum());ci=stats.binomtest(count,S).proportion_ci()
            result.append(dict(design=label,method=method,alpha=alpha,null_simulations=S,rejections=count,type_I_error=count/S,CI95_low=ci.low,CI95_high=ci.high))
    pd.DataFrame(result).to_csv(A/('calibration_'+label+'_summary.csv'),index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for method,color in [('permutation_p','#0072B2'),('wald_p','#D55E00')]:
        p=np.sort(df[method]);u=(np.arange(S)+.5)/S
        axes[0].plot(-np.log10(u),-np.log10(p),'.',ms=2,label=method,c=color)
        axes[1].plot(p,np.arange(1,S+1)/S,label=method,c=color)
    axes[0].plot([0,4],[0,4],'--',c='gray');axes[0].set(xlabel='Expected -log10(P)',ylabel='Observed -log10(P)',title=label+' null QQ')
    axes[1].plot([0,1],[0,1],'--',c='gray');axes[1].set(xlabel='P',ylabel='Empirical cumulative probability');axes[1].legend()
    fig.tight_layout();fig.savefig(A/('calibration_'+label+'.png'),dpi=180);plt.close(fig)

def fast_sensitivities():
    complete=np.isfinite(edges).all(1);x=edges[complete];gg=g[complete]
    res=stats.ttest_ind(x[gg==1],x[gg==0],axis=0,equal_var=False)
    cc=pd.DataFrame(dict(node_A=[names[i] for i in ij[0]],node_B=[names[j] for j in ij[1]],Welch_t=res.statistic,Welch_df=res.df,Welch_p=res.pvalue,difference_Fisher_z=x[gg==1].mean(0)-x[gg==0].mean(0)))
    cc['BH_q']=bh(cc.Welch_p);cc=cc.sort_values('Welch_p').reset_index(drop=True);cc['rank']=np.arange(1,436);cc.to_csv(A/'common_sample_all435.csv',index=False)
    fr=cc[(cc.node_A.isin(['R_VPL','SMA']))&(cc.node_B.isin(['R_VPL','SMA']))].iloc[0]
    js(A/'common_sample_summary.json',dict(n=int(complete.sum()),events=int(gg.sum()),non_events=int((gg==0).sum()),focus=fr.to_dict(),minimum_group_size=int(min(gg.sum(),(gg==0).sum()))))
    # Patient-level common-sample membership is intentionally not written by the public release.
    # Synchronized label permutations preserve dependence and fixed missingness.
    filled=np.nan_to_num(edges);avail=np.isfinite(edges).astype(float);sq=filled**2
    rng=np.random.default_rng(SEED+500000);null=[];degenerate=0
    for start in range(0,10000,500):
        labels=np.array([rng.permutation(g) for _ in range(500)])
        n1=labels@avail;n0=(1-labels)@avail;s1=labels@filled;s0=(1-labels)@filled
        q1=labels@sq;q0=(1-labels)@sq
        with np.errstate(invalid='ignore',divide='ignore'):
            v1=(q1-s1*s1/n1)/(n1-1);v0=(q0-s0*s0/n0)/(n0-1)
            tt=(s1/n1-s0/n0)/np.sqrt(v1/n1+v0/n0)
        degenerate+=int((~np.isfinite(tt)).sum());null.extend(np.nanmax(abs(tt),axis=1))
    null=np.array(null);np.save(A/'max_abs_Welch_t_null_10000.npy',null)
    observed=pd.read_csv(A/'discovery_reproduced_with_Welch_statistics.csv').iloc[0]
    extreme=int((null>=abs(observed.Welch_t)).sum());ci=stats.binomtest(extreme,10000).proportion_ci()
    js(A/'max_statistic_summary.json',dict(B=10000,seed=SEED+500000,observed_top_abs_t=abs(observed.Welch_t),FWER_p=(extreme+1)/10001,MC95_low=ci.low,MC95_high=ci.high,degenerate_edge_permutations=degenerate,exchangeability='Unadjusted outcome-label exchangeability under global null; heteroscedastic observational groups make this a sensitivity, not an exact randomization test.'))
    nonfocus=np.delete(edges,focus,axis=1);globalfc=np.nanmean(nonfocus,axis=1);count=np.isfinite(nonfocus).sum(1)
    ri=names.index('R_VPL');cs=[i for i in range(22) if names[i]!='SMA'];rvpl=np.nanmean(P[:,ri,cs],axis=1)
    d['global_FC']=globalfc;d['nonfocused_edge_count']=count;d['RVPL_other_cortex_mean']=rvpl
    rows=[];base=None
    for label,covs in [('base',['CRS_pre','log1p_days']),('global_FC',['CRS_pre','log1p_days','global_FC']),('RVPL_other_cortex',['CRS_pre','log1p_days','RVPL_other_cortex_mean'])]:
        dd,Z,x,y=build(d[fcol].to_numpy(),covs);r=infer(Z,x,y)
        if base is None:base=r['beta']
        rows.append(dict(model=label,**r,coefficient_attenuation_percent=100*(1-r['beta']/base)))
    dd=d[d.primary_available].copy();corr=stats.spearmanr(dd.nonfocused_edge_count,dd.global_FC);pout=stats.mannwhitneyu(dd.loc[dd.favourable6==1,'nonfocused_edge_count'],dd.loc[dd.favourable6==0,'nonfocused_edge_count'],method='asymptotic')
    if abs(corr.statistic)>=.4 or corr.pvalue<.05 or pout.pvalue<.05:
        dd,Z,x,y=build(d[fcol].to_numpy(),['CRS_pre','log1p_days','global_FC','nonfocused_edge_count']);r=infer(Z,x,y)
        rows.append(dict(model='global_FC_plus_availability',**r,coefficient_attenuation_percent=100*(1-r['beta']/base)))
    pd.DataFrame(rows).to_csv(A/'global_connectivity_specificity.csv',index=False)
    js(A/'global_FC_availability_diagnostics.json',dict(count_vs_global_rho=corr.statistic,count_vs_global_p=corr.pvalue,count_vs_outcome_MWU_p=pout.pvalue))
    # Public aggregate only, no participant identifiers or clinical rows.
    clinical=pd.read_csv(SUPPLEMENT_INPUTS/'clinical50_anonymous_values.csv')
    table=[]
    groups=[('All treated',clinical),('Primary imaging',clinical[clinical.QC30]),('Focused edge',clinical[clinical.focus26]),('Focused below MCS+',clinical[clinical.focus26 & (clinical.favourable6==0)]),('Focused MCS+/EMCS',clinical[clinical.focus26 & (clinical.favourable6==1)]),('No focused edge',clinical[~clinical.focus26])]
    for group,q in groups:
        row=dict(group=group,n=len(q),male_n=int(q.male.sum()),female_n=int(len(q)-q.male.sum()),TBI_n=int(q.TBI.sum()),baseline_MCSminus_n=int(q.baseline_MCSminus.sum()),MCSplus_EMCS_n=int(q.favourable6.sum()))
        for col in ['age','CRS_pre','days_to_surgery']:
            row[col+'_median']=q[col].median();row[col+'_q25']=q[col].quantile(.25);row[col+'_q75']=q[col].quantile(.75)
        table.append(row)
    pd.DataFrame(table).to_csv(A/'Table1_cohort_composition.csv',index=False)
    loo=pd.read_csv(V/'05_master_results/leave_one_out.csv');loo.groupby('cohort').agg(beta_min=('beta','min'),beta_max=('beta','max')).to_csv(A/'LOO_magnitude_summary.csv')
    boot=pd.read_csv(V/'05_master_results/resampling_audit_summary.csv');boot['bootstrap_failed']=boot.bootstrap_attempted-boot.bootstrap_valid
    boot['bootstrap_OR_q025']=np.exp(boot.bootstrap_beta_q025);boot['bootstrap_OR_q975']=np.exp(boot.bootstrap_beta_q975);boot.to_csv(A/'bootstrap_uncertainty_original_audit.csv',index=False)
    emit('Complete-case, 10000 max-statistic permutations, global FC and Table 1 completed.')

def robustness():
    old=pd.read_csv(V/'05_master_results/robustness_verified.csv');rows=[]
    for k,r in enumerate(old.itertuples()):
        _,Z,x,y=build(d[fcol].to_numpy(),r.covariates.split('+'),r.model.startswith('strict'))
        if r.model=='main':
            shared=pd.read_csv(A/'families_uniform5000_Wald_and_permutation.csv')
            out=shared[(shared.family=='seed_to_SMA')&(shared.label=='R_VPL')].iloc[0].to_dict()
            for key in ['family','label','permutation_BH_q','wald_BH_q','rank_permutation','rank_wald']:out.pop(key,None)
        elif r.model=='strict':out=json.loads((A/'strict_uniform5000.json').read_text())
        else:out=uniform('robustness_'+r.model,Z,x,y,SEED+600000+k)
        assert np.isclose(out['OR'],r.OR,rtol=1e-7)
        rows.append(dict(model=r.model,**out,covariates=r.covariates,historical_permutation_p=r.frozen_permutation_p,historical_iterations=r.frozen_permutations))
    pd.DataFrame(rows).to_csv(A/'robustness_uniform5000.csv',index=False)

def spatial_audit():
    qc=pd.read_csv(V/'00_inputs/SPM_SecondLevel_Analysis/subject_master_QC.csv')
    meta=pd.read_csv(V/'02_spatial_convergence/spatial_subjects_deidentified.csv')
    sens=meta[meta.n_retained>=90]
    counts=pd.read_csv(Path(os.environ.get('RVPL_SMA_SMA6MP_COUNTS', ROOT/'restricted_inputs'/'spatial_validation'/'SMA_6mp_model_counts.csv')))
    metrics=pd.read_csv(V/'02_spatial_convergence/seed2voxel_sma_convergence_metrics.csv')
    assert counts.significant_voxels_in_SMA_6mp.tolist()==[42,38,37,0,0,47,37,38,21,23]
    assert counts.original_fixed_ROI_overlap_voxels.tolist()==[19,17,14,0,0,19,16,17,18,16]
    assert (metrics.significant_clusters>0).all()
    counts.to_csv(A/'spatial_overlap_two_definitions.csv',index=False)
    allcov=['age','duration_days','crs_pre','mean_fd']
    records=[]
    for i,r in qc.iterrows():
        path=Path('./restricted_inputs/archive/harmonized_output/harmonized_output')/r.Subject/(r.Subject+'_Seed_R_VPL_FCz.nii.gz')
        records.append(dict(private_code=r.Subject,no_scan_QC_exclusion=pd.isna(r.exclude_reason) or str(r.exclude_reason).strip()=='',all_covariates_present=not r[allcov].isna().any(),seed_map_exists=path.exists(),spatial_main=r.Subject in set(meta.Subject),spatial_sensitivity=r.Subject in set(sens.Subject),focused_main=r.Subject in set(d.loc[d.primary_available,'Subject']),focused_strict=r.Subject in set(d.loc[d.strict_available,'Subject']),retained_frames=r.n_retained))
    private=pd.DataFrame(records)
    # Patient-level spatial audit rows are retained in memory only; the public release writes aggregate counts.
    grouped=private.groupby(['no_scan_QC_exclusion','all_covariates_present','seed_map_exists','spatial_main','spatial_sensitivity'],dropna=False).size().reset_index(name='n')
    grouped.to_csv(A/'spatial_cohort_rule_counts.csv',index=False)
    js(A/'spatial_cohort_summary.json',dict(scan_QC_records=len(qc),spatial_main=len(meta),spatial_sensitivity=len(sens),main_events=int(meta.group.sum()),sensitivity_events=int(sens.group.sum()),removed_by_retained_frame_rule=len(meta)-len(sens),primary_ROI_imaging=30,strict_ROI_imaging=27,focused_main=26,focused_strict=23,primary_not_spatial=int((~d.Subject.isin(meta.Subject)).sum()),spatial_not_primary=int((~meta.Subject.isin(d.Subject)).sum()),strict_focused_not_spatial_sensitivity=int((~d.loc[d.strict_available,'Subject'].isin(sens.Subject)).sum()),inclusion_rule='No archived scan-QC exclusion; R_VPL seed map available and finite in group mask; age, injury-to-fMRI duration, baseline CRS-R and mean FD all present. Sensitivity additionally requires >=90 retained frames.'))
    # Influence only: t values at atlas voxels; no new corrected spatial inference.
    import nibabel as nib
    atlasfiles=list(Path(os.environ.get('RVPL_SMA_SMA6MP_SPATIAL_DIR', ROOT/'restricted_inputs'/'spatial_validation')).glob('*.nii*'))
    js(A/'spatial_influence_feasibility.json',dict(atlas_files=[p.name for p in atlasfiles],status='Cohort and overlap audit completed. Full permutation influence maps were not rerun; existing inference is retained without selecting a favourable deletion.'))
    roi=pd.read_csv(SUPPLEMENT_INPUTS/'S1_ROI_definitions.csv')
    roi['archived_System']=roi.System
    roi.loc[roi.Node=='MCC','System']='Cingulate (anatomical label; network unassigned)'
    roi.to_csv(A/'ROI_metadata_audit.csv',index=False)
    clinical=pd.read_csv(V/'00_inputs/clinical_followup_deidentified.csv')
    js(A/'concurrent_intervention_audit.json',dict(retained_fields=clinical.columns.tolist(),reliable_patient_level_medication_rehabilitation_other_neuromodulation_fields=False,decision='No concurrent-intervention adjustment invented; lack of complete standardized capture is reported as a limitation.'))
    emit('Spatial cohort, both overlap metrics, ROI metadata and concurrent-intervention availability audited.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['truth','families','calibration','sensitivity','robustness','spatial']);args=parser.parse_args()
    if args.stage=='truth':truthlock()
    if args.stage=='families':families()
    if args.stage=='sensitivity':fast_sensitivities()
    if args.stage=='robustness':robustness()
    if args.stage=='spatial':spatial_audit()
    if args.stage=='calibration':
        for label,xx,strict in [('main_primary',d[fcol].to_numpy(),False),('strict_primary',d[fcol].to_numpy(),True),('RmRt_SMA',(P[:,names.index('R_mRt'),names.index('SMA')]-H[:,names.index('R_mRt'),names.index('SMA')].mean())/H[:,names.index('R_mRt'),names.index('SMA')].std(ddof=1),False)]:
            _,Z,x,y=build(xx,strict=strict);calibration(label,Z,x,y)
