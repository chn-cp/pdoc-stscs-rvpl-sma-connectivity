#!/usr/bin/env python3
"""Locked low-dimensional clinical models after outcome-blind Stage D sealing."""
from pathlib import Path
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path('./restricted_inputs/legacy_stage2')
STAGEC = ROOT / 'CORRECTIVE_AUDIT_20260909/STAGEC_FIXED30_TOPOLOGY'
STAGED = ROOT / 'CORRECTIVE_AUDIT_20260909/STAGED_PATIENT_FIXED30_PROJECTION'
OUT = ROOT / 'CORRECTIVE_AUDIT_20260909/STAGEE_CLINICAL_FIXED30_MODELS'
MRI = Path('./restricted_inputs/MRI')
SEED = 20260920
PRIMARY = 'R_VPL_sensorimotor_preference_local26_z'
SECONDARY = [
    'R_VPL_SMA_minus_othercortex_local26_z',
    'R_VPL_SMA_cortical_percentile_local26_z',
    'SMA_RVPL_minus_other_right_thalamus_local26_z',
    'R_VPL_SMA_z_local26_z',
]
warnings.filterwarnings('ignore')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def penalized_loglik(beta, X, y):
    eta=np.clip(X@beta,-35,35); p=expit(eta)
    w=np.clip(p*(1-p),1e-9,None)
    info=X.T@(w[:,None]*X)
    sign,logdet=np.linalg.slogdet(info)
    if sign<=0: return -np.inf
    ll=np.sum(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1)))
    return float(ll+0.5*logdet)


def firth_fit(X,y,max_iter=300,tol=1e-8):
    X=np.asarray(X,float); y=np.asarray(y,float); beta=np.zeros(X.shape[1])
    converged=False
    for iteration in range(max_iter):
        eta=np.clip(X@beta,-35,35); p=expit(eta); w=np.clip(p*(1-p),1e-9,None)
        info=X.T@(w[:,None]*X); inv=np.linalg.pinv(info)
        h=w*np.einsum('ij,jk,ik->i',X,inv,X)
        score=X.T@(y-p+h*(0.5-p)); step=inv@score
        old=penalized_loglik(beta,X,y); scale=1.0
        while scale>1e-7 and penalized_loglik(beta+scale*step,X,y)<old:
            scale*=0.5
        new=beta+scale*step
        if np.max(np.abs(new-beta))<tol:
            beta=new; converged=True; break
        beta=new
    eta=np.clip(X@beta,-35,35); p=expit(eta); w=np.clip(p*(1-p),1e-9,None)
    info=X.T@(w[:,None]*X); cov=np.linalg.pinv(info); se=np.sqrt(np.clip(np.diag(cov),0,None))
    return {'beta':beta,'se':se,'cov':cov,'pred':p,'pll':penalized_loglik(beta,X,y),
            'converged':converged,'iterations':iteration+1}


def design(frame, feature=None):
    b=(frame.CRS_pre-frame.CRS_pre.mean())/frame.CRS_pre.std(ddof=1)
    lt=np.log1p(frame.days_to_surgery)
    lt=(lt-lt.mean())/lt.std(ddof=1)
    cols=[np.ones(len(frame)),b.to_numpy(),lt.to_numpy()]
    names=['Intercept','baseline_CRS_R_z','log1p_timing_z']
    if feature:
        cols.append(frame[feature].to_numpy(float)); names.append(feature)
    return np.column_stack(cols),names


def feature_test(frame,outcome,feature,n_perm,n_boot,rng):
    frame=frame[['Subject',outcome,'CRS_pre','days_to_surgery',feature]].dropna().copy()
    y=frame[outcome].to_numpy(int); X0,_=design(frame); X1,names=design(frame,feature)
    m0=firth_fit(X0,y); m1=firth_fit(X1,y); idx=names.index(feature)
    beta=float(m1['beta'][idx]); se=float(m1['se'][idx]); z=beta/se if se>0 else np.nan
    base_hat=np.linalg.lstsq(X0,frame[feature].to_numpy(float),rcond=None)[0]
    fitted=X0@base_hat; resid=frame[feature].to_numpy(float)-fitted
    null=[]
    for _ in range(n_perm):
        fp=fitted+rng.permutation(resid)
        mp=firth_fit(np.column_stack([X0,fp]),y)
        null.append(float(mp['beta'][-1]))
    null=np.asarray(null)
    pperm=float((1+np.sum(np.abs(null)>=abs(beta)))/(n_perm+1))
    boots=[]; failed=0
    n=len(frame)
    for _ in range(n_boot):
        ix=rng.integers(0,n,n)
        try:
            mb=firth_fit(X1[ix],y[ix])
            if mb['converged'] and np.isfinite(mb['beta'][idx]): boots.append(float(mb['beta'][idx]))
            else: failed+=1
        except Exception: failed+=1
    boots=np.asarray(boots)
    loo=[]
    for i in range(n):
        keep=np.arange(n)!=i
        ml=firth_fit(X1[keep],y[keep])
        loo.append({'removed_Subject':frame.Subject.iloc[i], 'beta_feature':float(ml['beta'][idx]),
                    'converged':bool(ml['converged'])})
    result={
        'outcome':outcome,'feature':feature,'n':n,'events':int(y.sum()),
        'beta':beta,'OR':float(np.exp(beta)),'se':se,'wald_z':z,
        'wald_p':float(2*stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        'wald95_low':float(beta-1.96*se),'wald95_high':float(beta+1.96*se),
        'OR95_low':float(np.exp(beta-1.96*se)),'OR95_high':float(np.exp(beta+1.96*se)),
        'M0_pll':m0['pll'],'M1_pll':m1['pll'],'delta_penalized_ll':m1['pll']-m0['pll'],
        'permutation_p':pperm,'n_permutations':n_perm,
        'bootstrap_n_valid':int(len(boots)),'bootstrap_n_failed':failed,
        'bootstrap_beta_low':float(np.quantile(boots,.025)) if len(boots) else np.nan,
        'bootstrap_beta_high':float(np.quantile(boots,.975)) if len(boots) else np.nan,
        'bootstrap_same_direction':float(np.mean(np.sign(boots)==np.sign(beta))) if len(boots) else np.nan,
        'loo_min_beta':float(min(x['beta_feature'] for x in loo)),
        'loo_max_beta':float(max(x['beta_feature'] for x in loo)),
        'loo_direction_reversals':int(sum(np.sign(x['beta_feature'])!=np.sign(beta) for x in loo)),
        'M0_converged':bool(m0['converged']),'M1_converged':bool(m1['converged']),
    }
    return result,pd.DataFrame(loo)


def calibration(y,p):
    lp=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    X=np.column_stack([np.ones(len(y)),lp])
    try:
        fit=firth_fit(X,y)
        return float(fit['beta'][0]),float(fit['beta'][1])
    except Exception:
        return np.nan,np.nan


def repeated_nested_cv(frame,outcome,n_repeats=200):
    frame=frame[['Subject',outcome,'CRS_pre','days_to_surgery',PRIMARY]].dropna().copy()
    y=frame[outcome].to_numpy(int)
    x0=frame[['CRS_pre','days_to_surgery']].copy(); x0['days_to_surgery']=np.log1p(x0.days_to_surgery)
    x1=x0.copy(); x1[PRIMARY]=frame[PRIMARY].to_numpy(float)
    minclass=int(np.bincount(y).min()); folds=min(4,minclass)
    if folds<2: return pd.DataFrame()
    outer=RepeatedStratifiedKFold(n_splits=folds,n_repeats=n_repeats,random_state=SEED)
    rows=[]
    for split,(train,test) in enumerate(outer.split(x0,y)):
        repeat=split//folds; inner_folds=min(3,int(np.bincount(y[train]).min()))
        preds={}
        for label,X in [('M0',x0),('M1',x1)]:
            pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(solver='liblinear',max_iter=2000))])
            if inner_folds>=2:
                inner=StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=SEED+split)
                fit=GridSearchCV(pipe,{'model__C':[.01,.1,1,10,100]},scoring='neg_log_loss',cv=inner,n_jobs=1)
            else:
                fit=pipe.set_params(model__C=1)
            fit.fit(X.iloc[train],y[train]); preds[label]=fit.predict_proba(X.iloc[test])[:,1]
        for j,idx in enumerate(test):
            rows.append({'repeat':repeat,'fold':split%folds,'Subject':frame.Subject.iloc[idx],
                         'y':int(y[idx]),'p_M0':float(preds['M0'][j]),'p_M1':float(preds['M1'][j])})
    pred=pd.DataFrame(rows)
    metrics=[]
    for repeat,g in pred.groupby('repeat'):
        row={'repeat':int(repeat),'n':len(g),'events':int(g.y.sum())}
        for label in ['M0','M1']:
            p=g[f'p_{label}'].to_numpy(); yy=g.y.to_numpy()
            row[f'{label}_brier']=brier_score_loss(yy,p)
            row[f'{label}_logloss']=log_loss(yy,p,labels=[0,1])
            row[f'{label}_auc']=roc_auc_score(yy,p)
            ci,cs=calibration(yy,p); row[f'{label}_calibration_intercept']=ci; row[f'{label}_calibration_slope']=cs
        row['delta_brier_M1_minus_M0']=row['M1_brier']-row['M0_brier']
        row['delta_logloss_M1_minus_M0']=row['M1_logloss']-row['M0_logloss']
        row['delta_auc_M1_minus_M0']=row['M1_auc']-row['M0_auc']
        metrics.append(row)
    return pd.DataFrame(metrics),pred


def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); q=np.empty(n); prev=1.0
    for rank in range(n-1,-1,-1):
        i=order[rank]; prev=min(prev,p[i]*n/(rank+1)); q[i]=prev
    return q


def main():
    OUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(SEED)
    qcd=json.load(open(STAGED/'INDEPENDENT_QC/stageD_independent_qc.json'))
    if qcd.get('status')!='pass': raise RuntimeError('Stage D independent QC not passed')
    lock=json.load(open(STAGEC/'STAGEC_TOPOLOGY_ANALYSIS_LOCK_ADDENDUM.json',encoding='utf-8'))
    if lock['patient_primary_feature']!='R_VPL_sensorimotor_preference': raise RuntimeError('Primary feature lock mismatch')
    feat=pd.read_csv(STAGED/'patient_fixed30_topology_features_outcome_blind.csv')
    clinical=pd.read_csv(MRI/'deltaCRS_imaging_cohort.csv')
    clinical=clinical.drop(columns=[c for c in ['ID','admission_id'] if c in clinical],errors='ignore')
    keep=['Subject','age','etiology_group','days_to_surgery','CRS_pre','CRS_6mo','diagnosis_6mo','died']
    data=feat.merge(clinical[keep],on='Subject',how='left',validate='one_to_one')
    data['TBI']=(data.etiology_group=='TBI').astype(int)
    data['favourable6']=(data.diagnosis_6mo>=3).astype(int)
    data['delta_CRS_6mo']=data.CRS_6mo-data.CRS_pre
    data['delta_ge3']=(data.delta_CRS_6mo>=3).astype(int)
    data['delta_ge2']=(data.delta_CRS_6mo>=2).astype(int)
    allowed=['Subject','legacy30_primary_cohort','strict27_sensitivity_cohort',
             'legacy30_primary_feature_analysis','strict27_primary_feature_analysis',
             'R_VPL_sensorimotor_n_available','R_VPL_othercortex_n_available',
             'R_VPL_cortex_n_available','R_VPL_sensorimotor_preference_available',
             'age','TBI','days_to_surgery',
             'CRS_pre','CRS_6mo','diagnosis_6mo','died','favourable6','delta_CRS_6mo','delta_ge3','delta_ge2',
             PRIMARY]+SECONDARY
    data[allowed].to_csv(OUT/'clinical_model_dataset_deidentified.csv',index=False)

    cohorts={'primary_n30':data.copy(),'strict_n27':data[data.strict27_sensitivity_cohort].copy()}
    outcomes=['favourable6','delta_ge3','delta_ge2']; results=[]; loo_all=[]
    for cname,frame in cohorts.items():
        for outcome in outcomes:
            res,loo=feature_test(frame,outcome,PRIMARY,5000 if outcome=='favourable6' else 2000,
                                 5000 if outcome=='favourable6' else 2000,rng)
            res['cohort']=cname; results.append(res); loo['cohort']=cname; loo['outcome']=outcome; loo_all.append(loo)
    primary=pd.DataFrame(results)
    primary.to_csv(OUT/'primary_feature_firth_permutation_bootstrap.csv',index=False)
    pd.concat(loo_all,ignore_index=True).to_csv(OUT/'primary_feature_leave_one_out.csv',index=False)

    secondary=[]
    for cname,frame in cohorts.items():
        cres=[]
        for feature in SECONDARY:
            res,_=feature_test(frame,'favourable6',feature,2000,1000,rng); res['cohort']=cname; cres.append(res)
        q=bh([x['permutation_p'] for x in cres])
        for x,qq in zip(cres,q): x['BH_FDR_q']=float(qq); secondary.append(x)
    pd.DataFrame(secondary).to_csv(OUT/'secondary_feature_family_fdr.csv',index=False)

    cv_summary=[]
    for cname,frame in cohorts.items():
        cv,pred=repeated_nested_cv(frame,'favourable6',200)
        cv['cohort']=cname; pred['cohort']=cname
        cv.to_csv(OUT/f'nested_cv_{cname}.csv',index=False)
        pred.to_csv(OUT/f'nested_cv_predictions_{cname}.csv',index=False)
        cv_summary.append({
            'cohort':cname,'repeats':len(cv),
            'median_delta_brier':float(cv.delta_brier_M1_minus_M0.median()),
            'proportion_M1_brier_better':float((cv.delta_brier_M1_minus_M0<0).mean()),
            'median_delta_logloss':float(cv.delta_logloss_M1_minus_M0.median()),
            'proportion_M1_logloss_better':float((cv.delta_logloss_M1_minus_M0<0).mean()),
            'median_delta_auc':float(cv.delta_auc_M1_minus_M0.median()),
            'proportion_M1_auc_better':float((cv.delta_auc_M1_minus_M0>0).mean()),
            'median_M0_auc':float(cv.M0_auc.median()),'median_M1_auc':float(cv.M1_auc.median()),
            'median_M0_brier':float(cv.M0_brier.median()),'median_M1_brier':float(cv.M1_brier.median()),
        })
    pd.DataFrame(cv_summary).to_csv(OUT/'nested_cv_summary.csv',index=False)

    mainrow=primary[(primary.cohort=='primary_n30')&(primary.outcome=='favourable6')].iloc[0]
    strictrow=primary[(primary.cohort=='strict_n27')&(primary.outcome=='favourable6')].iloc[0]
    cvrow=pd.DataFrame(cv_summary).query("cohort=='primary_n30'").iloc[0]
    topology_decision=json.load(open(STAGEC/'STAGEC_TOPOLOGY_GO_NOGO.json'))
    go={
        'stageC_normative_topology_go':bool(topology_decision['normative_topology_go']),
        'direction_hypothesized_positive':bool(mainrow.beta>0),
        'strict_cohort_direction_consistent':bool(strictrow.beta>0),
        'permutation_p_lt_0_05':bool(mainrow.permutation_p<.05),
        'bootstrap_same_direction_ge_0_80':bool(mainrow.bootstrap_same_direction>=.80),
        'loo_no_direction_reversal':bool(mainrow.loo_direction_reversals==0),
        'cv_brier_improves_ge_0_80_repeats':bool(cvrow.proportion_M1_brier_better>=.80),
    }
    summary={
        'status':'complete','primary_feature':PRIMARY,'primary_endpoint':'favourable6',
        'primary_source_cohort_n':int(len(cohorts['primary_n30'])),
        'strict_source_cohort_n':int(len(cohorts['strict_n27'])),
        'primary_analysis_n':int(mainrow['n']),
        'strict_analysis_n':int(strictrow['n']),
        'primary_endpoint_events_analysis':int(mainrow['events']),
        'primary_result':{k:(v.item() if hasattr(v,'item') else v) for k,v in mainrow.to_dict().items()},
        'nested_cv_primary':{k:(v.item() if hasattr(v,'item') else v) for k,v in cvrow.to_dict().items()},
        'predefined_go_criteria':go,
        'core_claim_go':bool(all(go.values())),
        'interpretation_boundary':(
            'Association/prognostic increment only; no causal treatment-effect, mediation, '
            'content-specificity, or deployable-biomarker claim.'),
        'stageD_feature_sha256':sha256(STAGED/'patient_fixed30_topology_features_outcome_blind.csv'),
        'stageC_lock_sha256':sha256(STAGEC/'STAGEC_TOPOLOGY_ANALYSIS_LOCK_ADDENDUM.json'),
        'analysis_script_sha256':sha256(Path(__file__).resolve()),
        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (OUT/'stageE_clinical_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    decision='GO' if summary['core_claim_go'] else 'NO-GO / DOWNGRADE'
    (OUT/'STAGEE_REPORT.md').write_text(
        '# Stage E — locked clinical topology models\n\n'
        f'- Decision: **{decision}** under the fully prespecified criteria.\n'
        f"- Source cohorts n={summary['primary_source_cohort_n']}/{summary['strict_source_cohort_n']}; lesion-support-complete analyses n={summary['primary_analysis_n']}/{summary['strict_analysis_n']} (primary events={summary['primary_endpoint_events_analysis']}).\n"
        f"- Primary feature OR={mainrow.OR:.3f} (Wald 95% CI {mainrow.OR95_low:.3f}–{mainrow.OR95_high:.3f}); residual-permutation P={mainrow.permutation_p:.4g}.\n"
        f"- Bootstrap same-direction proportion={mainrow.bootstrap_same_direction:.3f}; LOO direction reversals={int(mainrow.loo_direction_reversals)}.\n"
        f"- Nested-CV median Brier M0/M1={cvrow.median_M0_brier:.3f}/{cvrow.median_M1_brier:.3f}; M1 improves in {cvrow.proportion_M1_brier_better:.3f} of repeats.\n"
        '- Clinical data remain total-score/state summaries; item-level content specificity is not tested.\n',encoding='utf-8')
    lines=[]
    for path in sorted(OUT.glob('*')):
        if path.name=='SHA256SUMS.txt' or not path.is_file(): continue
        lines.append(f'{sha256(path)}  {path.name}')
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
