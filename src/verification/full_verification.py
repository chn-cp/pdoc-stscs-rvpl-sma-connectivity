from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from core_verification import OUT, ROOT, LEGACY, FEATURE, SEED, BLUE, DARK, GREY, save_json, savefig, sha, emit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'primary'))
import clinical_models as core


def z(v):
    x=np.asarray(v,float)
    return (x-x.mean())/x.std(ddof=1)


def design(d, cov):
    return np.column_stack([np.ones(len(d))]+[d[c].to_numpy(float) if c in ['TBI','baseline_MCSminus'] else z(d[c]) for c in cov]+[d[FEATURE].to_numpy(float)])


def fit(d,cov):
    used=d.dropna(subset=[FEATURE,'favourable6']+cov).copy()
    X=design(used,cov);y=used.favourable6.to_numpy(int)
    f=core.firth_fit(X,y);b=float(f['beta'][-1]);se=float(f['se'][-1])
    return {'n':len(used),'events':int(y.sum()),'beta':b,'OR':float(np.exp(b)),
            'OR95_low':float(np.exp(b-1.96*se)),'OR95_high':float(np.exp(b+1.96*se)),
            'wald_p':float(2*stats.norm.sf(abs(b/se))),'se':se,
            'converged':bool(f['converged']),'iterations':f['iterations'],
            'design_condition_number':float(np.linalg.cond(X))},used,X,f


def bh(p):
    p=np.asarray(p,float);ix=np.argsort(p);q=np.empty(len(p));value=1.
    for j in range(len(p)-1,-1,-1):
        i=ix[j];value=min(value,p[i]*len(p)/(j+1));q[i]=value
    return q


def family():
    base=OUT/'00_inputs/FCmat_work'
    d=pd.read_csv(OUT/'00_inputs/master_imaging_cohort_n30.csv')
    names=pd.read_csv(base/'node_info.csv').name.tolist()
    pm={p.stem:np.load(p) for p in (base/'patient_mats').glob('*.npy')}
    hm=[np.load(p) for p in (base/'HC_mats').glob('*.npy')]
    primary=pd.read_csv(OUT/'00_inputs/historical/RVPL_SMA_primary_endpoints.csv').query("cohort=='main_source_n30' and outcome=='favourable6'").iloc[0]
    results=[];check=[]
    for family,tag,key in [('seed_to_SMA','thalamic_seed','seed'),('RVPL_to_cortex','cortical_target','target')]:
        old=pd.read_csv(OUT/f'00_inputs/peer_review/{tag}_family_unified_p.csv')
        rows=[]
        for r in old.to_dict('records'):
            a,b=(names.index(r[key]),names.index('SMA')) if key=='seed' else (names.index('R_VPL'),names.index(r[key]))
            reference=np.array([m[a,b] for m in hm]);reference=reference[np.isfinite(reference)]
            mean=reference.mean();sd=reference.std(ddof=1)
            assert len(reference)==int(r['HC_n'])==26
            assert np.isclose(mean,r['HC_mean'],atol=1e-10) and np.isclose(sd,r['HC_sd'],atol=1e-10)
            dd=d.copy();dd[FEATURE]=dd.Subject.map({s:(m[a,b]-mean)/sd for s,m in pm.items()})
            f,used,_,_=fit(dd,['CRS_pre','log1p_days'])
            for col in ['n','events','OR','OR95_low','OR95_high','wald_p']:
                check.append({'family':family,'edge':r[key],'quantity':col,'recomputed':f[col],
                              'archived':r[col],'match':bool(np.isclose(f[col],r[col],rtol=1e-7,atol=1e-9))})
            assert all(x['match'] for x in check),check[-6:]
            focus=(key=='seed' and r[key]=='R_VPL') or (key=='target' and r[key]=='SMA')
            rows.append({**f,'family':family,'label':r[key],'HC_n':len(reference),'HC_mean':mean,'HC_sd':sd,
                         'permutation_p':float(primary.permutation_p if focus else r['permutation_p']),
                         'permutations':5000 if focus else int(r['n_permutations']),
                         'focused':focus,'sample_subjects':';'.join(used.Subject),
                         'standardization':'per-edge 26-person HC sample SD',
                         'covariates':'baseline CRS-R + log1p(injury-to-stimulation days)'})
        dd=pd.DataFrame(rows);dd['BH_q']=bh(dd.permutation_p);dd['rank_by_permutation_p']=dd.permutation_p.rank(method='min').astype(int)
        dd=dd.sort_values('permutation_p')
        dd.to_csv(OUT/f'03_anatomical_competition/{family}_verified.csv',index=False)
        results.append(dd)
    pd.DataFrame(check).to_csv(OUT/'03_anatomical_competition/family_point_estimate_QC.csv',index=False)
    fig=plt.figure(figsize=(12.5,10.0));gs=fig.add_gridspec(1,2,width_ratios=[1,1.35],wspace=.4)
    for j,dd in enumerate(results):
        ax=fig.add_subplot(gs[0,j]);pos=np.arange(len(dd))
        for k,r in enumerate(dd.itertuples()):
            color=BLUE if r.focused else DARK
            ax.errorbar(r.OR,k,xerr=[[r.OR-r.OR95_low],[r.OR95_high-r.OR]],fmt='o',color=color,
                        ms=6 if r.focused else 4,capsize=2,lw=1.5 if r.focused else .9)
            ax.text(1.03,k,f'{r.BH_q:.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=8.5,color=color,
                    fontweight='bold' if r.focused else 'normal')
        labels=[f'{r.label.replace("_","-")} (n={r.n})' for r in dd.itertuples()]
        ax.set(yticks=pos,yticklabels=labels,xscale='log',xlabel='Odds ratio per healthy-reference SD')
        for t,focus in zip(ax.get_yticklabels(),dd.focused):
            if focus:t.set_color(BLUE);t.set_fontweight('bold')
        ax.invert_yaxis();ax.axvline(1,c=GREY,lw=.8,ls='--');ax.grid(axis='x',alpha=.13)
        ax.set_title(('A  Subcortical seeds to SMA' if j==0 else 'B  R-VPL to cortical targets'),loc='left',fontweight='bold',pad=25)
        ax.text(1.03,1.015,'BH q',transform=ax.transAxes,fontsize=9,fontweight='bold')
        if j==0:ax.set_ylim(len(dd)-.4,-.8)
    fig.text(.06,.035,'Firth logistic models: baseline CRS-R and log1p injury-to-stimulation time. Bars: Wald 95% CI.\n'
             'Ordered by frozen permutation P; edgewise sample availability differs. BH correction is within each family.',fontsize=9,color=DARK)
    fig.subplots_adjust(left=.11,right=.91,top=.9,bottom=.13)
    savefig(fig,'Fig_B1_anatomical_competition')
    emit('B1 all 30 edge fits, reference moments and BH corrections verified.')


def continuous(d):
    x=d[FEATURE].to_numpy();y=(d.CRS_6mo-d.CRS_pre).to_numpy()
    rows=[]
    rng=np.random.default_rng(SEED+200)
    for kind,method in [('Pearson',stats.pearsonr),('Spearman',stats.spearmanr)]:
        res=method(x,y);obs=float(res.statistic)
        # A separate null is fixed for each estimator; no choosing between tests.
        null=np.array([method(x,rng.permutation(y)).statistic for _ in range(5000)])
        p=(1+np.sum(np.abs(null)>=abs(obs)-1e-12))/5001
        rows.append({'analysis':kind,'n':len(d),'coefficient':obs,'asymptotic_p':res.pvalue,
                     'outcome_label_permutation_p':p,'n_permutations':5000,'seed_stream':SEED+200})
    pd.DataFrame(rows).to_csv(OUT/'05_master_results/continuous_change_associations.csv',index=False)
    emit('Continuous change '+str(rows))
    return rows


def ordinal_optional(d):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    predictors=pd.DataFrame({'edge':d[FEATURE].to_numpy(),'baseline_CRSR_z':z(d.CRS_pre),
                             'log1p_timing_z':z(d.log1p_days)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model=OrderedModel(d.final_state_code.to_numpy(int),predictors,distr='logit')
        result=model.fit(method='bfgs',maxiter=2000,disp=False)
    ci=result.conf_int();beta=float(result.params['edge']);se=float(result.bse['edge'])
    probs=result.model.predict(result.params)
    cut=[]
    for k in range(3):
        dd=d.copy();dd['favourable6']=(dd.final_state_code>k).astype(int)
        r,_,_,_=fit(dd,['CRS_pre','log1p_days'])
        cut.append({'cutpoint':k,'outcome':f'state code>{k}',**r})
    pd.DataFrame(cut).to_csv(OUT/'01_ordered_state/ordinal_cutpoint_firth_diagnostics.csv',index=False)
    info={'n':len(d),'events_by_category':d.final_state_code.value_counts().sort_index().to_dict(),
          'parameters':len(result.params),'converged':bool(result.mle_retvals.get('converged')),
          'edge_beta':beta,'edge_OR':float(np.exp(beta)),'edge_OR95_low':float(np.exp(ci.loc['edge',0])),
          'edge_OR95_high':float(np.exp(ci.loc['edge',1])),'edge_wald_p':float(result.pvalues['edge']),
          'hessian_eigenvalues':np.linalg.eigvalsh(-model.hessian(result.params)).tolist(),
          'maximum_fitted_category_probability':float(probs.max()),'warnings':[str(x.message) for x in w],
          'timing_transform':'log1p days, matching historical primary; not unannounced log(days)',
          'PO_diagnostic':'threshold-specific Firth coefficients are descriptive only; no adequately powered formal PO acceptance test with n=26',
          'usage':'exploratory supplement only; no claim that proportional odds has been established; rank trend remains principal A1 analysis'}
    save_json(OUT/'01_ordered_state/ordinal_model_exploratory.json',info)
    (OUT/'01_ordered_state/ordinal_model_summary.txt').write_text(str(result.summary()),encoding='utf-8')
    emit('Ordinal exploratory '+json.dumps(info))


def robustness():
    d=pd.read_csv(OUT/'00_inputs/master_imaging_cohort_n30.csv')
    old=pd.read_csv(OUT/'00_inputs/historical/RVPL_SMA_primary_endpoints.csv')
    sensitivity=pd.read_csv(OUT/'00_inputs/historical/RVPL_SMA_covariate_sensitivity.csv')
    targeted=pd.read_csv(OUT/'00_inputs/peer_review/focused_edge_targeted_sensitivity.csv')
    specs=[]
    for cohort,label in [('main_source_n30','main'),('strict_source_n27','strict')]:
        frame=d[d.strict_available].copy() if label=='strict' else d.copy()
        row=old.query('cohort==@cohort and outcome=="favourable6"').iloc[0]
        specs.append((label,frame,['CRS_pre','log1p_days'],row,'historical primary'))
        for r in sensitivity.query('cohort==@cohort and specification!="base"').to_dict('records'):
            specs.append((label+'_'+r['specification'],frame,r['covariates'].split('+'),pd.Series(r),'historical sensitivity'))
    for name,cov in [('lesion',['CRS_pre','log1p_days','log1p_lesion_vol_ml']),('baseline_diagnosis',['baseline_MCSminus','log1p_days'])]:
        row=targeted[targeted.model=='+'.join(cov)].iloc[0]
        row=row.rename({'OR_feature':'OR','wald_CI_low':'OR95_low','wald_CI_high':'OR95_high',
                        'predictor_residual_permutation_p':'permutation_p'})
        specs.append((name,d,cov,row,'targeted peer-review sensitivity'))
    rows=[]
    for name,frame,cov,prior,source in specs:
        result,used,X,f=fit(frame,cov)
        match=all(np.isclose(result[k],prior[k],rtol=1e-7,atol=1e-9) for k in ['n','events','OR','OR95_low','OR95_high','wald_p'])
        assert match,(name,result,prior.to_dict())
        rows.append({'model':name,**result,'covariates':'+'.join(cov),'frozen_permutation_p':prior['permutation_p'],
                     'frozen_permutations':int(prior['n_permutations']),'point_estimate_matches_archived':match,'frozen_source':source})
    pd.DataFrame(rows).to_csv(OUT/'05_master_results/robustness_verified.csv',index=False)
    emit('C: all primary/strict/age/TBI/motion/support/lesion/diagnosis fits match sealed estimates.')
    return d


def resampling(d):
    # New audit run; frozen primary P stays frozen. Explicitly record every fit.
    base=OUT/'05_master_results'
    rows=[];bootrows=[];loolist=[];nullrows=[]
    for label,frame in [('main',d),('strict',d[d.strict_available])]:
        result,used,X,f=fit(frame,['CRS_pre','log1p_days']);y=used.favourable6.to_numpy(int)
        rng=np.random.default_rng(SEED+(400 if label=='main' else 500))
        for b in range(5000):
            ix=rng.integers(0,len(used),len(used));bf=core.firth_fit(X[ix],y[ix])
            bootrows.append({'cohort':label,'iteration':b,'beta':bf['beta'][-1],
                             'converged':bool(bf['converged']),'distinct_outcome_classes':len(np.unique(y[ix]))})
            if (b+1)%1000==0:emit(f'C: {label} bootstrap {b+1}/5000')
        for k,subject in enumerate(used.Subject):
            keep=np.arange(len(used))!=k;lf=core.firth_fit(X[keep],y[keep])
            loolist.append({'cohort':label,'removed_Subject':subject,'beta':lf['beta'][-1],
                            'converged':bool(lf['converged'])})
        if label=='main':
            rngp=np.random.default_rng(SEED+600);nuis=X[:,:-1]
            pred=nuis@np.linalg.lstsq(nuis,X[:,-1],rcond=None)[0];res=X[:,-1]-pred
            for b in range(5000):
                ff=core.firth_fit(np.column_stack([nuis,pred+rngp.permutation(res)]),y)
                nullrows.append({'iteration':b,'beta':ff['beta'][-1],'converged':bool(ff['converged'])})
                if (b+1)%1000==0:emit(f'C: new audit permutation {b+1}/5000 (not replacing frozen primary P)')
    b=pd.DataFrame(bootrows);l=pd.DataFrame(loolist);p=pd.DataFrame(nullrows)
    b.to_csv(base/'bootstrap_verification.csv',index=False);l.to_csv(base/'leave_one_out.csv',index=False)
    p.to_csv(base/'predictor_residual_permutation.csv',index=False)
    old=pd.read_csv(OUT/'00_inputs/historical/RVPL_SMA_primary_endpoints.csv')
    for label in ['main','strict']:
        bb=b[(b.cohort==label)&b.converged&(b.distinct_outcome_classes==2)]
        ll=l[l.cohort==label]; prior=old.query('cohort==@label+"_source_n30"') if False else old[(old.cohort==('main_source_n30' if label=='main' else 'strict_source_n27'))&(old.outcome=='favourable6')].iloc[0]
        rows.append({'cohort':label,'bootstrap_attempted':5000,'bootstrap_valid':len(bb),'bootstrap_same_direction_new':float((bb.beta>0).mean()),
                     'bootstrap_beta_q025':bb.beta.quantile(.025),'bootstrap_beta_q975':bb.beta.quantile(.975),
                     'bootstrap_same_direction_frozen':prior.bootstrap_same_direction,
                     'LOO_n':len(ll),'LOO_all_converged':bool(ll.converged.all()),'LOO_direction_reversals':int((ll.beta<=0).sum()),
                     'new_audit_permutation_p':float((1+(p.beta.abs()>=prior.beta-1e-12).sum())/5001) if label=='main' else np.nan,
                     'frozen_permutation_p':prior.permutation_p,'new_resampling_seed':SEED+(400 if label=='main' else 500),
                     'primary_permutation_new_seed':SEED+600 if label=='main' else np.nan})
    pd.DataFrame(rows).to_csv(base/'resampling_audit_summary.csv',index=False)
    emit('C resampling audit complete '+str(rows))


def cv_scores():
    summaries=[];check=[]
    for label in ['main_source_n30','strict_source_n27']:
        pp=pd.read_csv(OUT/f'00_inputs/historical/RVPL_SMA_nested_CV_predictions_{label}.csv')
        prior=pd.read_csv(OUT/f'00_inputs/historical/RVPL_SMA_nested_CV_{label}.csv')
        allrows=[]
        for rep,g in pp.groupby('repeat'):
            assert not g.Subject.duplicated().any()
            row={'repeat':rep,'n':len(g),'events':int(g.y.sum())}
            for m in ['M0','M1']:
                prob=g['p_'+m]; y=g.y
                row.update({m+'_auc':roc_auc_score(y,prob),m+'_brier':brier_score_loss(y,prob),m+'_logloss':log_loss(y,prob,labels=[0,1])})
            allrows.append(row)
        vals=pd.DataFrame(allrows)
        for col in ['M0_auc','M1_auc','M0_brier','M1_brier','M0_logloss','M1_logloss']:
            ok=np.allclose(vals[col],prior[col],rtol=1e-8,atol=1e-10);assert ok
            check.append({'cohort':label,'metric':col,'all_200_repeat_scores_match':bool(ok)})
        vals.to_csv(OUT/f'05_master_results/CV_scores_recomputed_{label}.csv',index=False)
        for metric in ['auc','brier','logloss']:
            summaries.append({'cohort':label,'metric':metric,'repeats':len(vals),'n':int(vals.n.iloc[0]),
                              'M0_median':vals['M0_'+metric].median(),'M1_median':vals['M1_'+metric].median(),
                              'median_paired_delta_M1_minus_M0':(vals['M1_'+metric]-vals['M0_'+metric]).median(),
                              'fraction_M1_better':float(((vals['M1_'+metric]>vals['M0_'+metric]) if metric=='auc' else (vals['M1_'+metric]<vals['M0_'+metric])).mean()),
                              'verification':'metrics recalculated from saved out-of-fold predictions, not new CV fitting',
                              'claim_ceiling':'same-cohort internal information; outcome-informed edge selection occurred outside CV'})
    pd.DataFrame(summaries).to_csv(OUT/'05_master_results/CV_summary_recomputed.csv',index=False)
    pd.DataFrame(check).to_csv(OUT/'logs/CV_metric_QC.csv',index=False)
    # Diagnosis-based clinical comparator: recalculate from its archived predictions.
    pp=pd.read_csv(OUT/'00_inputs/peer_review/clinical_baseline_nested_cv_predictions.csv')
    rows=[]
    for rep,g in pp.groupby('repeat'):
        assert not g.Subject.duplicated().any()
        for m in ['DIAG','DIAG_EDGE']:
            rows.append({'repeat':rep,'model':m,'auc':roc_auc_score(g.y,g['p_'+m]),
                         'brier':brier_score_loss(g.y,g['p_'+m]),'logloss':log_loss(g.y,g['p_'+m],labels=[0,1])})
    pd.DataFrame(rows).to_csv(OUT/'05_master_results/diagnosis_CV_recomputed.csv',index=False)
    emit('C CV: all saved OOF AUC/Brier/log loss reproduced; no model-selection claims upgraded.')


def icc(a):
    n,k=a.shape;grand=a.mean();row=a.mean(axis=1);col=a.mean(axis=0)
    msr=k*np.sum((row-grand)**2)/(n-1)
    msc=n*np.sum((col-grand)**2)/(k-1)
    mse=np.sum((a-row[:,None]-col[None,:]+grand)**2)/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)


def gsp():
    src=ROOT/'work/stageb_server_snapshot'
    d=pd.read_csv(src/'gsp_vpl_sma_subjectmeans_by_length.csv')
    a=d[d.length==120].pivot(index='subject',columns='run',values='R_VPL_SMA_z')
    paired=a.dropna().to_numpy();assert len(paired)==746
    value=icc(paired);corr=stats.pearsonr(paired[:,0],paired[:,1])
    frozen=pd.read_csv(src/'INDEPENDENT_QC/stageB_reliability_independent_qc.csv').query('length==120').iloc[0]
    assert np.isclose(value,frozen.ICC_2_1) and np.isclose(corr.statistic,frozen.pearson_r)
    rng=np.random.default_rng(SEED+700)
    boots=np.array([icc(paired[rng.integers(0,len(paired),len(paired))]) for _ in range(5000)])
    a.reset_index().to_csv(OUT/'00_inputs/gsp_paired_run120_values.csv',index=False)
    pd.DataFrame({'ICC_2_1':boots}).to_csv(OUT/'05_master_results/GSP_ICC_bootstrap_new_audit.csv',index=False)
    result={'paired_n':len(paired),'run1_n':int(a[1].notna().sum()),'run2_n':int(a[2].notna().sum()),
            'ICC_2_1':value,'Pearson_r':corr.statistic,
            'frozen_bootstrap_CI_low':frozen.ICC_2_1_bootstrap95_low,
            'frozen_bootstrap_CI_high':frozen.ICC_2_1_bootstrap95_high,
            'new_bootstrap_CI_low':np.quantile(boots,.025),'new_bootstrap_CI_high':np.quantile(boots,.975),
            'new_bootstraps':5000,'new_seed':SEED+700,
            'point_estimates_match_archived':True,'source':str(src/'gsp_vpl_sma_subjectmeans_by_length.csv')}
    save_json(OUT/'05_master_results/GSP_reliability_verified.json',result)
    emit('C GSP paired-run ICC and r reproduced.')


if __name__=='__main__':
    family()
    d=robustness()
    focus=d[d[FEATURE].notna()].copy()
    continuous(focus)
    ordinal_optional(focus)
    cv_scores()
    gsp()
    resampling(d)
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','nibabel','nilearn','scikit-learn','statsmodels','PyMuPDF']}
    versions['python']=sys.version;save_json(OUT/'logs/software_versions.json',versions)
