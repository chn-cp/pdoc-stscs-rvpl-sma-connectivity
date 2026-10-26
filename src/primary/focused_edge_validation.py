#!/usr/bin/env python3
"""Focused R_VPL-SMA validation and robustness analyses."""
from pathlib import Path
import hashlib
import json
import time
import warnings
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
ROOT = Path(os.environ.get("RVPL_SMA_PROJECT_ROOT", Path.cwd()))
BASE = Path(os.environ.get("RVPL_SMA_DISCOVERY_DIR", ROOT / "restricted_inputs" / "discovery_analysis"))
MRI = ROOT / "restricted_inputs" / "MRI"
OUT = ROOT / "outputs" / "focused_edge_validation"
LOCK = OUT / 'focused_edge_validation_lock.json'
SEED = 20260920
PRIMARY = 'R_VPL_SMA_z_local26_z'
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clinical_models as core


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_z(series):
    x = pd.to_numeric(series, errors='coerce').astype(float)
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f'Cannot standardize {getattr(series, "name", "variable")}')
    return (x - x.mean()) / sd


def make_design(frame, covariates, feature=None):
    cols = [np.ones(len(frame))]
    names = ['Intercept']
    for cov in covariates:
        x = frame[cov].astype(float)
        if cov == 'TBI':
            cols.append(x.to_numpy()); names.append(cov)
        else:
            cols.append(safe_z(x).to_numpy()); names.append(f'{cov}_z')
    if feature:
        cols.append(frame[feature].astype(float).to_numpy()); names.append(feature)
    return np.column_stack(cols), names


def feature_test(frame, outcome, feature, covariates, n_perm, n_boot, rng, do_loo=True):
    required = ['Subject', outcome, feature] + covariates
    d = frame[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = d[outcome].to_numpy(int)
    if len(d) < 15 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 4:
        return {'status': 'not_estimable', 'outcome': outcome, 'feature': feature,
                'n': len(d), 'events': int(y.sum()) if len(d) else 0}, pd.DataFrame()
    X0, _ = make_design(d, covariates)
    X1, names = make_design(d, covariates, feature)
    idx = names.index(feature)
    m0 = core.firth_fit(X0, y); m1 = core.firth_fit(X1, y)
    beta = float(m1['beta'][idx]); se = float(m1['se'][idx]); z = beta/se if se > 0 else np.nan
    coef = np.linalg.lstsq(X0, d[feature].to_numpy(float), rcond=None)[0]
    fitted = X0 @ coef; resid = d[feature].to_numpy(float) - fitted
    null = np.empty(n_perm, float)
    for i in range(n_perm):
        fp = fitted + rng.permutation(resid)
        null[i] = core.firth_fit(np.column_stack([X0, fp]), y)['beta'][-1]
    pperm = float((1 + np.sum(np.abs(null) >= abs(beta))) / (n_perm + 1))
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(d), len(d))
        mb = core.firth_fit(X1[ix], y[ix])
        if mb['converged'] and np.isfinite(mb['beta'][idx]):
            boots.append(float(mb['beta'][idx]))
    loo = []
    if do_loo:
        for i in range(len(d)):
            keep = np.arange(len(d)) != i
            ml = core.firth_fit(X1[keep], y[keep])
            loo.append({'removed_Subject': d.Subject.iloc[i],
                        'beta_feature': float(ml['beta'][idx]),
                        'converged': bool(ml['converged'])})
    boots = np.asarray(boots, float)
    result = {
        'status': 'ok', 'outcome': outcome, 'feature': feature,
        'covariates': '+'.join(covariates), 'n': int(len(d)), 'events': int(y.sum()),
        'beta': beta, 'OR': float(np.exp(beta)), 'se': se,
        'wald_p': float(2*stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        'OR95_low': float(np.exp(beta-1.96*se)), 'OR95_high': float(np.exp(beta+1.96*se)),
        'permutation_p': pperm, 'n_permutations': int(n_perm),
        'bootstrap_n_valid': int(len(boots)),
        'bootstrap_beta_low': float(np.quantile(boots, .025)) if len(boots) else np.nan,
        'bootstrap_beta_high': float(np.quantile(boots, .975)) if len(boots) else np.nan,
        'bootstrap_same_direction': float(np.mean(np.sign(boots) == np.sign(beta))) if len(boots) else np.nan,
        'loo_min_beta': float(min(x['beta_feature'] for x in loo)) if loo else np.nan,
        'loo_max_beta': float(max(x['beta_feature'] for x in loo)) if loo else np.nan,
        'loo_direction_reversals': int(sum(np.sign(x['beta_feature']) != np.sign(beta) for x in loo)) if loo else np.nan,
        'M0_pll': float(m0['pll']), 'M1_pll': float(m1['pll']),
        'delta_penalized_ll': float(m1['pll']-m0['pll']),
        'M0_converged': bool(m0['converged']), 'M1_converged': bool(m1['converged']),
        'design_condition_number': float(np.linalg.cond(X1)),
    }
    return result, pd.DataFrame(loo)


def bh_fdr(values):
    values = np.asarray(values, float); q = np.full(len(values), np.nan)
    good = np.isfinite(values)
    p = values[good]; n = len(p)
    if not n: return q
    order = np.argsort(p); qq = np.empty(n); prev = 1.0
    for rank in range(n-1, -1, -1):
        i = order[rank]; prev = min(prev, p[i]*n/(rank+1)); qq[i] = prev
    q[good] = qq
    return q


def load_matrices(directory, order_file):
    order = pd.read_csv(order_file).iloc[:, 0].astype(str).tolist()
    mats = {s: np.asarray(np.load(directory/f'{s}.npy'), float) for s in order}
    return order, mats


def edge_standardized(patient_mats, hc_mats, a, b):
    hc = np.asarray([m[a, b] for m in hc_mats.values()], float)
    hc = hc[np.isfinite(hc)]
    mean = float(hc.mean()); sd = float(hc.std(ddof=1))
    if len(hc) < 20 or not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f'Invalid HC reference for edge {a}-{b}')
    return {s: ((m[a, b]-mean)/sd if np.isfinite(m[a, b]) else np.nan)
            for s, m in patient_mats.items()}, mean, sd, len(hc)


def repeated_nested_cv(frame, outcome, feature, n_repeats=200):
    d = frame[['Subject', outcome, 'CRS_pre', 'log1p_days', feature]].dropna().copy()
    y = d[outcome].to_numpy(int)
    x0 = d[['CRS_pre', 'log1p_days']].copy()
    x1 = x0.copy(); x1[feature] = d[feature].to_numpy(float)
    folds = min(4, int(np.bincount(y).min()))
    outer = RepeatedStratifiedKFold(n_splits=folds, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for split, (train, test) in enumerate(outer.split(x0, y)):
        repeat = split//folds; inner_folds = min(3, int(np.bincount(y[train]).min()))
        pred = {}
        for label, X in [('M0', x0), ('M1', x1)]:
            pipe = Pipeline([('scale', StandardScaler()),
                             ('model', LogisticRegression(solver='liblinear', max_iter=3000))])
            if inner_folds >= 2:
                inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED+split)
                fit = GridSearchCV(pipe, {'model__C': [.01, .1, 1, 10, 100]},
                                   scoring='neg_log_loss', cv=inner, n_jobs=1)
            else:
                fit = pipe.set_params(model__C=1)
            fit.fit(X.iloc[train], y[train]); pred[label] = fit.predict_proba(X.iloc[test])[:, 1]
        for j, idx in enumerate(test):
            rows.append({'repeat': repeat, 'fold': split % folds, 'Subject': d.Subject.iloc[idx],
                         'y': int(y[idx]), 'p_M0': float(pred['M0'][j]), 'p_M1': float(pred['M1'][j])})
    predictions = pd.DataFrame(rows)
    metrics = []
    for repeat, g in predictions.groupby('repeat'):
        row = {'repeat': int(repeat), 'n': len(g), 'events': int(g.y.sum())}
        for label in ['M0', 'M1']:
            yy = g.y.to_numpy(); pp = g[f'p_{label}'].to_numpy()
            row[f'{label}_brier'] = brier_score_loss(yy, pp)
            row[f'{label}_logloss'] = log_loss(yy, pp, labels=[0, 1])
            row[f'{label}_auc'] = roc_auc_score(yy, pp)
        row['delta_brier_M1_minus_M0'] = row['M1_brier']-row['M0_brier']
        row['delta_logloss_M1_minus_M0'] = row['M1_logloss']-row['M0_logloss']
        row['delta_auc_M1_minus_M0'] = row['M1_auc']-row['M0_auc']
        metrics.append(row)
    return pd.DataFrame(metrics), predictions, d


def availability_audit(data):
    rows = []
    available = data[PRIMARY].notna()
    table = pd.crosstab(available, data.favourable6).reindex(index=[False, True], columns=[0, 1], fill_value=0)
    odds, p = stats.fisher_exact(table.to_numpy())
    rows.append({'variable': 'favourable6', 'available_n': int(available.sum()),
                 'unavailable_n': int((~available).sum()),
                 'available_summary': float(data.loc[available, 'favourable6'].mean()),
                 'unavailable_summary': float(data.loc[~available, 'favourable6'].mean()),
                 'test': 'Fisher exact', 'p': float(p), 'statistic': float(odds)})
    for var in ['CRS_pre', 'days_to_surgery', 'age', 'meanFD_retained']:
        a = data.loc[available, var].dropna(); b = data.loc[~available, var].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided') if len(b) else (np.nan, np.nan)
        rows.append({'variable': var, 'available_n': len(a), 'unavailable_n': len(b),
                     'available_summary': float(a.median()), 'unavailable_summary': float(b.median()) if len(b) else np.nan,
                     'test': 'Mann-Whitney U', 'p': float(p), 'statistic': float(stat)})
    return pd.DataFrame(rows)


def main():
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    lock = json.load(open(LOCK, encoding='utf-8'))
    rng = np.random.default_rng(SEED)
    node = pd.read_csv(MRI/'FCmat_work/node_info.csv', encoding='utf-8-sig')
    names = node['name'].astype(str).tolist(); idx = {n: i for i, n in enumerate(names)}
    if names != ['L.M1','R.M1','SMA','L.V1','R.V1','L.V4','R.V4','L.A1','R.A1','MCC','aMPFC','PCC','L.LatP','R.LatP','DMPFC','L.PFC','R.PFC','L.SPC','R.SPC','L.AIns','R.AIns','dACC','L_CL','R_CL','L_VPL','R_VPL','L_PTg','R_PTg','L_mRt','R_mRt']:
        raise RuntimeError('Unexpected node order')
    patient_order, patient_mats = load_matrices(MRI/'FCmat_work/patient_mats', MRI/'FCmat_work/patient_order.csv')
    _, hc_mats = load_matrices(MRI/'FCmat_work/HC_mats', MRI/'FCmat_work/HC_order.csv')
    edge, hc_mean, hc_sd, hc_n = edge_standardized(patient_mats, hc_mats, idx['R_VPL'], idx['SMA'])

    features = pd.read_csv(BASE/'STAGED_PATIENT_FIXED30_PROJECTION/patient_fixed30_topology_features_outcome_blind.csv')
    clinical = pd.read_csv(MRI/'deltaCRS_imaging_cohort.csv')
    keep = ['Subject','age','etiology_group','days_to_surgery','CRS_pre','CRS_6mo','diagnosis_6mo']
    data = features.merge(clinical[keep], on='Subject', how='left', validate='one_to_one')
    data[PRIMARY] = data.Subject.map(edge)
    prior = pd.to_numeric(features.R_VPL_SMA_z_local26_z, errors='coerce')
    if not np.allclose(data[PRIMARY], prior, equal_nan=True, atol=1e-10):
        raise RuntimeError('Recomputed historical feature differs from sealed Stage D feature')
    data['TBI'] = data.etiology_group.astype(str).str.upper().eq('TBI').astype(int)
    data['log1p_days'] = np.log1p(data.days_to_surgery.astype(float))
    data['favourable6'] = (data.diagnosis_6mo >= 3).astype(int)
    data['delta_CRS_6mo'] = data.CRS_6mo-data.CRS_pre
    data['delta_ge3'] = (data.delta_CRS_6mo >= 3).astype(int)
    data['delta_ge2'] = (data.delta_CRS_6mo >= 2).astype(int)
    data['primary_available'] = data[PRIMARY].notna()
    data['strict_available'] = data.strict27_sensitivity_cohort.astype(bool) & data.primary_available
    analysis_cols = ['Subject','age','TBI','days_to_surgery','log1p_days','CRS_pre','CRS_6mo','diagnosis_6mo',
                     'favourable6','delta_CRS_6mo','delta_ge3','delta_ge2','meanFD_retained',
                     'R_VPL_cortex_n_available','strict27_sensitivity_cohort','primary_available','strict_available',PRIMARY]
    data[analysis_cols].to_csv(OUT/'RVPL_SMA_analysis_dataset_deidentified.csv', index=False)

    cohorts = {'main_source_n30': data.copy(),
               'strict_source_n27': data[data.strict27_sensitivity_cohort.astype(bool)].copy()}
    base_cov = ['CRS_pre','log1p_days']
    primary_rows = []; loo_rows = []
    for cname, frame in cohorts.items():
        for endpoint in ['favourable6','delta_ge3','delta_ge2']:
            res, loo = feature_test(frame, endpoint, PRIMARY, base_cov,
                                    5000 if endpoint == 'favourable6' else 2000,
                                    5000 if endpoint == 'favourable6' else 2000,
                                    rng, do_loo=True)
            res['cohort'] = cname; primary_rows.append(res)
            if len(loo):
                loo['cohort'] = cname; loo['outcome'] = endpoint; loo_rows.append(loo)
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(OUT/'RVPL_SMA_primary_endpoints.csv', index=False)
    pd.concat(loo_rows, ignore_index=True).to_csv(OUT/'RVPL_SMA_leave_one_out.csv', index=False)

    specs = {
        'base': base_cov,
        'plus_age': base_cov+['age'],
        'plus_TBI': base_cov+['TBI'],
        'plus_age_TBI': base_cov+['age','TBI'],
        'plus_motion': base_cov+['meanFD_retained'],
        'plus_RVPL_support': base_cov+['R_VPL_cortex_n_available'],
    }
    sensitivity = []
    for cname, frame in cohorts.items():
        for spec, cov in specs.items():
            res, _ = feature_test(frame, 'favourable6', PRIMARY, cov, 2000, 2000, rng, do_loo=False)
            res['cohort'] = cname; res['specification'] = spec; sensitivity.append(res)
    pd.DataFrame(sensitivity).to_csv(OUT/'RVPL_SMA_covariate_sensitivity.csv', index=False)

    cortical = []
    for target in names[:22]:
        values, mean, sd, nref = edge_standardized(patient_mats, hc_mats, idx['R_VPL'], idx[target])
        col = f'edge_R_VPL__{target}'
        data[col] = data.Subject.map(values)
        res, _ = feature_test(data, 'favourable6', col, base_cov, 2000, 0, rng, do_loo=False)
        res.update({'target': target, 'HC_mean': mean, 'HC_sd': sd, 'HC_n': nref,
                    'historical_candidate': target == 'SMA'})
        cortical.append(res)
    cortical = pd.DataFrame(cortical)
    cortical['BH_FDR_q'] = bh_fdr(cortical.permutation_p)
    cortical['permutation_p_rank'] = cortical.permutation_p.rank(method='min')
    cortical.to_csv(OUT/'RVPL_SMA_cortical_target_specificity.csv', index=False)

    thalamic = []
    for seed in names[22:30]:
        values, mean, sd, nref = edge_standardized(patient_mats, hc_mats, idx[seed], idx['SMA'])
        col = f'edge_{seed}__SMA'; data[col] = data.Subject.map(values)
        res, _ = feature_test(data, 'favourable6', col, base_cov, 2000, 0, rng, do_loo=False)
        res.update({'seed': seed, 'HC_mean': mean, 'HC_sd': sd, 'HC_n': nref,
                    'historical_candidate': seed == 'R_VPL'})
        thalamic.append(res)
    thalamic = pd.DataFrame(thalamic)
    thalamic['BH_FDR_q'] = bh_fdr(thalamic.permutation_p)
    thalamic['permutation_p_rank'] = thalamic.permutation_p.rank(method='min')
    thalamic.to_csv(OUT/'RVPL_SMA_thalamic_seed_specificity.csv', index=False)

    avail = availability_audit(data)
    avail.to_csv(OUT/'RVPL_SMA_feature_availability_audit.csv', index=False)
    available = data[data[PRIMARY].notna()]
    rho_motion = stats.spearmanr(available[PRIMARY], available.meanFD_retained, nan_policy='omit')
    rho_support = stats.spearmanr(available[PRIMARY], available.R_VPL_cortex_n_available, nan_policy='omit')
    pd.DataFrame([
        {'comparison':'feature_vs_meanFD_retained','rho':rho_motion.statistic,'p':rho_motion.pvalue,'n':len(available)},
        {'comparison':'feature_vs_RVPL_cortical_support','rho':rho_support.statistic,'p':rho_support.pvalue,'n':len(available)},
    ]).to_csv(OUT/'RVPL_SMA_imaging_confound_correlations.csv', index=False)

    cv_summaries = []
    for cname, frame in cohorts.items():
        metrics, predictions, used = repeated_nested_cv(frame, 'favourable6', PRIMARY, 200)
        metrics.to_csv(OUT/f'RVPL_SMA_nested_CV_{cname}.csv', index=False)
        predictions.to_csv(OUT/f'RVPL_SMA_nested_CV_predictions_{cname}.csv', index=False)
        cv_summaries.append({
            'cohort': cname, 'analysis_n': len(used), 'events': int(used.favourable6.sum()),
            'repeats': len(metrics),
            'median_M0_brier': float(metrics.M0_brier.median()),
            'median_M1_brier': float(metrics.M1_brier.median()),
            'median_delta_brier': float(metrics.delta_brier_M1_minus_M0.median()),
            'proportion_M1_brier_better': float((metrics.delta_brier_M1_minus_M0 < 0).mean()),
            'median_M0_logloss': float(metrics.M0_logloss.median()),
            'median_M1_logloss': float(metrics.M1_logloss.median()),
            'median_delta_logloss': float(metrics.delta_logloss_M1_minus_M0.median()),
            'proportion_M1_logloss_better': float((metrics.delta_logloss_M1_minus_M0 < 0).mean()),
            'median_M0_auc': float(metrics.M0_auc.median()),
            'median_M1_auc': float(metrics.M1_auc.median()),
            'median_delta_auc': float(metrics.delta_auc_M1_minus_M0.median()),
            'proportion_M1_auc_better': float((metrics.delta_auc_M1_minus_M0 > 0).mean()),
        })
    cv_summary = pd.DataFrame(cv_summaries)
    cv_summary.to_csv(OUT/'RVPL_SMA_nested_CV_summary.csv', index=False)

    main = primary[(primary.cohort == 'main_source_n30') & (primary.outcome == 'favourable6')].iloc[0]
    strict = primary[(primary.cohort == 'strict_source_n27') & (primary.outcome == 'favourable6')].iloc[0]
    ct = cortical[cortical.target == 'SMA'].iloc[0]
    ts = thalamic[thalamic.seed == 'R_VPL'].iloc[0]
    cvmain = cv_summary[cv_summary.cohort == 'main_source_n30'].iloc[0]
    checks = {
        'primary_permutation_p_lt_0_05': bool(main.permutation_p < .05),
        'strict_direction_consistent': bool(strict.beta > 0),
        'bootstrap_same_direction_ge_0_90': bool(main.bootstrap_same_direction >= .90),
        'loo_no_direction_reversal': bool(main.loo_direction_reversals == 0),
        'cortical_family_BH_q_lt_0_05': bool(ct.BH_FDR_q < .05),
        'thalamic_family_BH_q_lt_0_05': bool(ts.BH_FDR_q < .05),
        'prediction_brier_better_ge_0_70': bool(cvmain.proportion_M1_brier_better >= .70),
        'prediction_median_delta_brier_lt_0': bool(cvmain.median_delta_brier < 0),
    }
    summary = {
        'status': 'complete', 'historical_feature': PRIMARY,
        'chronology': lock['chronology_disclosure'],
        'HC_reference': {'n': hc_n, 'mean_raw_z': hc_mean, 'sd_raw_z': hc_sd},
        'source_n': len(data), 'strict_source_n': int(data.strict27_sensitivity_cohort.sum()),
        'analysis_n': int(data.primary_available.sum()), 'strict_analysis_n': int(data.strict_available.sum()),
        'primary_result': {k:(v.item() if hasattr(v,'item') else v) for k,v in main.to_dict().items()},
        'strict_result': {k:(v.item() if hasattr(v,'item') else v) for k,v in strict.to_dict().items()},
        'cortical_specificity': {'candidate_rank': int(ct.permutation_p_rank), 'family_n': len(cortical),
                                 'candidate_BH_q': float(ct.BH_FDR_q)},
        'thalamic_specificity': {'candidate_rank': int(ts.permutation_p_rank), 'family_n': len(thalamic),
                                 'candidate_BH_q': float(ts.BH_FDR_q)},
        'nested_CV_primary': {k:(v.item() if hasattr(v,'item') else v) for k,v in cvmain.to_dict().items()},
        'decision_checks': checks,
        'association_robust': bool(all(checks[k] for k in [
            'primary_permutation_p_lt_0_05','strict_direction_consistent',
            'bootstrap_same_direction_ge_0_90','loo_no_direction_reversal'])),
        'anatomical_specificity_supported': bool(checks['cortical_family_BH_q_lt_0_05'] and checks['thalamic_family_BH_q_lt_0_05']),
        'internal_prediction_supported': bool(checks['prediction_brier_better_ge_0_70'] and checks['prediction_median_delta_brier_lt_0']),
        'claim_boundary': 'Same-cohort internal robustness and specificity; not independent replication, causality, or deployment.',
        'lock_sha256': sha256(LOCK), 'analysis_script_sha256': sha256(Path(__file__).resolve()),
        'runtime_minutes': (time.time()-started)/60,
    }
    (OUT/'RVPL_SMA_VALIDATION_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
