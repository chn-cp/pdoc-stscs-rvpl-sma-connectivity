from pathlib import Path
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from core_verification import OUT, SEED, save_json

checks=[]
def record(label,value,details=''):
    checks.append({'check':label,'pass':bool(value),'details':details})
    if not value:raise AssertionError((label,details))

d=pd.read_csv(OUT/'01_ordered_state/ordered_state_analysis.csv')
summary=json.loads((OUT/'01_ordered_state/ordered_state_summary.json').read_text(encoding='utf-8'))
record('focused_unique_n26',len(d)==26 and d.patient_id.is_unique)
record('state_counts_preserved',d.final_state_code.value_counts().sort_index().tolist()==[4,10,4,8])
x=d.RVPL_SMA_raw.to_numpy();g=d.final_state_code.to_numpy()
r=stats.spearmanr(x,g).statistic
record('Spearman_independent_scipy',np.isclose(r,summary['spearman_rho']))
# JT computed via Mann-Whitney group-pair U statistics, a different implementation.
J=0.
for lo in range(4):
    for hi in range(lo+1,4):
        J+=stats.mannwhitneyu(x[g==hi],x[g==lo],alternative='greater').statistic
record('JT_independent_Mann_Whitney_sum',J==summary['JT_statistic'],str(J))
rng=np.random.default_rng(SEED);extreme=0
for _ in range(5000):
    extreme+=abs(stats.spearmanr(x,g[rng.permutation(len(g))]).statistic)>=abs(r)-1e-12
record('5000_label_permutations_reproduced',np.isclose((extreme+1)/5001,summary['spearman_permutation_p_two_sided']))
metrics=pd.read_csv(OUT/'02_spatial_convergence/seed2voxel_sma_convergence_metrics.csv')
record('10_prespecified_spatial_models',len(metrics)==10)
matched=pd.read_csv(OUT/'02_spatial_convergence/archived_cluster_reconstruction_checks.csv')
record('all_17_archived_RVPL_clusters_exactly_reconstructed',len(matched)==17 and matched.all_match.all())
sphere=nib.load(OUT/'02_spatial_convergence/SMA_fixed_sphere_6mm.nii.gz')
record('SMA_discretized_voxel_count',int(sphere.get_fdata().sum())==114)
for sset,name in [('main','RVPL_model_consistency_count'),('sensitivity','RVPL_strict_model_consistency_count')]:
    summed=np.zeros(sphere.shape,dtype=int)
    for m in range(5):
        img=nib.load(OUT/f'02_spatial_convergence/RVPL_{sset}_M{m}_significant_t.nii.gz')
        record(f'{sset}_M{m}_affine',np.allclose(img.affine,sphere.affine))
        mask=img.get_fdata()>0;summed+=mask
        row=metrics[(metrics.subject_set==sset)&metrics.model.str.startswith(f'M{m}')].iloc[0]
        record(f'{sset}_M{m}_overlap',(mask&(sphere.get_fdata()>0)).sum()==row.overlap_voxels)
    count=nib.load(OUT/f'02_spatial_convergence/{name}.nii.gz').get_fdata()
    record(f'{sset}_consistency_equals_sum_of_thresholded_models',np.array_equal(summed,count))
    record(f'{sset}_count_range_0_5',count.min()==0 and count.max()<=5)
family=pd.read_csv(OUT/'03_anatomical_competition/family_point_estimate_QC.csv')
record('all_180_edge_quantities_reproduced',len(family)==180 and family.match.all())
for name,n in [('seed_to_SMA',8),('RVPL_to_cortex',22)]:
    a=pd.read_csv(OUT/f'03_anatomical_competition/{name}_verified.csv')
    record(name+'_family_size',len(a)==n)
    q=stats.false_discovery_control(a.permutation_p.to_numpy(),method='bh')
    record(name+'_BH_independent_scipy',np.allclose(q,a.BH_q))
    record(name+'_primary_P_frozen',np.isclose(a[a.focused].permutation_p.iloc[0],.001799640071985603))
cv=pd.read_csv(OUT/'logs/CV_metric_QC.csv')
record('CV_12_score_vectors_all_match',len(cv)==12 and cv.all_200_repeat_scores_match.all())
rob=pd.read_csv(OUT/'05_master_results/robustness_verified.csv')
record('all_14_robustness_point_estimates_match',len(rob)==14 and rob.point_estimate_matches_archived.all())
record('all_14_point_estimate_fits_converged',rob.converged.all())
boots=pd.read_csv(OUT/'05_master_results/bootstrap_verification.csv')
loo=pd.read_csv(OUT/'05_master_results/leave_one_out.csv')
oldloo=pd.read_csv(OUT/'00_inputs/historical/RVPL_SMA_leave_one_out.csv')
for label,cohort in [('main','main_source_n30'),('strict','strict_source_n27')]:
    b=boots[boots.cohort==label]
    record(label+'_5000_bootstrap_attempts',len(b)==5000)
    l=loo[loo.cohort==label]
    old=oldloo[(oldloo.cohort==cohort)&(oldloo.outcome=='favourable6')]
    merged=l.merge(old,on='removed_Subject',validate='one_to_one')
    record(label+'_LOO_matches_frozen',np.allclose(merged.beta,merged.beta_feature,rtol=1e-7,atol=1e-8))
    record(label+'_LOO_no_direction_reversals',(l.beta>0).all())
null=pd.read_csv(OUT/'05_master_results/predictor_residual_permutation.csv')
record('new_primary_permutation_5000_attempts',len(null)==5000)
record('new_primary_permutation_all_converged',null.converged.all())
# The copied historical archive must agree with the server-data mirror, rather than assuming it does.
archive_match=[]
with ZipFile('./restricted_inputs/archive/RVPL_SMA_HISTORICAL_HYPOTHESIS_20260920.zip') as z:
    for p in (OUT/'00_inputs/historical').glob('*.csv'):
        member='RVPL_SMA_HISTORICAL_HYPOTHESIS_20260920/'+p.name
        archive_match.append({'file':p.name,'byte_identical_to_local_server_zip':p.read_bytes()==z.read(member)})
pd.DataFrame(archive_match).to_csv(OUT/'logs/historical_archive_identity.csv',index=False)
record('all_frozen_csvs_identical_to_local_server_archive',all(r['byte_identical_to_local_server_zip'] for r in archive_match))
pd.DataFrame(checks).to_csv(OUT/'logs/independent_numerical_QC.csv',index=False)
save_json(OUT/'logs/independent_qc_summary.json',{'status':'pass','checks':len(checks),'failed':0})
print('Independent numerical QC passed:',len(checks),flush=True)
