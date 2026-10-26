"""Build publication figures from the validated numerical outputs."""
from pathlib import Path
import sys,inspect,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
ROOT=Path(os.environ.get('RVPL_SMA_PROJECT_ROOT', Path.cwd()))
V=Path(os.environ.get('RVPL_SMA_DISCOVERY_DIR', ROOT/'restricted_inputs'/'discovery_analysis'))
OUT=Path(os.environ.get('RVPL_SMA_VALIDATION_OUTPUT', ROOT/'outputs'/'statistical_validation'));A=OUT/'analysis_outputs'
for label in ['main_primary','strict_primary','RmRt_SMA']:
    assert len(pd.read_csv(A/('calibration_'+label+'.csv')))==2000
    assert (A/('calibration_'+label+'_summary.csv')).exists()
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
QA=OUT/'qa';QA.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
import figure1_outcome_guided as f1
import figure3_robustness_anatomical as f3
import supporting_figures as f2
family=pd.read_csv(A/'families_uniform5000_Wald_and_permutation.csv')
rob=pd.read_csv(A/'robustness_uniform5000.csv')
primary=family[(family.family=='seed_to_SMA')&(family.label=='R_VPL')].iloc[0]

def prep1():
    node=pd.read_csv(V/'tables/figure1_outcome_guided/node_coordinates_and_networks.csv')
    diff=pd.read_csv(V/'tables/figure1_outcome_guided/matrix_group_difference_Fisher_z.csv',index_col=0)
    sig=pd.read_csv(V/'tables/figure1_outcome_guided/all_15_archived_nominal_edges.csv')
    focus=pd.read_csv(V/'tables/figure1_outcome_guided/focused_edge_patient_values.csv')
    candidate=sig[sig.focused].iloc[0]
    return node,diff.index.tolist(),diff.to_numpy(),sig,focus,candidate
f1.prepare=prep1;f1.FIG=FIG;f1.LOG=QA/'figure1';f1.LOG.mkdir(exist_ok=True)
f1.LABEL['Auditory']='Auditory / MCC';f1.SHORT['Auditory']='AUD/MCC'
# This repairs the anatomical label without retrospectively changing node order or features.
f1.build()
(FIG/'Figure_1_outcome_guided_identification.pdf').rename(FIG/'Figure_1.pdf')
(FIG/'Figure_1_outcome_guided_identification.png').rename(FIG/'Figure_1.png')

def prep3():
    a=rob.iloc[:12].copy();a['frozen_permutation_p']=a.permutation_p
    a['display_label']=['Main: Base','Main: + age','Main: + TBI','Main: + age and TBI','Main: + motion','Main: + R-VPL support','Strict: Base','Strict: + age','Strict: + TBI','Strict: + age and TBI','Strict: + motion','Strict: + R-VPL support']
    a['plot_y']=np.r_[np.arange(13,7,-1),np.arange(6,0,-1)]
    l=pd.read_csv(V/'05_master_results/leave_one_out.csv').query("cohort=='main'").copy();l['refit_index']=np.arange(1,27)
    fs=[]
    for name in ['seed_to_SMA','RVPL_to_cortex']:
        x=family[family.family==name].sort_values('rank_permutation').copy().reset_index(drop=True)
        x['BH_q']=x.permutation_BH_q;x['focused']=x.label.eq('R_VPL' if name=='seed_to_SMA' else 'SMA');fs.append(x)
    return a,l,*fs,a.iloc[0]
f3.prepare=prep3;f3.FIG=FIG;f3.LOG=QA/'figure3';f3.LOG.mkdir(exist_ok=True)
src=inspect.getsource(f3.build)
src=src.replace('use the frozen residual-permutation analyses.','use uniform 5,000-iteration residual permutations.')
src=src.replace("['1.23-17.31','26/26','0.014','0.040','0.0018']","['1.23-17.31','26/26','0.032','0.088','0.0040']")
exec(compile(src,'figure3_generator','exec'),f3.__dict__);f3.build()
(FIG/'Figure_3_robustness_anatomical_concentration.pdf').rename(FIG/'Figure_3.pdf')
(FIG/'Figure_3_robustness_anatomical_concentration.png').rename(FIG/'Figure_3.png')

# Keep every numerical plotted observation, curve and interval unchanged in Fig 2;
# only the explicitly labelled residual-permutation annotation changes.
f2.M=FIG
src=inspect.getsource(f2.main2).split('    for num,stem in ')[0]
assert 'Residual-permutation P=0.0018' in src
src=src.replace('Residual-permutation P=0.0018',f'Residual-permutation P={primary.permutation_p:.4f}')
src=src.replace('Complementary outcomes from the same 26 patients','Multiple readouts from the same 26 patients')
exec(compile(src,'figure2_generator','exec'),f2.__dict__);f2.main2()
(FIG/'Figure_2_behavioural_convergence.pdf').rename(FIG/'Figure_2.pdf')
(FIG/'Figure_2_behavioural_convergence.png').rename(FIG/'Figure_2.png')

fig,axes=plt.subplots(3,2,figsize=(10.6,11.2))
for row,label in enumerate(['main_primary','strict_primary','RmRt_SMA']):
    df=pd.read_csv(A/('calibration_'+label+'.csv'));s=pd.read_csv(A/('calibration_'+label+'_summary.csv'))
    for method,color,legend in [('permutation_p','#0072B2','Residual permutation'),('wald_p','#D55E00','Wald')]:
        p=np.sort(df[method]);u=(np.arange(len(p))+.5)/len(p)
        axes[row,0].plot(-np.log10(u),-np.log10(p),'.',ms=2.5,label=legend,c=color)
        axes[row,1].plot(p,np.arange(1,len(p)+1)/len(p),label=legend,c=color)
    axes[row,0].plot([0,4],[0,4],'--',c='gray',lw=1)
    axes[row,0].set(xlabel='Expected -log10(P)',ylabel='Observed -log10(P)',title=['Main R-VPL-SMA','Strict R-VPL-SMA','R-mRt-SMA'][row])
    axes[row,1].plot([0,1],[0,1],'--',c='gray',lw=1);axes[row,1].set(xlabel='P value',ylabel='Empirical cumulative probability')
    ss=s[s.alpha==.05];vals=dict(zip(ss.method,ss.type_I_error))
    axes[row,1].text(.06,.88,f"Rejection at alpha=.05\nPermutation {vals['permutation_p']:.3f}; Wald {vals['wald_p']:.3f}",transform=axes[row,1].transAxes,fontsize=9)
axes[0,1].legend(loc='lower right',fontsize=9)
fig.suptitle('Supplementary Figure S6  Null calibration of the fixed-edge procedures',fontweight='bold',fontsize=14)
fig.text(.5,.017,'2,000 nuisance-only null simulations per design; 5,000 residual permutations per simulation. No candidate-selection step simulated.',ha='center',fontsize=8.5)
fig.tight_layout(rect=[0,.035,1,.965]);fig.savefig(FIG/'Figure_S6_calibration.pdf');fig.savefig(FIG/'Figure_S6_calibration.png',dpi=170);plt.close(fig)
print('Publication Figures 1-3 and Supplementary Figure S6 generated.',flush=True)
