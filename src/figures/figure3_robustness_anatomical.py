"""Data-faithful forest/LOO figure; frozen estimates and family-specific BH q."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
import pymupdf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verification'))
from core_verification import OUT,sha,save_json

FIG=OUT/'figures'
LOG=OUT/'logs/robustness_anatomical_composite_20261020'
DATA=OUT/'tables/robustness_anatomical_composite'
for p in [LOG,DATA]:p.mkdir(parents=True,exist_ok=True)
INK='#20262D';NAVY='#3C5488';TEAL='#008F82';CORAL='#D97942'
GRAY='#9A9FA6';DARKGRAY='#65717D';PALE='#EFF4FA';RULE='#ACB3BC'
XLIM=(.3,30)
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.85,
 'text.color':INK,'axes.labelcolor':INK,'xtick.labelsize':8.5,'ytick.labelsize':9})


def panel(fig,letter,title,x,y,subtitle=None):
    fig.text(x,y,letter,fontsize=24,fontweight='bold',va='center')
    fig.text(x+.037,y,title,fontsize=14,fontweight='bold',va='center')
    if subtitle:fig.text(x+.037,y-.026,subtitle,fontsize=9.3,color=DARKGRAY,va='center')


def y_to_fig(ax,y):
    return ax.figure.transFigure.inverted().transform(ax.transData.transform([1,y]))[1]


def shade_row(fig,left,right,fy,height,color=PALE):
    fig.add_artist(plt.Rectangle((left,fy-height/2),right-left,height,
                   transform=fig.transFigure,facecolor=color,edgecolor='none',zorder=-3))


def format_p(p):return f'{p:.4f}'
def format_q(q):return f'{q:.3f}'
def format_or(r):return f'{r.OR:.2f} ({r.OR95_low:.2f}-{r.OR95_high:.2f})'


def forest_axis(fig,rect,ylim):
    ax=fig.add_axes(rect)
    ax.set(xscale='log',xlim=XLIM,ylim=ylim,yticks=[])
    ax.axvline(1,c='#747F8A',ls=(0,(4,3)),lw=.95,zorder=0)
    ax.xaxis.set_major_locator(FixedLocator([.5,1,2,5,10,20]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x',length=3,pad=4)
    ax.set_xlabel('OR per healthy-reference SD\n(log scale)',fontsize=9.1,labelpad=7)
    return ax


def interval(ax,y,r,color,focused=False):
    assert XLIM[0]<r.OR95_low<r.OR<r.OR95_high<XLIM[1]
    ax.errorbar(r.OR,y,xerr=[[r.OR-r.OR95_low],[r.OR95_high-r.OR]],fmt='o',
        color=color,ecolor=color,ms=6.8 if focused else 5.6,lw=1.45 if focused else 1.05,
        capsize=2.5,capthick=1.1,zorder=4)


def prepare():
    paths={
      'robustness':OUT/'05_master_results/robustness_verified.csv',
      'LOO':OUT/'05_master_results/leave_one_out.csv',
      'seeds':OUT/'03_anatomical_competition/seed_to_SMA_verified.csv',
      'targets':OUT/'03_anatomical_competition/RVPL_to_cortex_verified.csv'}
    r,l,s,t=[pd.read_csv(paths[k]) for k in ['robustness','LOO','seeds','targets']]
    assert r.converged.all() and r.point_estimate_matches_archived.all()
    models=['main','main_plus_age','main_plus_TBI','main_plus_age_TBI','main_plus_motion',
       'main_plus_RVPL_support','strict','strict_plus_age','strict_plus_TBI',
       'strict_plus_age_TBI','strict_plus_motion','strict_plus_RVPL_support']
    a=r.set_index('model').loc[models].reset_index()
    labels=['Main: Base','Main: + age','Main: + TBI','Main: + age and TBI',
       'Main: + motion','Main: + R-VPL support','Strict: Base','Strict: + age',
       'Strict: + TBI','Strict: + age and TBI','Strict: + motion','Strict: + R-VPL support']
    a['display_label']=labels
    a['plot_y']=np.r_[np.arange(13,7,-1),np.arange(6,0,-1)]
    assert a.n.tolist()==[26]*6+[23]*6
    assert (a.beta>0).all()
    l=l.loc[l.cohort.eq('main')].reset_index(drop=True)
    assert len(l)==26 and l.converged.all() and l.removed_Subject.nunique()==26
    l['refit_index']=np.arange(1,27)
    s=s.sort_values('rank_by_permutation_p').reset_index(drop=True)
    t=t.sort_values('rank_by_permutation_p').reset_index(drop=True)
    assert len(s)==8 and len(t)==22 and s.focused.sum()==t.focused.sum()==1
    for f in [s,t]:
        assert f.converged.all() and f.HC_n.eq(26).all()
        assert np.allclose(np.exp(f.beta),f.OR)
        assert np.allclose(np.exp(f.beta-1.96*f.se),f.OR95_low)
        assert np.allclose(np.exp(f.beta+1.96*f.se),f.OR95_high)
        assert np.allclose(stats.false_discovery_control(f.permutation_p),f.BH_q)
    primary=r.loc[r.model.eq('main')].iloc[0]
    for focused in [s.loc[s.focused].iloc[0],t.loc[t.focused].iloc[0]]:
        assert np.isclose(primary.OR,focused.OR) and np.isclose(primary.OR95_low,focused.OR95_low)
    a.to_csv(DATA/'panel_A_12_robustness_specifications.csv',index=False)
    l.to_csv(DATA/'panel_B_all_26_leave_one_out_refits.csv',index=False)
    s.to_csv(DATA/'panel_C_all_8_seeds_to_SMA.csv',index=False)
    t.to_csv(DATA/'panel_D_all_22_RVPL_targets.csv',index=False)
    save_json(LOG/'SOURCE_QC.json',{'source_sha256':{k:sha(p) for k,p in paths.items()},
      'all_values_copied_from_verified_outputs':True,'new_tests_performed':False,
      'main_OR':float(primary.OR),'main_CI':[float(primary.OR95_low),float(primary.OR95_high)],
      'main_frozen_residual_permutation_P':float(primary.frozen_permutation_p),
      'strict_CI_crossing_null_specifications':a.loc[a.model.str.startswith('strict')&a.OR95_low.le(1),'model'].tolist(),
      'LOO_main_n':len(l),'LOO_positive_coefficients':int(l.beta.gt(0).sum()),
      'LOO_main_beta_range':[float(l.beta.min()),float(l.beta.max())],
      'anatomical_seed_n':len(s),'anatomical_cortical_target_n':len(t),
      'target_patient_n_range':[int(t.n.min()),int(t.n.max())],
      'all_BH_q_reproduced_from_frozen_family_P':True,
      'shared_focused_edge_estimate_identical_in_A_C_D':True,
      'all_CI_fully_within_common_log_axis':bool(min(a.OR95_low.min(),s.OR95_low.min(),t.OR95_low.min())>XLIM[0] and
         max(a.OR95_high.max(),s.OR95_high.max(),t.OR95_high.max())<XLIM[1])})
    return a,l,s,t,primary


def family_panel(fig,f,rect,left_x,or_x,q_x,header_y,highlight_left,highlight_right,fontsize):
    n=len(f);ax=forest_axis(fig,rect,(-.65,n-.35))
    fig.text(or_x,header_y,'OR (95% CI)',ha='center',fontsize=10,fontweight='bold')
    fig.text(q_x,header_y,'BH q',ha='right',fontsize=10,fontweight='bold')
    for i,r in f.iterrows():
        yy=n-1-i;fy=y_to_fig(ax,yy);focused=bool(r.focused)
        color=NAVY if focused else DARKGRAY if r.BH_q<.05 else GRAY
        if focused:shade_row(fig,highlight_left,highlight_right,fy,rect[3]/(n+.3)*.90)
        interval(ax,yy,r,color,focused=focused)
        label=r.label.replace('_','-')+f' (n={int(r.n)})'
        fig.text(left_x,fy,label,ha='left',va='center',fontsize=fontsize,
           color=NAVY if focused else INK,fontweight='bold' if focused else 'normal')
        fig.text(or_x,fy,format_or(r),ha='center',va='center',fontsize=fontsize,
           color=NAVY if focused else INK,fontweight='bold' if focused else 'normal')
        fig.text(q_x,fy,format_q(r.BH_q),ha='right',va='center',fontsize=fontsize,
           color=NAVY if focused else INK,fontweight='bold' if focused else 'normal')
    return ax


def build():
    a,l,s,t,primary=prepare()
    fig=plt.figure(figsize=(17.2,12.2),facecolor='white')
    fig.text(.5,.98,'Robustness and anatomical concentration of the R-VPL-SMA association',
             ha='center',va='top',fontsize=20,fontweight='bold')
    panel(fig,'A','Covariate and QC robustness',.024,.926,
          'Main n=26; strict-QC n=23')
    panel(fig,'B','No leave-one-out direction reversal',.548,.926,
          'Main cohort; all 26 single-patient exclusions')
    panel(fig,'C','Subcortical seeds to SMA',.024,.468,
          'Eight fixed seeds; ordered by residual-permutation P')
    panel(fig,'D','R-VPL to cortical targets',.548,.468,
          '22 fixed targets; edge-wise sample sizes shown')

    # A: genuine Wald intervals, with a separately labelled residual-permutation P.
    ax=forest_axis(fig,[.176,.567,.164,.31],(.3,13.7))
    fig.text(.414,.887,'OR (95% CI)',ha='center',fontsize=10.2,fontweight='bold')
    fig.text(.508,.887,'Perm. P',ha='right',fontsize=10.2,fontweight='bold')
    for _,r in a.iterrows():
        fy=y_to_fig(ax,r.plot_y);color=TEAL if r.model.startswith('strict') else NAVY
        base=r.model in ['main','strict']
        if base:shade_row(fig,.025,.515,fy,.0185,'#F0F7F6' if r.model=='strict' else PALE)
        interval(ax,r.plot_y,r,color,focused=base)
        fig.text(.029,fy,r.display_label,va='center',fontsize=9.5,
                 color=color if base else INK,fontweight='bold' if base else 'normal')
        fig.text(.414,fy,format_or(r),ha='center',va='center',fontsize=9.3,
                 color=color if base else INK,fontweight='bold' if base else 'normal')
        fig.text(.508,fy,format_p(r.frozen_permutation_p),ha='right',va='center',fontsize=9.3,
                 color=color if base else INK,fontweight='bold' if base else 'normal')
    divider=y_to_fig(ax,7)
    fig.add_artist(plt.Line2D([.027,.514],[divider,divider],transform=fig.transFigure,
                              color=RULE,lw=.8,ls=(0,(3,3))))
    fig.text(.273,.508,'Base: baseline CRS-R + log1p injury-to-stimulation time.',
             ha='center',fontsize=8.7,color=DARKGRAY)

    # B: plot every actual refit, without flattening the high-influence exclusion.
    ax=fig.add_axes([.615,.592,.354,.284])
    ax.axhline(0,c=RULE,lw=1,ls=(0,(4,3)))
    ax.axhline(primary.beta,c=CORAL,lw=1.45,zorder=1)
    ax.scatter(l.refit_index,l.beta,s=38,c=NAVY,edgecolors='white',lw=.5,zorder=3)
    ax.set(xlim=(0,27),ylim=(-.25,max(l.beta.max()+.30,2.8)),xticks=[1,5,10,15,20,26],
           xlabel='Leave-one-out refit',ylabel='Firth coefficient for R-VPL-SMA FC')
    ax.set_yticks(np.arange(0,2.6,.5));ax.tick_params(length=3,pad=5)
    ax.text(.98,primary.beta+.065,'Full-cohort estimate',color=CORAL,ha='right',va='bottom',
        fontsize=9.3,fontweight='bold',transform=ax.get_yaxis_transform())
    ax.text(.04,.94,f'26/26 coefficients > 0\nRange {l.beta.min():.3f}-{l.beta.max():.3f}',
        transform=ax.transAxes,ha='left',va='top',fontsize=10,linespacing=1.5,
        bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
    fig.text(.79,.539,'Refits omit one patient each; direction stability is not independent validation.',
             ha='center',fontsize=8.6,color=DARKGRAY)

    # C/D: all members of each frozen family, not a selected subset.
    family_panel(fig,s,[.176,.100,.164,.311],.029,.414,.508,.426,.025,.515,10.0)
    family_panel(fig,t,[.690,.100,.139,.311],.553,.890,.978,.426,.550,.985,8.9)
    fig.text(.5,.035,'Points are Firth odds ratios; intervals are Wald 95% CIs. Perm. P and BH q use the frozen residual-permutation analyses.',
             ha='center',fontsize=9.0,color=DARKGRAY)
    fig.text(.5,.017,'Anatomical q values are corrected within the 8-seed and 22-target families; these are same-cohort follow-up analyses, not independent confirmation.',
             ha='center',fontsize=9.0,color=DARKGRAY)
    stem='Figure_3_robustness_anatomical_concentration'
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/(stem+'.'+ext),dpi=400,bbox_inches='tight',pad_inches=.13,facecolor='white')
    plt.close(fig)
    with pymupdf.open(FIG/(stem+'.pdf')) as doc:
        assert len(doc)==1
        text=doc[0].get_text()
        for term in ['1.23-17.31','26/26','0.014','0.040','0.0018']:
            assert term in text,term
        assert all(doc[0].rect.contains(pymupdf.Rect(b[:4])) for b in doc[0].get_text('blocks'))
        doc[0].get_pixmap(matrix=pymupdf.Matrix(1.35,1.35)).save(LOG/'FINAL_PDF_RENDER.png')
        save_json(LOG/'PDF_QA.json',{'pages':len(doc),'extractable_text_characters':len(text),
            'embedded_images':len(doc[0].get_images()),'all_elements_vector':len(doc[0].get_images())==0,
            'all_text_within_page':True})
    print('Robustness/anatomical composite exported as PNG/PDF/SVG.',flush=True)


if __name__=='__main__':build()
