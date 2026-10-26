"""Five supplement figures and the correctly ordered behavioural main Figure 2."""
from pathlib import Path
import json
import os,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import pymupdf

W=Path(__file__).resolve().parent;V=Path(os.environ.get('RVPL_SMA_DISCOVERY_DIR', Path.cwd()/'restricted_inputs'/'discovery_analysis'))
F=W/'figures';M=W/'main_figures'
NAVY='#3C5488';TEAL='#008F82';CORAL='#E64B35';GREY='#929AA4';INK='#20262D';ORANGE='#E69F00'
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,'axes.linewidth':.8})
q=pd.read_csv(W/'data/qc33_deidentified.csv')
av=pd.read_csv(W/'data/focused_availability30_deidentified.csv')
focus=pd.read_csv(V/'00_inputs/master_focused_cohort_n26.csv')
metrics=pd.read_csv(V/'02_spatial_convergence/seed2voxel_sma_convergence_metrics.csv')
clusters=pd.read_csv(V/'02_spatial_convergence/seed2voxel_cluster_peak_distances.csv')


def save(fig,stem,folder=F):
    for ext in ['png','pdf']:
        fig.savefig(folder/(stem+'.'+ext),dpi=330,bbox_inches='tight',pad_inches=.1,facecolor='white')
    plt.close(fig)


def title(fig,text):fig.suptitle(text,fontsize=16,fontweight='bold',y=.98)
def ptitle(ax,letter,text):ax.set_title(letter+'  '+text,loc='left',fontsize=11.5,fontweight='bold',pad=12)
def box(fig,rect,text,fc='#EDF3FB',fontsize=11):
    x,y,w,h=rect
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.014',
      transform=fig.transFigure,facecolor=fc,edgecolor='#CDD7E4',lw=.9))
    fig.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.4)
def arrow(fig,a,b):fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=12,color=INK,lw=1))


def s1():
    fig=plt.figure(figsize=(10,7.5),facecolor='white');title(fig,'Supplementary Figure S1  Cohort flow and focused-edge availability')
    boxes=[(.81,'50 patients receiving stSCS'),(.62,'33 pretreatment fMRI records'),(.43,'30 primary-QC scans'),(.24,'26 estimable R-VPL-SMA edges')]
    for i,(y,text) in enumerate(boxes):
        box(fig,(.06,y,.40,.09),text,'#EEF3FA' if i<3 else '#E8F3EE')
        if i<3:arrow(fig,(.26,y-.008),(.26,boxes[i+1][0]+.10))
    for yy,text in zip([.754,.564,.374],['17 without completed pretreatment fMRI','3 excluded at initial imaging QC','4 fail focused-edge lesion-support rules']):
        fig.text(.49,yy,text,fontsize=10.2,va='center')
    box(fig,(.59,.45,.32,.10),'27 strict-QC scans\n≥90 retained frames','#F0F7F5')
    box(fig,(.59,.24,.32,.10),'23 estimable strict-QC edges\n11 MCS+/EMCS outcomes','#E8F3EE')
    arrow(fig,(.467,.475),(.578,.50));arrow(fig,(.75,.44),(.75,.35))
    fig.text(.26,.185,'12 MCS+/EMCS; 14 below MCS+',ha='center',fontsize=10.8,color=NAVY,fontweight='bold')
    fig.text(.5,.108,'Six-month outcome recorded for all 50 patients; the focused cohort contains no deaths.',ha='center',fontsize=10)
    fig.text(.5,.068,'Primary scan QC and focused-edge estimability are distinct steps. No missing connection was imputed.',ha='center',fontsize=9,color='#606B76')
    save(fig,'Figure_S1_cohort_flow')


def s2():
    fig,axs=plt.subplots(2,2,figsize=(10,7.8));title(fig,'Supplementary Figure S2  Imaging quality-control diagnostics')
    x=np.arange(1,34);colors=np.where(~q.primary_QC,CORAL,np.where(q.strict_QC,NAVY,TEAL))
    ax=axs[0,0];ax.scatter(x,q.n_retained,c=colors,s=30)
    ax.axhline(75,c=GREY,ls='--',lw=.9,label='Primary ≥75');ax.axhline(90,c=TEAL,ls=':',lw=1,label='Strict ≥90')
    ax.set(xlabel='Scan record',ylabel='Retained frames');ax.legend(fontsize=8,frameon=False,loc='lower left');ptitle(ax,'A','Retained frames')
    ax=axs[0,1];ax.scatter(q.meanFD_all,q.n_retained,c=colors,s=30)
    ax.set(xlabel='Mean FD before censoring (mm)',ylabel='Retained frames');ptitle(ax,'B','Motion and data retention')
    ax=axs[1,0];ax.scatter(x,q.rank_final,c=colors,s=30);ax.axhline(20,c=GREY,ls='--',lw=.9)
    ax.set(xlabel='Scan record',ylabel='Residual design rank');ptitle(ax,'C','Residual design rank')
    ax=axs[1,1]
    for i,(flag,color,label) in enumerate([(~q.primary_QC,CORAL,'Excluded\n(n=3)'),(q.primary_QC&~q.strict_QC,TEAL,'Primary only\n(n=3)'),(q.strict_QC,NAVY,'Strict QC\n(n=27)')]):
        v=q.loc[flag,'meanFD_retained'].to_numpy();ax.scatter(i+np.linspace(-.08,.08,len(v)),v,c=color,s=28)
        ax.plot([i-.18,i+.18],[np.median(v)]*2,c=INK,lw=1.2)
    ax.set(xticks=[0,1,2],xticklabels=['Excluded\n(n=3)','Primary only\n(n=3)','Strict QC\n(n=27)'],ylabel='Mean retained-frame FD (mm)')
    ptitle(ax,'D','Retained-frame motion')
    fig.text(.5,.025,'Colors show the final QC category. Volume censoring used FD >0.5 mm; mean FD is descriptive, not a scan exclusion rule.',ha='center',fontsize=8.6,color='#606B76')
    fig.subplots_adjust(left=.09,right=.97,bottom=.12,top=.88,wspace=.28,hspace=.45)
    save(fig,'Figure_S2_imaging_QC')


def s3():
    d=pd.read_csv(W/'deliverables/Supplementary_Data_1_complete_435_edge_outcome_screen.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8));title(fig,'Supplementary Figure S3  Full outcome-guided screening landscape')
    colors=np.where(d.nominal_P_lt_05,TEAL,'#BAC2CC');focused=d.focused_followup_candidate
    ax=axs[0];ax.scatter(d.screen_rank,-np.log10(d.Welch_P_two_sided),s=10,c=colors,alpha=.85)
    ax.axhline(-np.log10(.05),c=GREY,ls='--',lw=.9);ax.scatter(d.loc[focused,'screen_rank'],-np.log10(d.loc[focused,'Welch_P_two_sided']),c=CORAL,s=60,zorder=5)
    ax.annotate('R-VPL-SMA\nRank 1; P=0.000798',xy=(1,-np.log10(d.Welch_P_two_sided.iloc[0])),xytext=(70,2.9),
        fontsize=9,color=CORAL,arrowprops={'arrowstyle':'-','color':CORAL})
    ax.text(.98,.94,'15/435 nominal P<0.05',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax.set(xlabel='Edge rank by outcome-group P',ylabel='-log10(two-sided Welch P)',xlim=(-8,445));ptitle(ax,'A','All 435 raw P values')
    ax=axs[1];ax.plot(d.screen_rank,d.BH_q_across_435,c=NAVY,lw=1.1);ax.scatter(d.screen_rank,d.BH_q_across_435,c='#BAC2CC',s=5)
    ax.axhline(.05,c=GREY,ls='--',lw=.9);ax.scatter([1],[d.BH_q_across_435.iloc[0]],c=CORAL,s=60,zorder=5)
    ax.annotate('R-VPL-SMA\nBH q=0.347',xy=(1,d.BH_q_across_435.iloc[0]),xytext=(65,.24),color=CORAL,fontsize=9,
       arrowprops={'arrowstyle':'-','color':CORAL})
    ax.set(xlabel='Edge rank by outcome-group P',ylabel='BH q across 435 edges',xlim=(-8,445),ylim=(0,1.03));ptitle(ax,'B','Whole-connectome multiplicity')
    fig.text(.5,.025,'All edges are retained. Teal denotes nominal candidates; coral identifies the focused follow-up candidate.',ha='center',fontsize=9,color='#606B76')
    fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.83,wspace=.30)
    save(fig,'Figure_S3_full_435_screen')


def s4():
    anatomy=load_mni152_template(resolution=1);sphere=nib.load(V/'02_spatial_convergence/SMA_fixed_sphere_6mm.nii.gz')
    paths=list((V/'02_spatial_convergence').glob('RVPL_*_M*_significant_t.nii.gz'))
    vmax=max(float(nib.load(p).get_fdata().max()) for p in paths)
    cmap=plt.get_cmap('YlOrRd');gray=ListedColormap(plt.get_cmap('gray')(np.linspace(0,1,512)**.65))
    with PdfPages(F/'Figure_S4_detailed_seed_maps.pdf') as pdf:
        for group,title_group in [('main','Main seed-to-voxel cohort n=27'),('sensitivity','Retained-frames sensitivity cohort n=24')]:
            fig=plt.figure(figsize=(10,12.0),facecolor='white')
            fig.text(.5,.985,'Supplementary Figure S4  Detailed R-VPL seed-to-voxel maps',ha='center',va='top',fontsize=15,fontweight='bold')
            fig.text(.5,.95,title_group,ha='center',fontsize=12,color=NAVY)
            model_rows=metrics.loc[metrics.subject_set.eq(group)].reset_index(drop=True)
            for i,r in model_rows.iterrows():
                yy=.76-i*.16
                hits=clusters.loc[clusters.subject_set.eq(group)&clusters.model.eq(r.model)].sort_values('fwe')
                z=float(hits.iloc[0].z);tmap=nib.load(V/'02_spatial_convergence'/f'RVPL_{group}_M{i}_significant_t.nii.gz')
                fig.text(.025,yy+.068,f'M{i}',fontsize=13,fontweight='bold')
                fig.text(.025,yy+.041,f'n={r.n}\ndf={r.df}',fontsize=8.6,linespacing=1.4)
                for mode,cut,rect in [('x',0,[.11,yy,.27,.137]),('z',z,[.40,yy,.23,.137])]:
                    ax=fig.add_axes(rect)
                    disp=plotting.plot_anat(anatomy,display_mode=mode,cut_coords=[cut],axes=ax,
                       annotate=False,draw_cross=False,black_bg=False,cmap=gray,vmin=.02,vmax=.92,
                       threshold=.04,dim=0,colorbar=False,radiological=mode=='z')
                    disp.add_overlay(tmap,threshold=.01,cmap=cmap,vmin=3.4,vmax=vmax,colorbar=False,
                         interpolation='nearest',resampling_interpolation='nearest')
                    if mode=='x' or abs(cut-51)<=6:
                        disp.add_contours(sphere,levels=[.5],colors=['#0072B2'],linewidths=1.1,resampling_interpolation='nearest')
                    da=next(iter(disp.axes.values())).ax
                    if mode=='x':da.invert_xaxis()
                    else:da.text(.02,.02,'R',transform=da.transAxes,fontsize=7);da.text(.98,.02,'L',ha='right',transform=da.transAxes,fontsize=7)
                    fig.text(rect[0]+rect[2]/2,yy-.009,f'{mode}={cut:g} mm',ha='center',fontsize=8)
                lines=[]
                for k,h in enumerate(hits.itertuples(),1):
                    lines.append(f'C{k}: {h.size} voxels; FWE P={h.fwe:.4f}\nPeak ({h.x:g}, {h.y:g}, {h.z:g}); T={h.peak_t:.2f}')
                fig.text(.66,yy+.115,'\n'.join(lines),ha='left',va='top',fontsize=8.4,linespacing=1.35)
                fig.text(.66,yy+.010,f'SMA overlap: {r.overlap_voxels}/114',fontsize=8.8,color=NAVY)
            cax=fig.add_axes([.22,.065,.38,.009]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(3.4,vmax),cmap=cmap),cax=cax,orientation='horizontal')
            fig.text(.41,.039,'Positive t statistic in archived significant clusters',ha='center',fontsize=8.5)
            fig.text(.5,.018,'Blue contours mark the fixed SMA sphere where it intersects the section; no contour is drawn on higher axial sections.',ha='center',fontsize=8.2,color='#606B76')
            pdf.savefig(fig,bbox_inches='tight',pad_inches=.1)
            fig.savefig(F/f'Figure_S4_{group}.png',dpi=300,bbox_inches='tight',pad_inches=.1)
            plt.close(fig)


def s5():
    fig,axs=plt.subplots(2,2,figsize=(10,7.6));title(fig,'Supplementary Figure S5  Focused-edge estimability and lesion support')
    colors=np.where(av.focused_available,NAVY,CORAL)
    ax=axs[0,0];ax.scatter(av.R_VPL_lesion_fraction,av.SMA_lesion_fraction,s=36,c=colors,edgecolors='white',lw=.3)
    ax.axvline(.3,c=GREY,ls='--',lw=.8);ax.axhline(.3,c=GREY,ls='--',lw=.8)
    ax.set(xlabel='R-VPL lesioned fraction',ylabel='SMA lesioned fraction',xlim=(-.03,1.03),ylim=(-.03,1.03));ptitle(ax,'A','Direct ROI lesion support')
    rng=np.random.default_rng(20261020)
    for ax,var,yl,letter,title_text in zip(axs.ravel()[1:],['lesion_vol_ml','meanFD_retained','R_VPL_cortex_n_available'],
        ['Lesion volume (mL)','Mean retained-frame FD (mm)','Estimable R-VPL cortical targets'],['B','C','D'],
        ['Overall lesion burden','Head motion','Focused-seed support']):
        values=[av.loc[av.focused_available.eq(g),var].to_numpy() for g in [True,False]]
        for k,(v,c) in enumerate(zip(values,[NAVY,CORAL])):
            ax.scatter(k+rng.uniform(-.10,.10,len(v)),v,c=c,s=24,alpha=.8,zorder=3)
            ax.plot([k-.18,k+.18],[np.median(v)]*2,c=INK,lw=1.4)
        ax.set(xticks=[0,1],xticklabels=['Estimable\n(n=26)','Not estimable\n(n=4)'],ylabel=yl);ptitle(ax,letter,title_text)
    fig.text(.5,.025,'Navy: estimable R-VPL-SMA. Coral: nonestimable. Lesion-support thresholds were not adjusted after viewing outcomes.',ha='center',fontsize=8.8,color='#606B76')
    fig.subplots_adjust(left=.09,right=.97,bottom=.13,top=.88,wspace=.29,hspace=.45)
    save(fig,'Figure_S5_estimability_lesion_support')


def main2():
    curve=pd.read_csv(V/'tables/discovery_composite/Firth_adjusted_probability_curve.csv')
    order=json.loads((V/'01_ordered_state/ordered_state_summary.json').read_text())
    change=pd.read_csv(V/'05_master_results/continuous_change_associations.csv').query("analysis=='Spearman'").iloc[0]
    fig,axs=plt.subplots(2,2,figsize=(12,9));title(fig,'Figure 2  Pretreatment R-VPL-SMA connectivity converges across behavioural outcomes')
    rng=np.random.default_rng(20261020);feat='R_VPL_SMA_z_local26_z'
    vals=[focus.loc[focus.favourable6.eq(g),feat].to_numpy() for g in [0,1]]
    ax=axs[0,0]
    bp=ax.boxplot(vals,positions=[0,1],widths=.45,showfliers=False,patch_artist=True,medianprops={'color':INK,'lw':1.3})
    for k,(v,c) in enumerate(zip(vals,[NAVY,CORAL])):
        bp['boxes'][k].set(facecolor=c,alpha=.18);ax.scatter(k+rng.uniform(-.11,.11,len(v)),v,s=24,c=c,zorder=4)
    ax.axhline(0,c=GREY,ls='--',lw=.8);ax.set(xticks=[0,1],xticklabels=['Below MCS+\n(n=14)','MCS+/EMCS\n(n=12)'],ylabel='R-VPL-SMA FC\n(healthy-reference SD)')
    exact=json.loads((V/'tables/discovery_composite/binary_group_comparison.json').read_text())
    top=max(max(v) for v in vals);by=top+.25
    ax.plot([0,0,1,1],[by-.06,by,by,by-.06],c=INK,lw=.8)
    ax.text(.5,by+.08,f'Exact two-sided Mann-Whitney P={exact["p_two_sided"]:.5f}',ha='center',va='bottom',fontsize=8.4)
    ax.set_ylim(min(min(v) for v in vals)-.2,top+.92)
    ptitle(ax,'A','Six-month state attainment')
    ax=axs[0,1]
    for k,c in enumerate([GREY,'#4C91D8',ORANGE,CORAL]):
        v=focus.loc[focus.final_state_code.eq(k),'RVPL_SMA_raw'].to_numpy();qq=np.quantile(v,[.25,.5,.75])
        ax.scatter(k+rng.uniform(-.12,.12,len(v)),v,c=c,s=26,zorder=4);ax.plot([k,k],[qq[0],qq[2]],c=c,lw=1.3);ax.plot([k-.17,k+.17],[qq[1]]*2,c=INK,lw=1.3)
    ax.set(xticks=range(4),xticklabels=['VS/UWS\n(n=4)','MCS-\n(n=10)','MCS+\n(n=4)','EMCS\n(n=8)'],ylabel='R-VPL-SMA FC (Fisher z)')
    ax.text(.02,.96,'Spearman rho=0.598; permutation P=0.0012',transform=ax.transAxes,va='top',fontsize=9)
    ax.set_ylim(focus.RVPL_SMA_raw.min()-.07,focus.RVPL_SMA_raw.max()+.16);ptitle(ax,'B','Ordered six-month diagnosis')
    ax=axs[1,0];ax.scatter(focus[feat],focus.delta_CRS_6mo,c=TEAL,s=28)
    xx=np.linspace(focus[feat].min(),focus[feat].max(),100);slope,intercept=np.polyfit(focus[feat],focus.delta_CRS_6mo,1)
    ax.plot(xx,intercept+slope*xx,c=TEAL,lw=1.3);ax.set(xlabel='R-VPL-SMA FC (healthy-reference SD)',ylabel='Six-month change in CRS-R')
    ax.text(.02,.96,'Spearman rho=0.586; permutation P=0.0022',transform=ax.transAxes,va='top',fontsize=9);ptitle(ax,'C','Continuous behavioural improvement')
    ax=axs[1,1];ax.plot(curve.edge_reference_SD,curve.adjusted_probability,c=NAVY,lw=2)
    ax.fill_between(curve.edge_reference_SD,curve.Wald95_low,curve.Wald95_high,color=NAVY,alpha=.15,lw=0)
    ax.set(ylim=(0,1),xlabel='R-VPL-SMA FC (healthy-reference SD)',ylabel='Adjusted probability of MCS+/EMCS')
    ax.text(.02,.97,'OR 4.62 (95% CI 1.23-17.31)\nResidual-permutation P=0.0018',transform=ax.transAxes,va='top',fontsize=9)
    ptitle(ax,'D','Adjusted focused-edge association')
    fig.text(.5,.025,'Complementary outcomes from the same 26 patients; not independent replications. D holds baseline CRS-R and log1p timing at cohort means.',ha='center',fontsize=8.8,color='#606B76')
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.88,wspace=.27,hspace=.42)
    save(fig,'Figure_2_behavioural_convergence',M)
    for num,stem in [(1,'Figure_1_outcome_guided_identification'),(3,'Figure_3_robustness_anatomical_concentration'),(4,'Fig_A2_cross_scale_convergence')]:
        for ext in ['png','pdf']:
            shutil.copy2(V/'figures'/(stem+'.'+ext),M/(f'Figure_{num}.'+ext))


if __name__=='__main__':
    s1();s2();s3();s4();s5();main2()
    info=[]
    for p in list(F.glob('*.pdf'))+list(M.glob('Figure_2*.pdf')):
        with pymupdf.open(p) as doc:
            for k,page in enumerate(doc):
                page.get_pixmap(matrix=pymupdf.Matrix(1.0,1.0)).save(W/'qa'/f'{p.stem}_page{k+1}.png')
            info.append({'file':str(p.relative_to(W)),'pages':len(doc),'all_pages_rendered':True})
    (W/'qa/FIGURE_PDF_QA.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('S1-S5 (S4 two pages) and synchronized main figures exported.',flush=True)
