"""New Figure 1: reproduce the archived outcome-guided screen, not a new search."""
from __future__ import annotations
import ast
import gzip
import io
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, PathPatch, Wedge
from matplotlib.path import Path as MPath
from matplotlib import cm
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import pandas as pd
from scipy import stats
import pymupdf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verification'))
from core_verification import OUT, ROOT, sha, save_json

LOG=OUT/'logs/figure1_outcome_guided_20261020'
DATA=OUT/'tables/figure1_outcome_guided'
FIG=OUT/'figures'
for p in [LOG,DATA]:p.mkdir(parents=True,exist_ok=True)
INK='#20262D';MUTED='#66717C';BLUE='#0072B2';ORANGE='#E69F00'
NAVY='#3C5488';CORAL='#E64B35'
NET_ORDER=['Subcortical','Sensorimotor','DMN','ExecuContr','Salience','Visual','Auditory']
COL=dict(zip(NET_ORDER,['#9471BC','#548CC9','#57A76C','#E5A14A','#B976AD','#C7B343','#47A9AE']))
SHORT=dict(zip(NET_ORDER,['Subc.','SM','DMN','ECN','SAL','VIS','AUD']))
LABEL=dict(zip(NET_ORDER,['Subcortical','Sensorimotor','Default mode','Executive control','Salience','Visual','Auditory']))
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.8,'text.color':INK,
 'axes.labelcolor':INK,'xtick.labelsize':10,'ytick.labelsize':10})


def mesh(path):
    with Path(path).open() as f:
        def number():
            for line in f:
                if line.strip() and not line.lstrip().startswith('#'):return int(line)
            raise ValueError('Unexpected end of mesh')
        n=number();v=np.loadtxt(f,max_rows=n);m=number();t=np.loadtxt(f,max_rows=m,dtype=int)-1
    assert t.max()<len(v)
    return v,t


def prepare():
    node=pd.read_csv(OUT/'00_inputs/FCmat_work/node_info.csv')
    names=node.name.tolist();net=dict(zip(node.name,node.network))
    assert len(node)==30 and node.name.nunique()==30
    counts=node.network.value_counts().to_dict()
    assert counts=={'Subcortical':8,'ExecuContr':5,'DMN':4,'Visual':4,'Sensorimotor':3,'Auditory':3,'Salience':3}
    with ZipFile('./restricted_inputs/archive/FCmat_work.zip') as z:
        raw=z.read('FCmat_work/edge_stats.csv')
    (DATA/'archived_edge_stats.csv').write_bytes(raw)
    e=pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig')
    d=pd.read_csv(OUT/'00_inputs/master_imaging_cohort_n30.csv')
    order=pd.read_csv(OUT/'00_inputs/FCmat_work/patient_order.csv').Subject.tolist()
    d=d.set_index('Subject').loc[order].reset_index()
    mats=np.array([np.load(OUT/'00_inputs/FCmat_work/patient_mats'/(s+'.npy')) for s in order])
    gr=d.favourable6.to_numpy(int)
    assert len(d)==30 and np.bincount(gr).tolist()==[17,13]
    qc=pd.read_csv(OUT/'00_inputs/SPM_SecondLevel_Analysis/subject_master_QC.csv').set_index('Subject')
    assert np.array_equal(gr,qc.loc[order,'group'].to_numpy(int))
    diff=np.nanmean(mats[gr==1],axis=0)-np.nanmean(mats[gr==0],axis=0)
    index={n:i for i,n in enumerate(names)}
    checks=[]
    for _,r in e.iterrows():
        vals=mats[:,index[r.nodeA],index[r.nodeB]]
        a=vals[(gr==1)&np.isfinite(vals)];b=vals[(gr==0)&np.isfinite(vals)]
        w=stats.ttest_ind(a,b,equal_var=False,alternative='two-sided')
        assert len(a)+len(b)==r.n
        assert np.isclose(w.pvalue,r.p_resp_vs_non,rtol=1e-9,atol=1e-13)
        assert np.isclose(round(a.mean(),3),r.resp_mean,atol=1e-10)
        assert np.isclose(round(b.mean(),3),r.non_mean,atol=1e-10)
        checks.append({'nodeA':r.nodeA,'nodeB':r.nodeB,'n_favourable':len(a),
                       'n_unfavourable':len(b),'difference_Fisher_z':a.mean()-b.mean(),
                       'Welch_t':w.statistic,'p_two_sided_reproduced':w.pvalue})
    checks=pd.DataFrame(checks)
    assert len(checks)==435
    e=e.merge(checks,on=['nodeA','nodeB'],validate='one_to_one')
    sig=e.loc[e.p_resp_vs_non.lt(.05)].copy()
    sig['focused']=sig.apply(lambda r:set([r.nodeA,r.nodeB])=={'R_VPL','SMA'},axis=1)
    assert len(sig)==15 and sig.focused.sum()==1 and (e.q_resp<.05).sum()==0
    selected=sig.loc[sig.focused].iloc[0]
    assert selected.p_resp_vs_non==e.p_resp_vs_non.min()
    i,j=index['R_VPL'],index['SMA'];focus=d.loc[np.isfinite(mats[:,i,j])].copy()
    focus['raw_Fisher_z']=mats[np.isfinite(mats[:,i,j]),i,j]
    sealed=pd.read_csv(OUT/'00_inputs/master_focused_cohort_n26.csv').set_index('Subject')
    assert np.allclose(focus.raw_Fisher_z,sealed.loc[focus.Subject,'RVPL_SMA_raw'])
    assert focus.favourable6.value_counts().sort_index().tolist()==[14,12]
    sig.to_csv(DATA/'all_15_archived_nominal_edges.csv',index=False)
    e.to_csv(DATA/'all_435_screening_edges_verified.csv',index=False)
    focus[['Subject','favourable6','raw_Fisher_z']].to_csv(DATA/'focused_edge_patient_values.csv',index=False)
    display_order=sorted(names,key=lambda n:(NET_ORDER.index(net[n]),n))
    o=[index[n] for n in display_order]
    pd.DataFrame(diff[np.ix_(o,o)],index=display_order,columns=display_order).to_csv(DATA/'matrix_group_difference_Fisher_z.csv')

    # Exact fixed-parcellation coordinate lineage: source spheres and original
    # overlap-precedence masks, not hand-placed anatomical points.
    source=OUT/'00_inputs/source_scripts/connectivity_matrix_pipeline.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    constants={}
    for item in tree.body:
        if isinstance(item,ast.Assign):
            for target in item.targets:
                if isinstance(target,ast.Name) and target.id in ['CORTICAL','NUCLEI']:
                    constants[target.id]=ast.literal_eval(item.value)
    ref=nib.load(OUT/'00_inputs/group_explicit_mask_90pct.nii.gz')
    world=nib.affines.apply_affine(ref.affine,np.indices(ref.shape).reshape(3,-1).T).reshape(*ref.shape,3)
    occupied=np.zeros(ref.shape,bool);coords={}
    for network,name,x,y,z in constants['CORTICAL']:
        centre=np.array([x,y,z]);mask=(np.sum((world-centre)**2,axis=-1)<=36)&~occupied
        occupied|=mask;coords[name]=centre
    with ZipFile('./restricted_inputs/archive/ROI.zip') as z:
        for name,files in constants['NUCLEI'].items():
            imgs=[nib.Nifti1Image.from_bytes(gzip.decompress(z.read('ROI/'+f))) for f in files]
            union=np.logical_or.reduce([im.get_fdata()>.5 for im in imgs])
            im=nib.Nifti1Image(union.astype('float32'),imgs[0].affine)
            mask=(resample_to_img(im,ref,interpolation='linear',force_resample=True,copy_header=True).get_fdata()>=.3)&~occupied
            occupied|=mask;coords[name]=world[mask].mean(axis=0)
    node['x_MNI']=[coords[n][0] for n in names]
    node['y_MNI']=[coords[n][1] for n in names]
    node['z_MNI']=[coords[n][2] for n in names]
    node.to_csv(DATA/'node_coordinates_and_networks.csv',index=False)
    save_json(LOG/'source_verification.json',{'all_435_archived_Welch_P_reproduced':True,
      'archive':'./restricted_inputs/archive/FCmat_work.zip::FCmat_work/edge_stats.csv',
      'network_counts':counts,'screening_n':30,'screening_group_n':[17,13],
      'nominal_edges':len(sig),'connectome_BH_q_lt_05_edges':0,
      'candidate_unadjusted_P':float(selected.p_resp_vs_non),'candidate_connectome_BH_q':float(selected.q_resp),
      'candidate_group_n':[14,12],'candidate_is_minimum_nominal_P':True,
      'matrix_color_limit_original_98th_percentile':float(np.nanpercentile(np.abs(diff),98)),
      'master_focused_sha256':sha(OUT/'00_inputs/master_focused_cohort_n26.csv'),
      'screen_source_script_sha256':sha(ROOT/'work/upload_ready_20260929/FIG3_BCD_outcome_terms.py'),
      'archived_edge_stats_sha256':sha(DATA/'archived_edge_stats.csv')})
    return node,display_order,diff[np.ix_(o,o)],sig,focus,selected


def projected_brain(fig,rect,vertices,faces,node,view,title):
    """Surface backdrop with explicit 2D coordinate projections, not localizations."""
    ax=fig.add_axes(rect);ax.set_aspect('equal');ax.axis('off')
    tri=vertices[faces]
    if view=='superior':
        dims=[0,1];camera=np.array([0,0,1.]);light=np.array([-.35,-.3,.9])
    else:
        dims=[1,2];camera=np.array([1. if view=='right' else -1.,0,0]);light=np.array([camera[0]*.7,-.35,.75])
    light/=np.linalg.norm(light)
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    centroids=tri.mean(axis=1)
    if np.median(np.sum(normals*(centroids-np.mean(vertices,axis=0)),axis=1))<0:normals*=-1
    brightness=.66+.25*np.maximum(normals@light,0)+.06*np.maximum(normals@camera,0)
    colors=np.repeat(brightness[:,None],3,axis=1)
    depth=centroids@camera;ix=np.argsort(depth)
    col=PolyCollection(tri[ix][:,:,dims],facecolors=colors[ix],edgecolors='none',
                       linewidths=0,rasterized=True,zorder=1)
    ax.add_collection(col)
    points=node.copy()
    if view!='superior':
        # All markers are projected onto the viewing plane, including deep nodes.
        hemi=-1 if 'Left' in title else 1
        points=points.loc[(points.x_MNI*hemi>=0)|(points.name.eq('SMA'))]
        xy=points[['y_MNI','z_MNI']].to_numpy()
        ax.set(xlim=(80,-110) if view=='right' else (-110,80),ylim=(-50,90))
    else:
        xy=points[['x_MNI','y_MNI']].to_numpy();ax.set(xlim=(78,-78),ylim=(-110,75))
    colors=[COL[n] for n in points.network]
    ax.scatter(xy[:,0],xy[:,1],s=15,facecolors=colors,edgecolors='white',linewidths=.3,zorder=5,alpha=.95)
    for name,color in [('SMA',ORANGE),('R_VPL',BLUE)]:
        if name not in points.name.tolist():continue
        p=points.loc[points.name.eq(name)].iloc[0]
        xx,yy=(p.x_MNI,p.y_MNI) if view=='superior' else (p.y_MNI,p.z_MNI)
        ax.scatter([xx],[yy],s=34,color=color,edgecolors=INK,lw=.7,zorder=7)
        if (name=='SMA' and title=='Left medial') or (name=='R_VPL' and view=='superior'):
            target=(xx+32,yy+30) if name=='SMA' else (xx-48,yy-35)
            ax.annotate(name,xy=(xx,yy),xytext=target,color=color,fontweight='bold',fontsize=8.8,
                ha='center',arrowprops={'arrowstyle':'-','color':color,'lw':.8},zorder=8)
    ax.text(.5,-.025,title,transform=ax.transAxes,ha='center',va='top',fontsize=7.2,color=MUTED)


def panel(fig,letter,title,x,y,subtitle=None):
    fig.text(x,y,letter,fontsize=23,fontweight='bold',va='center')
    fig.text(x+.035,y,title,fontsize=13,fontweight='bold',va='center')
    if subtitle:fig.text(x+.035,y-.025,subtitle,fontsize=9.4,color=MUTED,va='center')


def build():
    node,order,diff,sig,focus,candidate=prepare()
    net=dict(zip(node.name,node.network));bounds={};k=0
    for n in NET_ORDER:
        cnt=sum(net[name]==n for name in order);bounds[n]=(k,k+cnt);k+=cnt
    fig=plt.figure(figsize=(16.4,11.8),facecolor='white')
    fig.text(.5,.986,'Figure 1 | Outcome-guided identification of R_VPL-SMA',
             ha='center',va='top',fontsize=20,fontweight='bold')
    fig.text(.5,.957,'as a candidate thalamic-premotor connection',
             ha='center',va='top',fontsize=19,fontweight='bold')
    fig.text(.5,.918,'Predefined connectome  →  Exploratory screen  →  Candidate edge  →  Focused follow-up',
             ha='center',va='center',fontsize=10,color=MUTED)
    panel(fig,'A','Predefined ROI search space',.025,.876)
    panel(fig,'B','Connectome-wide outcome-group differences',.535,.876,
          'MCS+/EMCS minus below MCS+ at 6 months')
    panel(fig,'C','Exploratory candidate-edge map',.025,.454,
          'All 15 outcome-associated edges with uncorrected P < 0.05')
    panel(fig,'D','Focused R_VPL-SMA candidate',.535,.454,
          'Pretreatment connectivity by six-month outcome group')

    # A: counts and actual original network memberships, with orientation-only focus.
    box=FancyBboxPatch((.034,.685),.126,.153,boxstyle='round,pad=.007,rounding_size=.009',
                      transform=fig.transFigure,facecolor='#F3F6FA',edgecolor='#DAE2ED',lw=1)
    fig.add_artist(box)
    fig.text(.097,.815,'30 nodes',ha='center',fontsize=15,fontweight='bold')
    fig.text(.097,.787,'22 cortical ROIs\n8 subcortical ROIs',ha='center',va='center',fontsize=10,linespacing=1.45)
    fig.text(.097,.748,'↓',ha='center',fontsize=16,color=MUTED)
    fig.text(.097,.721,'435 pairwise edges',ha='center',fontsize=11.2,fontweight='bold')
    fig.text(.097,.699,'30 × 29 / 2',ha='center',fontsize=9.5,color=MUTED)
    counts=node.network.value_counts()
    for i,n in enumerate(NET_ORDER):
        yy=.66-i*.0215
        fig.add_artist(plt.Line2D([.038],[yy],marker='o',ms=7,color=COL[n],transform=fig.transFigure,ls='none'))
        fig.text(.049,yy,f'{LABEL[n]} (n={counts[n]})',fontsize=8.8,va='center')
    surface_root=os.environ.get('BRAINNET_SURFACE_ROOT')
    if not surface_root:
        raise RuntimeError('Set BRAINNET_SURFACE_ROOT to the BrainNet Viewer SurfTemplate directory.')
    mp=Path(surface_root)
    vl,tl=mesh(mp/'BrainMesh_ICBM152Left.nv');vr,tr=mesh(mp/'BrainMesh_ICBM152Right.nv')
    bothv=np.vstack([vl,vr]);botht=np.vstack([tl,tr+len(vl)])
    projected_brain(fig,[.185,.692,.144,.150],vl,tl,node,'left','Left lateral')
    projected_brain(fig,[.338,.692,.144,.150],vr,tr,node,'right','Right lateral')
    projected_brain(fig,[.175,.522,.114,.145],vl,tl,node,'right','Left medial')
    projected_brain(fig,[.290,.514,.103,.153],bothv,botht,node,'superior','Superior')
    projected_brain(fig,[.396,.522,.114,.145],vr,tr,node,'left','Right medial')
    fig.text(.335,.494,'ROI coordinate projections; R_VPL/SMA highlighted for orientation only.',
             ha='center',fontsize=7.6,color=MUTED)

    # B: original matrix reconstruction and original 98th-percentile display limit.
    ax=fig.add_axes([.647,.489,.239,.332])
    vmax=float(np.nanpercentile(np.abs(diff),98))
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#C9CDD1')
    shown=np.ma.masked_where(np.eye(30,dtype=bool),diff)
    im=ax.pcolormesh(np.arange(31)-.5,np.arange(31)-.5,shown,
                    cmap=cmap,vmin=-vmax,vmax=vmax,edgecolors='#F0F0F0',linewidth=.16,rasterized=False)
    ax.set(xlim=(-.5,29.5),ylim=(29.5,-.5),aspect='equal',xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_visible(False)
    for n,(s,e) in bounds.items():
        ax.add_patch(plt.Rectangle((s-.5,-1.72),e-s,1.05,facecolor=COL[n],edgecolor='white',lw=.5,clip_on=False))
        ax.add_patch(plt.Rectangle((-1.39,s-.5),.72,e-s,facecolor=COL[n],edgecolor='white',lw=.5,clip_on=False))
        ax.text((s+e-1)/2,-1.19,SHORT[n],ha='center',va='center',fontsize=6.8,color='white',fontweight='bold')
        ax.text(-1.77,(s+e-1)/2,LABEL[n],ha='right',va='center',fontsize=8.0,color=INK)
        if e<30:
            ax.axvline(e-.5,c='#727B85',lw=.45,alpha=.45)
            ax.axhline(e-.5,c='#727B85',lw=.45,alpha=.45)
    cbax=fig.add_axes([.912,.535,.011,.267])
    cb=fig.colorbar(im,cax=cbax);cb.set_ticks([-.3,-.15,0,.15,.3]);cb.ax.tick_params(labelsize=8,length=2)
    cb.outline.set_visible(False)
    fig.text(.949,.668,'Group-mean difference\n(Fisher z)',rotation=90,ha='center',va='center',fontsize=9)
    fig.text(.766,.474,'30-patient screen; edge-wise available observations',ha='center',fontsize=8.1,color=MUTED)

    # C: the complete archived 15-edge set, with a single presentation highlight.
    ax=fig.add_axes([.028,.062,.345,.348]);ax.set_aspect('equal');ax.axis('off')
    angles={n:np.deg2rad(135-360*i/30) for i,n in enumerate(order)}
    pos={n:np.array([np.cos(a),np.sin(a)]) for n,a in angles.items()}
    for n,(s,e) in bounds.items():
        a1=135-360*(e-.5)/30;a2=135-360*(s-.5)/30
        ax.add_patch(Wedge((0,0),1.065,a1,a2,width=.046,facecolor=COL[n],edgecolor='white',lw=.7,zorder=2))
        # Ring colours match A/B. Omit repeated group labels to keep all 30
        # individual node labels readable rather than colliding with the ring.
    norm=Normalize(-vmax,vmax);cmap=plt.get_cmap('RdBu_r')
    sig=sig.sort_values(['focused','p_resp_vs_non'],ascending=[True,False])
    for _,r in sig.iterrows():
        p1,p2=pos[r.nodeA],pos[r.nodeB];ctrl=(p1+p2)*.11
        path=MPath([p1,ctrl,p2],[MPath.MOVETO,MPath.CURVE3,MPath.CURVE3])
        width=.65+1.4*(-np.log10(r.p_resp_vs_non))/(-np.log10(candidate.p_resp_vs_non))
        color=cmap(norm(r.difference_Fisher_z))
        if r.focused:
            ax.add_patch(PathPatch(path,fill=False,ec='white',lw=5.0,zorder=6))
            ax.add_patch(PathPatch(path,fill=False,ec=CORAL,lw=3.0,zorder=7))
        else:ax.add_patch(PathPatch(path,fill=False,ec=color,lw=width,alpha=.45,zorder=3))
    for n in order:
        x,y=pos[n];focused=n in ['R_VPL','SMA']
        color=BLUE if n=='R_VPL' else ORANGE if n=='SMA' else COL[net[n]]
        ax.scatter(x,y,s=65 if focused else 22,c=color,edgecolors=INK if focused else 'white',
                   lw=.9 if focused else .5,zorder=8)
        # Radial text remains editable; other exploratory nodes retain equal weight.
        theta=np.degrees(angles[n]);ha='left' if x>=0 else 'right'
        rotation=theta if x>=0 else theta+180
        ax.text(1.115*x,1.115*y,n,ha=ha,va='center',rotation=rotation,
                rotation_mode='anchor',fontsize=8.4 if focused else 6.9,
                fontweight='bold' if focused else 'normal',color=color if focused else '#505B66')
    ax.set(xlim=(-1.58,1.58),ylim=(-1.49,1.49))
    # Explicit encoding: background links retain signed mean differences and P widths.
    fig.text(.382,.381,'Edge difference',fontsize=9.1,fontweight='bold')
    fig.text(.382,.365,'MCS+/EMCS minus\nbelow MCS+ (Fisher z)',fontsize=8.2,linespacing=1.35,va='top')
    cax=fig.add_axes([.391,.221,.010,.112]);cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    cb.set_ticks([-.3,0,.3]);cb.ax.tick_params(labelsize=8,length=2);cb.outline.set_visible(False)
    fig.text(.382,.188,'Width follows -log10(P)',fontsize=8.2)
    fig.add_artist(plt.Line2D([.382,.409],[.157,.157],color=CORAL,lw=3,transform=fig.transFigure))
    fig.text(.417,.157,'Focused edge',fontsize=8.5,va='center')
    fig.text(.382,.125,'All other nominal\nedges retained',fontsize=8.5,color=MUTED,linespacing=1.4)

    # D: same raw FC values as the original screen; two groups, no extra analyses.
    ax=fig.add_axes([.636,.103,.302,.285])
    values=[focus.loc[focus.favourable6.eq(i),'raw_Fisher_z'].to_numpy() for i in [0,1]]
    parts=ax.violinplot(values,positions=[0,1],widths=.72,showmeans=False,showmedians=False,
                        showextrema=False,points=150,bw_method='scott')
    for body,c in zip(parts['bodies'],[NAVY,CORAL]):
        body.set(facecolor=c,edgecolor='none',alpha=.16)
    bp=ax.boxplot(values,positions=[0,1],widths=.18,patch_artist=True,showfliers=False,
                  medianprops={'color':INK,'lw':1.5},boxprops={'lw':1.1},
                  whiskerprops={'lw':1.0},capprops={'lw':1.0})
    for b,c in zip(bp['boxes'],[NAVY,CORAL]):b.set(facecolor='white',edgecolor=c,alpha=.9)
    rng=np.random.default_rng(20261020)
    for k,(v,c) in enumerate(zip(values,[NAVY,CORAL])):
        ax.scatter(k+rng.uniform(-.105,.105,len(v)),v,s=30,c=c,edgecolors='white',lw=.35,alpha=.92,zorder=5)
    ax.axhline(0,c='#949CA3',ls='--',lw=.8)
    upper=float(max(np.max(v) for v in values));by=upper+.13
    ax.plot([0,0,1,1],[by-.025,by,by,by-.025],c=INK,lw=.9)
    ax.text(.5,by+.023,f'Two-sided Welch P = {candidate.p_resp_vs_non:.4f}',ha='center',va='bottom',fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['Below MCS+\n(n=14)','MCS+/EMCS\n(n=12)'],
           xlim=(-.55,1.55),ylim=(min(np.min(v) for v in values)-.11,upper+.34),
           ylabel='Pretreatment R_VPL-SMA connectivity\n(Fisher z)')
    ax.tick_params(axis='x',length=0,pad=9)
    fig.text(.787,.044,'Candidate selected for focused follow-up',ha='center',fontsize=9.6,
             color=BLUE,fontweight='bold')
    fig.text(.5,.015,'Exploratory, outcome-guided selection; focused follow-up uses the same patient cohort.',
             ha='center',fontsize=9,color=MUTED)
    stem='Figure_1_outcome_guided_identification'
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/(stem+'.'+ext),dpi=400,bbox_inches='tight',pad_inches=.13,facecolor='white')
    plt.close(fig)
    with pymupdf.open(FIG/(stem+'.pdf')) as pdf:
        assert len(pdf)==1
        text=pdf[0].get_text();assert '435' in text and '0.0008' in text and '(n=14)' in text
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.35,1.35)).save(LOG/'FINAL_PDF_RENDER.png')
        save_json(LOG/'PDF_QA.json',{'pages':len(pdf),'extractable_text_characters':len(text),
             'embedded_surface_images':len(pdf[0].get_images()),'charts_and_text_vector_editable':True})
    print('New Figure 1 exported as PNG/PDF/SVG. All 435 archived comparisons reproduced.',flush=True)


if __name__=='__main__':build()
