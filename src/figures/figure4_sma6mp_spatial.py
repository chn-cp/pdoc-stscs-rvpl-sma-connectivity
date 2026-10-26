"""Use exact bilateral Julich 3.1 6mp labels; preserve frozen patient maps."""
from pathlib import Path
import hashlib, json, shutil, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import pymupdf

import os
W=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('RVPL_SMA_PROJECT_ROOT', Path.cwd()))
B=Path(os.environ.get('RVPL_SMA_SPATIAL_INPUT', ROOT/'restricted_inputs'/'spatial_convergence'))
OLD=Path(os.environ.get('RVPL_SMA_PREVIOUS_SMA_ANNOTATION', ROOT/'restricted_inputs'/'previous_sma_annotation'))
O=Path(os.environ.get('RVPL_SMA_SPATIAL_OUTPUT', ROOT/'outputs'/'sma6mp_spatial'))
for sub in ['main_figures','supplementary_figures','spatial_data','Author_checks','qa']:
    (O/sub).mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=list(B.glob('*.nii.gz'))+list(B.glob('*.csv'))+list(OLD.rglob('*'))
frozen=[p for p in frozen if p.is_file()]
before={str(p):sha(p) for p in frozen}
jroot_env=os.environ.get('JULICH_ATLAS_ROOT')
if not jroot_env:
    raise RuntimeError('Set JULICH_ATLAS_ROOT to the directory containing JulichBrainAtlas 3.1 MPM NIfTI/XML files.')
jroot=Path(jroot_env);parts=[];sources=[];meta=[]
for h in ['lh','rh']:
    ip=jroot/f'JulichBrainAtlas_3.1_207areas_MPM_{h}_MNI152.nii.gz'
    xp=jroot/f'JulichBrainAtlas_3.1_207areas_MPM_{h}_MNI152.xml'
    atlas=nib.load(ip)
    entry=next(e for e in ET.parse(xp).findall('.//Structure') if 'Area 6mp (' in (e.text or ''))
    assert entry.attrib['grayvalue']=='13'
    if parts:assert atlas.shape==base.shape and np.allclose(atlas.affine,base.affine)
    else:base=atlas
    parts.append(atlas.get_fdata()==13);sources.extend([ip,xp])
    meta.append({'hemisphere':h,'name':entry.text,'attributes':entry.attrib})
mask=nib.Nifti1Image(np.logical_or.reduce(parts).astype('uint8'),base.affine)
assert int(mask.get_fdata().sum())==3952
nib.save(mask,O/'spatial_data/Julich31_bilateral_6mp_SMA_reference_1mm.nii.gz')
native=nib.load(B/'SMA_fixed_sphere_6mm.nii.gz');fixed=native.get_fdata()>0
sm=nib.processing.resample_from_to(mask,(native.shape,native.affine),order=0).get_fdata()>0
nib.save(nib.Nifti1Image(sm.astype('uint8'),native.affine),O/'spatial_data/Julich31_bilateral_6mp_SMA_reference_studygrid.nii.gz')
metrics=pd.read_csv(B/'seed2voxel_sma_convergence_metrics.csv')
clusters=pd.read_csv(B/'seed2voxel_cluster_peak_distances.csv')
oldcounts=pd.read_csv(OLD/'spatial_data/anatomical_SMA_model_counts.csv');rows=[]
for group in ['main','sensitivity']:
    for i,r in metrics.loc[metrics.subject_set.eq(group)].reset_index(drop=True).iterrows():
        im=nib.load(B/f'RVPL_{group}_M{i}_significant_t.nii.gz')
        assert im.shape==native.shape and np.allclose(im.affine,native.affine)
        sig=im.get_fdata()>0
        assert int((sig&fixed).sum())==r.overlap_voxels
        rows.append({'cohort':group,'model':f'M{i}','n':int(r.n),
          'total_significant_voxels':int(sig.sum()),'significant_voxels_in_SMA_6mp':int((sig&sm).sum()),
          'SMA_6mp_studygrid_voxels':int(sm.sum()),
          'original_fixed_ROI_overlap_voxels':int((sig&fixed).sum()),'original_fixed_ROI_total_voxels':114,
          'previous_AAL3_annotation_voxels':int(oldcounts.loc[oldcounts.cohort.eq(group)&oldcounts.model.eq(f'M{i}'),'significant_voxels_in_anatomical_SMA'].iloc[0])})
counts=pd.DataFrame(rows)
assert counts.significant_voxels_in_SMA_6mp.tolist()==[42,38,37,0,0,47,37,38,21,23]
counts.to_csv(O/'spatial_data/SMA_6mp_model_counts.csv',index=False)
ref=load_mni152_template(resolution=1)
GREEN='#009E73';NAVY='#3C5488';TEAL='#008F82';INK='#20262D';MUTED='#606B76'
gray=ListedColormap(plt.get_cmap('gray')(np.linspace(0,1,512)**.60))
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
  'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'axes.spines.top':False,'axes.spines.right':False})
world=nib.affines.apply_affine(mask.affine,np.argwhere(mask.get_fdata()>0));ledger=[]
def brain(fig,rect,img,mode,cut,cmap,lo,hi,label=False):
    ax=fig.add_axes(rect)
    d=plotting.plot_anat(ref,display_mode=mode,cut_coords=[cut],axes=ax,annotate=False,
      draw_cross=False,black_bg=False,cmap=gray,vmin=.02,vmax=.92,threshold=.04,dim=0,
      interpolation='bicubic',colorbar=False,radiological=mode=='z')
    d.add_overlay(img,threshold=.01 if lo>1 else .5,cmap=cmap,vmin=lo,vmax=hi,
      colorbar=False,interpolation='nearest',resampling_interpolation='nearest')
    axis=0 if mode=='x' else 2
    has=bool(world[:,axis].min()<=cut<=world[:,axis].max())
    if has:d.add_contours(mask,levels=[.5],colors=[GREEN],linewidths=1.3,resampling_interpolation='nearest')
    da=next(iter(d.axes.values())).ax
    if mode=='x':
        da.invert_xaxis()
        if label:da.annotate('SMA (6mp)',xy=(-11,62),xytext=(19,85),color=GREEN,
          fontsize=9.4,fontweight='bold',ha='center',arrowprops={'arrowstyle':'-','color':GREEN,'lw':1})
    else:
        da.text(.02,.03,'R',transform=da.transAxes,fontsize=8)
        da.text(.98,.03,'L',ha='right',transform=da.transAxes,fontsize=8)
        if label and has:da.annotate('SMA (6mp)',xy=(4,-12),xytext=(-26,29),color=GREEN,
          fontsize=9.3,fontweight='bold',ha='center',arrowprops={'arrowstyle':'-','color':GREEN,'lw':1})
    ledger.append({'plane':mode,'cut_mm':cut,'contour_present':has,'atlas':'Julich-Brain 3.1 MPM bilateral 6mp',
      'sphere_contour':False,'manual_mask_operations':False})

warm=ListedColormap(['#FEE8C8','#FDBB84','#FC8D59','#E34A33','#B30000'])
fig=plt.figure(figsize=(14,10.2),facecolor='white')
fig.text(.5,.976,'Cross-scale spatial correspondence in medial premotor cortex',ha='center',va='top',fontsize=19,fontweight='bold')
for i,(group,file,letter,title) in enumerate([
  ('main','RVPL_model_consistency_count','A','Main seed-to-voxel cohort (n=27)'),
  ('sensitivity','RVPL_strict_model_consistency_count','B','Retained-frames sensitivity (n=24)')]):
    left=.035+i*.495
    fig.text(left,.91,letter,fontsize=22,fontweight='bold',va='center')
    fig.text(left+.033,.91,title,fontsize=13.2,fontweight='bold',va='center')
    img=nib.load(B/(file+'.nii.gz'))
    for mode,cut,rect in [('x',-4,[left+.004,.555,.272,.315]),('z',65.5,[left+.275,.555,.181,.315])]:
        brain(fig,rect,img,mode,cut,warm,.5,5.5,True)
        fig.text(rect[0]+rect[2]/2,.541,f'{mode} = {cut:g} mm',ha='center',fontsize=10.5)
fig.text(.5,.497,'Green contours: bilateral SMA (6mp), Julich-Brain 3.1; atlas boundaries unchanged',ha='center',fontsize=10,color=GREEN)
fig.text(.5,.469,'Number of nested models with cluster-FWE P < .05',ha='center',fontsize=11,fontweight='bold')
cax=fig.add_axes([.29,.432,.42,.017]);norm=BoundaryNorm(np.arange(.5,6),warm.N)
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=warm),cax=cax,orientation='horizontal',ticks=range(1,6))
cb.outline.set_linewidth(.6);cb.ax.tick_params(length=3,pad=3,labelsize=10)
for i,(group,letter,color) in enumerate([('main','C',NAVY),('sensitivity','D',TEAL)]):
    left=.035+i*.495
    fig.text(left,.379,letter,fontsize=22,fontweight='bold',va='center')
    fig.text(left+.033,.379,'Direct overlap with atlas SMA (6mp)',fontsize=12.8,fontweight='bold',va='center')
    ax=fig.add_axes([left+.055,.108,.394,.23]);vals=counts.loc[counts.cohort.eq(group),'significant_voxels_in_SMA_6mp'].to_numpy()
    ax.bar(range(5),vals,width=.66,color=color)
    for j,v in enumerate(vals):ax.text(j,v+1.3,str(v),ha='center',fontsize=11)
    ax.set(xticks=range(5),xticklabels=[f'M{j}' for j in range(5)],ylim=(0,60),yticks=range(0,61,15),
      ylabel='Significant native-grid voxels',xlim=(-.55,4.55))
    ax.grid(axis='y',color='#E8ECEF',linewidth=.6);ax.set_axisbelow(True)
fig.text(.5,.055,'Counts describe the location of existing significant maps; direct 6mp overlap is absent in main-cohort M3 and M4.',ha='center',fontsize=9.1)
fig.text(.5,.034,'M0: unadjusted; M1: +age; M2: +injury-to-fMRI duration; M3: +baseline CRS-R; M4: +motion.',ha='center',fontsize=9.1)
fig.text(.5,.013,'Original 6-mm coordinate-ROI overlap remains in Table S8; the anatomical reference does not redefine the connectivity feature.',ha='center',fontsize=9,color=MUTED)
for ext in ['png','pdf','svg']:fig.savefig(O/'main_figures'/('Figure_4.'+ext),dpi=360,bbox_inches='tight',pad_inches=.13)
plt.close(fig)

vmax=max(float(nib.load(p).get_fdata().max()) for p in B.glob('RVPL_*_M*_significant_t.nii.gz'))
with PdfPages(O/'supplementary_figures/Figure_S4_detailed_seed_maps.pdf') as pdf:
    for group,title in [('main','Main seed-to-voxel cohort n=27'),('sensitivity','Retained-frames sensitivity cohort n=24')]:
        fig=plt.figure(figsize=(10,12),facecolor='white')
        fig.text(.5,.985,'Supplementary Figure S4  Detailed R-VPL seed-to-voxel maps',ha='center',va='top',fontsize=15,fontweight='bold')
        fig.text(.5,.95,title,ha='center',fontsize=12,color=NAVY)
        fig.text(.5,.925,'Green contours: bilateral SMA (6mp), Julich-Brain 3.1',ha='center',fontsize=9,color=GREEN)
        for i,r in metrics.loc[metrics.subject_set.eq(group)].reset_index(drop=True).iterrows():
            yy=.76-i*.16;hits=clusters.loc[clusters.subject_set.eq(group)&clusters.model.eq(r.model)].sort_values('fwe')
            z=float(hits.iloc[0].z);img=nib.load(B/f'RVPL_{group}_M{i}_significant_t.nii.gz')
            fig.text(.025,yy+.068,f'M{i}',fontsize=13,fontweight='bold')
            fig.text(.025,yy+.041,f'n={r.n}\ndf={r.df}',fontsize=8.6,linespacing=1.4)
            for mode,cut,rect in [('x',-4,[.11,yy,.27,.137]),('z',z,[.40,yy,.23,.137])]:
                brain(fig,rect,img,mode,cut,'YlOrRd',3.4,vmax,label=(i==0 and mode=='x'))
                fig.text(rect[0]+rect[2]/2,yy-.009,f'{mode}={cut:g} mm',ha='center',fontsize=8)
            lines=[f'C{k}: {h.size} voxels; FWE P={h.fwe:.4f}\nPeak ({h.x:g}, {h.y:g}, {h.z:g}); T={h.peak_t:.2f}' for k,h in enumerate(hits.itertuples(),1)]
            fig.text(.66,yy+.115,'\n'.join(lines),ha='left',va='top',fontsize=8.4,linespacing=1.35)
            cnt=int(counts.loc[counts.cohort.eq(group)&counts.model.eq(f'M{i}'),'significant_voxels_in_SMA_6mp'].iloc[0])
            fig.text(.66,yy+.010,f'Significant voxels in SMA (6mp): {cnt}',fontsize=8.8,color=GREEN)
        cax=fig.add_axes([.22,.065,.38,.009]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(3.4,vmax),cmap='YlOrRd'),cax=cax,orientation='horizontal')
        fig.text(.41,.039,'Positive t statistic in archived significant clusters',ha='center',fontsize=8.5)
        fig.text(.5,.018,'No contour appears on axial sections above the atlas parcel. Original coordinate-ROI overlap remains in Table S8.',ha='center',fontsize=8.1,color=MUTED)
        pdf.savefig(fig,bbox_inches='tight',pad_inches=.1)
        fig.savefig(O/'supplementary_figures'/f'Figure_S4_{group}.png',dpi=300,bbox_inches='tight',pad_inches=.1)
        plt.close(fig)
for folder,exclude in [('main_figures','Figure_4'),('supplementary_figures','Figure_S4')]:
    for p in (OLD/folder).iterdir():
        if p.is_file() and not p.name.startswith(exclude):shutil.copy2(p,O/folder/p.name)
shutil.copy2(OLD/'Supplementary_Data_1_complete_435_edge_outcome_screen.csv',O/'Supplementary_Data_1_complete_435_edge_outcome_screen.csv')
assert before=={p:sha(Path(p)) for p in before}
qc={'frozen_input_sha256':before,'source_atlas_sha256':{str(p):sha(p) for p in sources},
    'structures':meta,'mask_matches_bilateral_grayvalue13_exactly':True,'native_1mm_voxels':3952,
    'studygrid_voxels':int(sm.sum()),'resampling':'nearest-neighbour, no additional deformation',
    'display_ledger':ledger,'new_counts':counts.significant_voxels_in_SMA_6mp.tolist(),
    'scientific_scope':'Post hoc atlas reference only, not a new predictor or corrected ROI test.',
    'template_scope':'MNI template-level reference; Julich ICBM2009c distribution, not subject-specific anatomy.',
    'reference':'https://doi.org/10.1007/s00429-018-1738-6'}
(O/'qa/FIGURE_SPATIAL_QC.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
for file in [O/'main_figures/Figure_4.pdf',O/'supplementary_figures/Figure_S4_detailed_seed_maps.pdf']:
    with pymupdf.open(file) as pdf:
        for i,page in enumerate(pdf):page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(O/'qa'/f'{file.stem}_page{i+1}.png')
print(counts.to_string(index=False),flush=True)
print('Updated Figure 4 and S4. All original inputs and previous package remain unchanged.',flush=True)
