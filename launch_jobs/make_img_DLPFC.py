from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
import nilearn as nil
import nibabel as nib


imgr = nib.load('/home/ubuntu/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')


# [for right]
fig = plotting.plot_surf_stat_map('Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii', imgr.get_fdata(), hemi='right',threshold=1.,title='Surface right hemisphere', colorbar=True)

print('hi')

# fig = plotting.plot_surf_stat_map(fsaverage.infl_right, imgr.get_fdata(), hemi='right',threshold=1., bg_map=fsaverage.sulc_right,title='Surface right hemisphere', colorbar=True)
