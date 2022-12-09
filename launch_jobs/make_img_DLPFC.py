from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
import nilearn as nil
    


# Here, we will project a 3D statistical map onto a cortical mesh using vol_to_surf, 
# display a surface plot of the projected map using plot_surf_stat_map with different plotting engines,
# and add contours of regions of interest using plot_surf_contours.


# Get statistical map
# img=nib.load(f'ff.{hemi}.label.gii')



img = nil.image.load_img('/Users/chiara/docker-hcp/Q1-Q6_RelatedParcellation210_tfMRI_ALLTASKS_level3_beta_hp200_s2_MSMAll_2_d41_WRN_DeDrift_norm_grad.dscalar.nii') 
# img = datasets.fetch_neurovault_motor_task()
stat_img = img.images[0]

# Get cortical mesh 
fsaverage = datasets.fetch_surf_fsaverage()


# Sample the 3D data around each node of the mesh
texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)


# Plot the results
fig = plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture, hemi='right',
    title='Surface right hemisphere', colorbar=True,
    threshold=1., bg_map=fsaverage.sulc_right
)
fig.show()