from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
import nilearn as nil
import nibabel as nib
import numpy as np

from matplotlib import pyplot as plt

for hemi in ['left','right']:
    frontal = nib.load(f'/home/chiaracaldinelli/smartontheinside/smartontheinside/frontal.{hemi[0].upper()}.label.gii')
    dat = frontal.agg_data('NIFTI_INTENT_LABEL')
    print(f'Hemisphere {hemi} frontal voxels {np.sum(dat)}')
    fig = plotting.plot_surf_stat_map(f'Q1-Q6_RelatedParcellation210.{hemi[0].upper()}.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii', dat, hemi=hemi,threshold=1.,title=f'Surface {hemi} hemisphere', colorbar=True)
    plt.savefig(f'test_{hemi}.png')

