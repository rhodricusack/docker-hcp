import nibabel as nib
import numpy as np
import os
import boto3
from matplotlib import pyplot as plt
from nilearn import plotting
import hcp_utils as hcp

# Credentials for uploading and downloading data to the cusack lab s3
s3 = boto3.client('s3')
session = boto3.Session(profile_name='default')

taskcondict_selected = {
    'tfMRI_WM': [8],  # 8 2BK
    'tfMRI_MOTOR': [6],  # 6 AVG
    'tfMRI_LANGUAGE': [1],  # STORY
    'tfMRI_SOCIAL': [1],     # TOM
    'tfMRI_EMOTION': [1]     # SHAPES
}

nsub=155

analysis_root = '/home/chiaracaldinelli'

fig,ax=plt.subplots(subplot_kw={'projection': '3d'}, nrows=len(taskcondict_selected.keys()), figsize=(3.5,10))

for taskind, (task, taskcons) in enumerate(taskcondict_selected.items()):
    print(task)
        
    # Same for both hemispheres so outside hemi loop
    # Download activation for this task
    remotepath_act = (f'Results/act_for_classifier.npy')
    s3.download_file('smartontheinside', remotepath_act, f'act_for_classifier_{task}_N-{nsub}.npy')
    act = np.load(os.path.join(analysis_root, f'act_for_classifier_{task}_N-{nsub}.npy'), allow_pickle=True).ravel()[0]

    for hemi in ['left','right']:

        frontal = nib.load(f'/home/chiaracaldinelli/smartontheinside/smartontheinside/frontal.{hemi[0].upper()}.label.gii')
        dat = frontal.agg_data('NIFTI_INTENT_LABEL')

        # Put the frontal values in 
        dat[dat==1] = np.mean(act[hemi[0].upper()], axis=0)

        # fig = plotting.plot_surf_stat_map(f'Q1-Q6_RelatedParcellation210.{hemi[0].upper()}.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii', dat, view='dorsal', bg_map=hcp.mesh[f'sulc_{hemi}'], hemi=hemi,threshold=0.00001,title=f'{task}', colorbar=(hemi=='right') and (taskind==0), axes=ax[taskind], darkness=0.75, symmetric_cbar=True, vmax=5, title_font_size=10)
        # No colorbar
        fig = plotting.plot_surf_stat_map(f'Q1-Q6_RelatedParcellation210.{hemi[0].upper()}.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii', dat, view='dorsal', bg_map=hcp.mesh[f'sulc_{hemi}'], hemi=hemi, threshold=0.00001,title=f'{task}', colorbar=False, axes=ax[taskind], darkness=0.75, symmetric_cbar=True, vmax=5, title_font_size=10)

plt.savefig(f'DLPFC_contrasts_no_colorbar.png')
print('All done!')