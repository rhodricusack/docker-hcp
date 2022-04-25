import boto3
import nilearn
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import logging
from botocore.exceptions import ClientError
import argparse
import msgpack
import msgpack_numpy as m
from os import path

import pandas as pd

sublist = ['178950','189450', '199453','209228','220721','298455','356948','419239','499566','561444','618952','680452','757764','841349','908860', '103818','113922','121618','130619','137229','151829','158035','171633','179346','190031','200008','210112','221319','299154','361234', '424939','500222','570243','622236','687163','769064','845458','911849','104416','114217','122317','130720','137532','151930','159744', '172029','180230','191235','200614','211316','228434','300618','361941','432332','513130','571144','623844','692964','773257','857263', '926862','105014','114419','122822','130821','137633','152427','160123','172938','180432','192035','200917','211417','239944','303119', '365343','436239','513736','579665','638049','702133','774663','865363','930449','106521','114823','123521','130922','137936','152831', '160729','173334','180533','192136','201111','211619','249947','305830','366042','436845','516742','580650','645450','715041','782561', '871762','942658','106824','117021','123925','131823','138332','153025','162026','173536','180735','192439','201414','211821','251833', '310621','371843','445543','519950','580751','647858','720337','800941','871964','955465','107018','117122','125222','132017','138837', '153227','162329','173637','180937','193239','201818','211922','257542','314225','378857','454140','523032','585862','654350','725751', '803240','872562','959574','107422','117324','125424','133827','142828','153631','164030','173940','182739','194140','202719','212015', '257845','316633','381543','459453','525541','586460','654754','727553','812746','873968','966975']

session = boto3.Session(profile_name='hcp')

s3 = session.client('s3')

taskcondictnoneg = {
    'tfMRI_WM': {5:'face', 6:'place', 7:'tool'}, # 0 2BK_BODY, 1 2BK_FACE, 2 2BK_PLACE, 3 2BK_TOOL, 4 0BK_BODY, 5 0BK_FACE, 6 0BK_PLACE, 7 0BK_TOOL, 8 2BK, 9 0BK, 10 2BK-0BK, 11 neg_2BK, 12 neg_0BK, 13 0BK-2BK, 14 BODY, 15 FACE, 16 PLACE, 17 TOOL, 18 BODY-AVG, 19 FACE-AVG, 20 PLACE-AVG, 21 TOOL-AVG, 22 neg_BODY, 23 neg_FACE, 24 neg_PLACE, 25 neg_TOOL, 26 AVG-BODY, 27 AVG-FACE, 28 AVG-PLACE, 29 VG-TOOL
    }

df=pd.DataFrame()

# All ROIS
roilist=[7,163,22,18,153,154,160]
roilist.extend([x+180 for x in roilist])

# Load up ROI files for L and R
roi_L_img=nib.load('rois/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')
roi_R_img=nib.load('rois/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')
roi_L_dat=roi_L_img.get_fdata().ravel().astype(int)
roi_R_dat=180 + roi_R_img.get_fdata().ravel().astype(int)
roi_dat=np.concatenate((roi_L_dat,roi_R_dat))

# Handy later 
nroi = len(roilist)
ntask = len(taskcondictnoneg)

# Initialise a space for the output summary values
meanact={}

        

for task, taskcon in taskcondictnoneg.items():
    meanact[task]={}
    for sub in sublist:
        # For each task
        # Download file from HCP S3. 
        hcpbucket = 'hcp-openaccess'
        hcpkey = f'HCP_1200/{sub}/MNINonLinear/Results/{task}/{task}_hp200_s2_level2.feat/{sub}_{task}_level2_hp200_s2.dscalar.nii'
        try:
            s3.download_file(hcpbucket, hcpkey, '/tmp/timeseries.nii')     
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f'Not found on s3 bucket:{hcpbucket} key:{hcpkey}')
                raise
            else:
                print("Unexpected error: %s" % e)
                raise

        task_img = nib.load('/tmp/timeseries.nii')

        # Pick out only voxels on the cortical surface
        task_dat=task_img.get_fdata()
        task_surfmask=task_img.header.get_axis(1).surface_mask
        task_dat_surf=task_dat[:,task_surfmask]


        # For each ROI, summarise activity in the selected regions and contrasts for this task
        for onetask,taskname in taskcon.items():
            meanact={}
            meanact['sub']=sub
            meanact['con_0bk']=taskname
            for roiind, roi in enumerate(roilist):
                sel=task_dat_surf[:, roi_dat == roi]
                meanact[f"roi-{roi:03d}"] = np.mean(sel, 1)[onetask]
            df=df.append(meanact, ignore_index=True)

df.to_csv('hcp_contasts.csv')
print(df)
