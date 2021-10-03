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

def parse_args():
    parser = argparse.ArgumentParser(description='ROI Extraction from HCP Data')

    parser.add_argument('--subject', type=str, default=None, help='HCP subject number')
    parser.add_argument('--profile_name', type=str, default=None, help='Profile in aws credentials')
    parser.add_argument('--output_bucket', type=str, default='neurana-imaging', help='Bucket')
    parser.add_argument('--output_prefix', type=str, default='roi_extract', help='Output prefix')
  
    return parser.parse_args()

import boto3
import base64
from botocore.exceptions import ClientError
import json
import os

def get_aws_hcp_keys():
    # First try environment variables
    print(os.getenv('HCP_KEYS'))
        
    secret_name = "aws/hcp"
    region_name = "eu-west-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        print(e.response['Error']['Code'])
        
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        
        secret = json.loads(get_secret_value_response['SecretString'])
        
    return secret


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3') # Don't use special profile
    response = s3_client.upload_file(file_name, bucket, object_name)
    print(f'Uploading to bucket {bucket} object_name {object_name}')
    return response

def main(args):
    sub=args.subject
    print(f'Working on subject {sub}')

    hcp_keys = get_aws_hcp_keys()
    session = boto3.Session(aws_access_key_id=hcp_keys['AWS_ACCESS_KEY_ID'], aws_secret_access_key=hcp_keys['AWS_SECRET_ACCESS_KEY'])

    s3 = session.client('s3')

    # List of tasks and contrasts to be analysed for each
    taskcondict = {
        'tfMRI_WM': [8, 9, 14, 15, 16, 17], # 2BK, 0BK, BODY, FACE, PLACE, TOOL
        'tfMRI_GAMBLING': [0, 1],     # PUNISH, REWARD
        'tfMRI_MOTOR': [0, 1, 2, 3, 4, 5],     # CUE, LF, LH, RF, RH, T
        'tfMRI_LANGUAGE': [0, 1],     # MATH, STORY 
        'tfMRI_SOCIAL': [0, 1],     # RANDOM, TOM
        'RELATIONAL':[0, 1, 2, 3, 4, 5],   # MATCH, REL, MATCH-REL, REL-MATCH, neg_MATCH, neg_REL
        'tfMRI_EMOTION': [0, 1],     # FACES, SHAPES
        }   

    # ROIS (DLPFC left and right
    # We should change this to all regions
    roilist=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 73, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360]
    # Load up ROI files for L and R
    roi_L_img=nib.load('rois/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')
    roi_R_img=nib.load('rois/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')
    roi_L_dat=roi_L_img.get_fdata().ravel().astype(int)
    roi_R_dat=180 + roi_R_img.get_fdata().ravel().astype(int)
    roi_dat=np.concatenate((roi_L_dat,roi_R_dat))

    # Handy later 
    nroi = len(roilist)
    ntask = len(taskcondict)

    # Initialise a space for the output summary values
    meanact={}
    for task, taskcon in taskcondict.items():
        meanact[task]=np.zeros((len(taskcon), nroi))
        

    # Main loop over contrast files
    for task, taskcon in taskcondict.items():
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
        except:
            print("Unexpected error: %s" % e)
            raise

        task_img = nib.load('/tmp/timeseries.nii')

        # Pick out only voxels on the cortical surface
        task_dat=task_img.get_fdata()
        task_surfmask=task_img.header.get_axis(1).surface_mask
        task_dat_surf=task_dat[:,task_surfmask]

        # For each ROI, summarise activity in the selected regions and contrasts for this task
        for roiind, roi in enumerate(roilist):
            sel=task_dat_surf[:, roi_dat == roi]
            meanact[task][:, roiind] = np.mean(sel, 1)[taskcon]

    # Write output file 
    outfn=f'sub-{sub}_roi.msgpack-numpy'
    outpth='/tmp'
    x_enc = msgpack.packb(meanact, default=m.encode)
    with open(path.join(outpth,outfn),'wb') as f:
        msgpack.dump(meanact, f, default=m.encode)

    # Upload to s3
    print(upload_file(path.join(outpth,outfn), args.output_bucket, path.join(args.output_prefix, outfn)))
        
if __name__=='__main__':
    args = parse_args()
    main(args)

