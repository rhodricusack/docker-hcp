import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch

indexDF = ['WORKING_MEM_2BK_BODY', 'WORKING_MEM_2BK_FACE', 'WORKING_MEM_2BK_PLACE', 'WORKING_MEM_2BK_TOOL', 'WORKING_MEM_0BK_BODY', 'WORKING_MEM_0BK_FACE', 'WORKING_MEM_0BK_PLACE', 'WORKING_MEM_0BK_TOOL', 'WORKING_MEM_2BK', 'WORKING_MEM_0BK', 'WORKING_MEM_2BK-0BK', 'WORKING_MEM_neg_2BK', 'WORKING_MEM_neg_0BK', 'WORKING_MEM_0BK-2BK', 'WORKING_MEM_BODY', 'WORKING_MEM_FACE', 'WORKING_MEM_PLACE', 'WORKING_MEM_TOOL', 'WORKING_MEM_BODY-AVG', 'WORKING_MEM_FACE-AVG', 'WORKING_MEM_PLACE-AVG', 'WORKING_MEM_TOOL-AVG', 'WORKING_MEM_neg_BODY', 'WORKING_MEM_neg_FACE', 'neg_PLACE', 'WORKING_MEM_neg_TOOL', 'WORKING_MEM_AVG-BODY', 'WORKING_MEM_AVG-FACE', 'WORKING_MEM_AVG-PLACE', 'WORKING_MEM_VG-TOOL', 'GAMBLING_PUNISH', 'GAMBLING_REWARD', 'GAMBLING_PUNISH-REWARD', 'GAMBLING_neg_PUNISH', 'GAMBLING_neg_REWARD', 'GAMBLING_REWARD-PUNISH', 'MOTOR_CUE', 'MOTOR_LF', 'MOTOR_LH', 'MOTOR_RF', 'MOTOR_RH', 'MOTOR_T', 'MOTOR_AVG', 'MOTOR_CUE-AVG', 'MOTOR_LF-AVG', 'MOTOR_LH-AVG', 'MOTOR_RF-AVG', 'MOTOR_RH-AVG', 'MOTOR_T-AVG', 'MOTOR_neg_CUE', 'MOTOR_neg_LF', 'MOTOR_neg_LH', 'MOTOR_neg_RF', 'MOTOR_neg_RH', 'MOTOR_neg_T', 'MOTOR_neg_AVG', 'MOTOR_AVG-CUE', 'MOTOR_AVG-LF', 'MOTOR_AVG-LH', 'MOTOR_AVG-RF', 'MOTOR_AVG-RH', 'MOTOR_AVG-T', 'LANGUAGE_MATH', 'LANGUAGE_STORY', 'LANGUAGE_MATH-STORY', 'LANGUAGE_STORY-MATH', 'LANGUAGE_neg_MATH', 'LANGUAGE_neg_STORY', 'SOCIAL_RANDOM', 'SOCIAL_TOM', 'SOCIAL_RANDOM-TOM', 'SOCIAL_neg_RANDOM', 'SOCIAL_neg_TOM', 'SOCIAL_TOM-RANDOM', 'RELATIONAL_MATCH', 'RELATIONAL_REL', 'RELATIONAL_MATCH-REL', 'RELATIONAL_REL-MATCH', 'RELATIONAL_neg_MATCH', 'RELATIONAL_neg_REL', 'EMOTION_FACES', 'EMOTION_SHAPES', 'EMOTION_FACES-SHAPES', 'EMOTION_neg_FACES', 'EMOTION_neg_SHAPES', 'EMOTION_SHAPES-FACES']   
indexDFnoneg = ['WORKING_MEM_2BK_BODY', 'WORKING_MEM_2BK_FACE', 'WORKING_MEM_2BK_PLACE', 'WORKING_MEM_2BK_TOOL', 'WORKING_MEM_0BK_BODY', 'WORKING_MEM_0BK_FACE', 'WORKING_MEM_0BK_PLACE', 'WORKING_MEM_0BK_TOOL', 'WORKING_MEM_2BK', 'WORKING_MEM_0BK', 'WORKING_MEM_2BK-0BK', 'WORKING_MEM_0BK-2BK', 'WORKING_MEM_BODY', 'WORKING_MEM_FACE', 'WORKING_MEM_PLACE', 'WORKING_MEM_TOOL', 'WORKING_MEM_BODY-AVG', 'WORKING_MEM_FACE-AVG', 'WORKING_MEM_PLACE-AVG', 'WORKING_MEM_TOOL-AVG', 'WORKING_MEM_AVG-BODY', 'WORKING_MEM_AVG-FACE', 'WORKING_MEM_AVG-PLACE', 'WORKING_MEM_VG-TOOL', 'GAMBLING_PUNISH', 'GAMBLING_REWARD', 'GAMBLING_PUNISH-REWARD', 'GAMBLING_REWARD-PUNISH', 'MOTOR_CUE', 'MOTOR_LF', 'MOTOR_LH', 'MOTOR_RF', 'MOTOR_RH', 'MOTOR_T', 'MOTOR_AVG', 'MOTOR_CUE-AVG', 'MOTOR_LF-AVG', 'MOTOR_LH-AVG', 'MOTOR_RF-AVG', 'MOTOR_RH-AVG', 'MOTOR_T-AVG', 'MOTOR_AVG-CUE', 'MOTOR_AVG-LF', 'MOTOR_AVG-LH', 'MOTOR_AVG-RF', 'MOTOR_AVG-RH', 'MOTOR_AVG-T', 'LANGUAGE_MATH', 'LANGUAGE_STORY', 'LANGUAGE_MATH-STORY', 'LANGUAGE_STORY-MATH', 'SOCIAL_RANDOM', 'SOCIAL_TOM', 'SOCIAL_RANDOM-TOM', 'SOCIAL_TOM-RANDOM', 'RELATIONAL_MATCH', 'RELATIONAL_REL', 'RELATIONAL_MATCH-REL', 'RELATIONAL_REL-MATCH', 'EMOTION_FACES', 'EMOTION_SHAPES', 'EMOTION_FACES-SHAPES', 'EMOTION_SHAPES-FACES']   

DLPFroilist = [26, 67, 68, 70, 71, 73, 83, 84, 85, 86, 87, 96, 98, 206, 247, 148, 250, 251, 253, 264, 265, 266, 267, 276, 278]

data = np.load('DLPFCroi.npy')

data=np.mean(data,0)
print(data.shape)
rdm = np.corrcoef(data)
print(rdm)
# should give numcons * numcons matrix
#if not then do rdm = np.corrcoef(X.T)

# Show matrix plot
ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, len(indexDFnoneg), 1))
ax.set_yticks(np.arange(0, len(indexDFnoneg), 1))

# Labels for major ticks
ax.set_xticklabels(indexDFnoneg, fontsize=6)
ax.set_yticklabels(indexDFnoneg, fontsize=6)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
         
plt.imshow(rdm)
plt.savefig(('rdmnoneg.png'), bbox_inches='tight')

fig,ax = plt.subplots(figsize=(10,10))
dend = sch.dendrogram(sch.linkage(rdm, method='ward'), 
    ax=ax, 
    labels=indexDFnoneg, 
    orientation='right'
    )
plt.xlim(0,50)
y = [0,50]
plt.xticks(np.arange(min(y), max(y)+1, 1.0))
plt.show()
plt.savefig(('rdm-hie_noneg.png'), bbox_inches='tight')










for nroi in range(25):
    for ncon in range(20):

        m = np.mean(data[:,ncon,nroi])
        print(m)
        std = np.std(data[:,ncon,nroi])
        file = open(('DLPFC_ROI_mean_stdev.txt'),'a') 
        file.write("\n Mean for ROI %d, contrast %d is %f, st dev is  %f"%(nroi,ncon,m,std))
        file.close()


rdm = np.corrcoef(data)
print(rdm)
# should give numcons * numcons matrix
#if not then do rdm = np.corrcoef(X.T)

# Show matrix plot
plt.imshow(rdm)
ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, numcons, 1))
ax.set_yticks(np.arange(0, numcons, 1))

# Labels for major ticks
ax.set_xticklabels(connames)
ax.set_yticklabels(connames)


for ncon in range(19):
    plt.figure()
    ax=sns.stripplot(data = (data[:,ncon,:]), edgecolor = "white", size = 3, jitter = 1, zorder = 0)
    plt.xlabel('DLPFC ROI')
    #plt.title('26 DLPFCs ROIs during contrast %s'%(indexDF[ncon]))
    #plt.savefig(('ROIsperCON%d.png'%(ncon)), dpi=200)