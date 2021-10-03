import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

indexDF = ['tfMRI_WM_2BK', 'tfMRI_WM_0BK', 'tfMRI_WM_BODY', 'tfMRI_WM_FACE', 'tfMRI_WM_PLACE', 'tfMRI_WM_TOOL', 'tfMRI_GAMBLING_PUNISH', 'tfMRI_GAMBLING_REWARD', 'tfMRI_MOTOR_CUE', 'tfMRI_MOTOR_LF', 'tfMRI_MOTOR_LH', 'tfMRI_MOTOR_RF', 'tfMRI_MOTOR__RH', 'tfMRI_MOTOR_T', 'tfMRI_LANGUAGE_MATH', 'tfMRI_LANGUAGE_STORY', 'tfMRI_SOCIAL_RANDOM', 'tfMRI_SOCIAL_TOM', 'tfMRI_EMOTION_FACES', 'tfMRI_EMOTION_SHAPES']
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
ax.set_xticks(np.arange(0, len(indexDF), 1))
ax.set_yticks(np.arange(0, len(indexDF), 1))

# Labels for major ticks
ax.set_xticklabels(indexDF)
ax.set_yticklabels(indexDF)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
         
plt.imshow(rdm)
plt.savefig(('rdm.png'), dpi=200)

'''''
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
'''''