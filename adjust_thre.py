import numpy as np
import pickle
from numpy.lib.format import open_memmap
from tqdm import tqdm
from large2small import Feeder,resmall,gen_smalldata,stat_multiscale
import os


benchmark = ['xview','xsub']
part = ['train','val']
ntupath = '/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len'
outdata = '/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-segment'
scorepa = '/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/score'

# a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
# thres = np.zeros((a.shape[0], 6))  # 10,6
#
# for b in benchmark:
#     for p in part:
#         datapath = ntupath + '/' + str(b) + '/' + str(p) + '_data.npy'
#         labelpath = ntupath + '/' + str(b) + '/' + str(p) + '_label.pkl'
#         scorepath = scorepa + '/' + str(b) + '/' + str(p) + '_score.npy'
#         outpath = outdata + '/' + str(b)
#         score=np.array(np.load(scorepath))  #N,M,V
#         #a = np.array([i * 10 for i in range(1, 10)])
#         for i, t in tqdm(enumerate(a)):
#             for j in range(0, score.shape[2]):
#                 thres[i, j] = thres[i,j]+np.percentile(score[:, 0, j], t, interpolation='midpoint')

def gen_thre(scorepath,label_path,quant):  #quant:分位数
    score=np.array(np.load(scorepath))   #N,M,V=6
    with open(label_path, 'rb') as f:
        _,label,_ = pickle.load(f)
    scores=[]
    douper = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    for i in tqdm(range(0, len(label))):
        if label[i] in douper:
            scores.append(score[i,0,:])
            scores.append(score[i, 1, :])
        else:
            scores.append(score[i, 0, :])
    scores=np.array(scores)
    thre=np.zeros(score.shape[2]) #V=6
    for i in range(0,score.shape[2]):
        thre[i]=np.percentile(scores[:,i], quant, interpolation='midpoint')
    return thre   # V=6

quant=np.array([98.7,92.1,78.9,65.8,52.6,39.5,26.3,13.2])  #25%,30%,40%,50%,60%,70%,80%,90%

##计算xsub,xview里train的阈值
thress=[]
for b in benchmark:
    scorepath = scorepa + '/' + str(b) + '/' + 'train' + '_score.npy'
    labelpath = ntupath + '/' + str(b) + '/' + 'train' + '_label.pkl'
    #quant=np.array([0.947,0.737,0.526,0.316])    #7,11,15,19
    thres=[]
    for q in quant:
        thre=gen_thre(scorepath,labelpath,q)
        thres.append(thre)
    thress.append(thres)
thress=np.array(thress)   #2,10,V=6

#print(thress)


# with open('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-segment/illustration0.pkl', 'rb') as f:
#         thres,_ = pickle.load(f)

# thres=np.array(thres)
# comrate=np.zeros(thres.shape[0])  #10
#
# for b in benchmark:
#     for p in part:
#         datapath = ntupath + '/' + str(b) + '/' + str(p) + '_data.npy'
#         labelpath = ntupath + '/' + str(b) + '/' + str(p) + '_label.pkl'
#         scorepath = scorepa + '/' + str(b) + '/' + str(p) + '_score.npy'
#         outpath = outdata + '/' + str(b)
#         for i,thre in tqdm(enumerate(thres)):
#             sums, total, labels = stat_multiscale(datapath, scorepath, labelpath, thre)
#             comrate[i]=comrate[i]+np.sum(sums)/(total*25)  #计算压缩率
#             partno=p+'_'+'seg'+str(a[i])
#             #gen_smalldata(datapath, scorepath, labelpath, outpath, thre, partno)
#
# comrate=comrate/4
#
# with open('{}/illustration.pkl'.format(outdata), 'wb') as f:
#     pickle.dump((list(thres),list(comrate)), f)


# comrate=np.zeros(thres.shape[0])
# for i,thre in enumerate(thres):
#     sums, total, labels = stat_multiscale(datapath, scorepath, labelpath, thre)
#     comrate[i]=np.sum(sums)/(total*25)





#print(comrate)

comrate=np.zeros((len(benchmark),len(part),len(quant)))   #2,2,10
kejot=np.zeros((len(benchmark),len(part),len(quant)))   #2,2,10

#illpath='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-segment'

for i,b in enumerate(benchmark):
    for j,p in enumerate(part):
        # ill_path = illpath + '/' + str(b) + '_' + str(p)+'_illustration.pkl'
        # with open(ill_path,'rb') as f:
        #     thress, tocom = pickle.load(f)  # 10,6   10
        # thress, tocom = np.array(thress), np.array(tocom)

        datapath = ntupath + '/' + str(b) + '/' + str(p) + '_data.npy'
        labelpath = ntupath + '/' + str(b) + '/' + str(p) + '_label.pkl'
        scorepath = scorepa + '/' + str(b) + '/' + str(p) + '_score.npy'
        outpath = outdata + '/' + str(b)
        for k,thre in tqdm(enumerate(thress[i])):
            sums,joints, total, labels = stat_multiscale(datapath, scorepath, labelpath, thre)
            comrate[i,j,k]=np.sum(sums)/(total*25)  #计算压缩率
            partno=p+'_'+'seg'+str(quant[k])
            kejot[i,j,k]=np.sum(joints)/total      #2,2,10,joints num to keep
            gen_smalldata(datapath, scorepath,labelpath, outpath, thre,partno)

        #with open('{}/{}_{}_illustration_4.0_.pkl'.format(outdata,str(b),str(p)), 'wb') as f:
         #   pickle.dump((list(thress[i,:,:]),list(comrate[i,j,:]),list(kejot[i,j,:])), f)
#print(kejot)



