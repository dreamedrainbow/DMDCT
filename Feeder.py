import numpy as np
from numpy.lib.format import open_memmap
import pickle
import tqdm
import os


class Feeder():
    def __init__(self,data_path,label_path,score_path,debug=False,mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.score_path=score_path
        self.load_data(mmap)
    def load_data(self, mmap):
        # data: N C V T M
        # score: N M V
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label,self.numframe = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            self.score=np.load(self.score_path,mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            self.score=np.load(self.score_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.score = self.score[0:100]
            self.sample_name = self.sample_name[0:100]
            self.numframe=self.numframe[0:100]
        self.N, self.C, self.T, self.V, self.M = self.data.shape
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        score_numpy=np.array(self.score[index])  #N,M,V
        return data_numpy, label,score_numpy

class AveargePart():
    def __init__(self):
        super().__init__()
        self.torso = [0,1,20]
        self.right_leg = [16,17,18,19]
        self.left_leg = [12,13,14,15]
        self.right_arm = [8,9,10,11,24,23]
        self.left_arm = [4,5,6,7,22,21]
        self.head=[2,3]

    def forward(self, x):
        x_torso = self.avg_pool(x[:, :, :, self.torso])  # [N, C, T, V=1]
        x_leftleg = self.avg_pool(x[:, :, :, self.left_leg])  # [N, C, T, V=1]
        x_rightleg = self.avg_pool(x[:, :, :, self.right_leg])  # [N, C, T, V=1]
        x_leftarm = self.avg_pool(x[:, :, :, self.left_arm])  # [N, C, T, V=1]
        x_rightarm = self.avg_pool(x[:, :, :, self.right_arm])  # [N, C, T, V=1]
        x_head=self.avg_pool(x[:,:,:,self.head])  # [N, C, T, V=1]
        x_body = np.stack((x_leftleg, x_rightleg, x_head,x_torso, x_leftarm, x_rightarm),
                          axis=-1)  # [N, C, T, V=1]), dim=-1)        # [N, C, T, 6]
        return x_body

    def avg_pool(self,x):
        x = np.mean(x, axis=-1)
        return x



benchmark=['xsub','xview']
part=['train','val']
out_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len'
# for b in benchmark:
#     for p in part:
#         outpath=os.path.join(out_path,b)
#         ntu_path = '/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len'
#         scorepath='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/score'
#         score_path=os.path.join(scorepath,b,p+'_score.npy')
#         data_path = os.path.join(ntu_path, b, p + '_data.npy')
#         label_path = os.path.join(ntu_path, b, p + '_label.pkl')
#
#         dataset = Feeder(data_path, label_path, score_path)
#         label = dataset.label
#         name = dataset.sample_name
#         numframe = np.array(dataset.numframe)
#         #with open('{}/{}_label0.pkl'.format(outpath, p), 'wb') as f:
#         #   pickle.dump((name, list(label)), f)
#         fp = open_memmap(
#             '{}/{}_num_frame.npy'.format(outpath, p),
#             dtype='int',
#             mode='w+',
#             shape=numframe.shape)
#         fp[:]=numframe[:]



def large_part_mean():
    N,C,T,V,M=datarray.shape
    data=np.zeros((N*M,C,T,V))
    data=(datarray.transpose(0,4,1,2,3)).reshape((N*M,C,T,V))
    S2J=AveargePart()
    s2x=S2J.forward(data)  #N*M,C,T,V=6
    meanscore=np.mean(score[:,0,:],axis=0)
    return meanscore
    #print(data)


data=np.load('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/train_num_frame.npy')
print(data)