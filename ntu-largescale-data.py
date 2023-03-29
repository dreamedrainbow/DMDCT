import numpy as np
import pickle
from numpy.lib.format import open_memmap
import tqdm
from time import sleep
import sys
###生成score文件###
toolbar_width = 30
def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

class Feeder():
    def __init__(self,data_path,label_path,debug=False,mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.load_data(mmap)
    def load_data(self, mmap):
        # data: N C V T M
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label,self.numframe = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        self.N, self.C, self.T, self.V, self.M = self.data.shape
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        return data_numpy, label

def avg_pool(x):
    x=np.mean(x,axis=-1)
    return x


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
        x_torso = avg_pool(x[:, :, :, self.torso])  # [N, C, T, V=1]
        x_leftleg = avg_pool(x[:, :, :, self.left_leg])  # [N, C, T, V=1]
        x_rightleg = avg_pool(x[:, :, :, self.right_leg])  # [N, C, T, V=1]
        x_leftarm = avg_pool(x[:, :, :, self.left_arm])  # [N, C, T, V=1]
        x_rightarm = avg_pool(x[:, :, :, self.right_arm])  # [N, C, T, V=1]
        x_head=avg_pool(x[:,:,:,self.head])  # [N, C, T, V=1]
        x_body = np.stack((x_leftleg, x_rightleg, x_head,x_torso, x_leftarm, x_rightarm),
                          axis=-1)  # [N, C, T, V=1]), dim=-1)        # [N, C, T, 6]
        return x_body

    def traj_dct(self,x,cr):
        x=self.forward(x)   #N,C,T=300,6
        N, C, T, V = x.shape
        x=x.transpose(0,1,3,2)  #N,C,V=6,T=300
        fp=np.zeros((N,C,V,T))
        xf=dct()                  #N,C,V=6,T=75
        return xf.transpose(0,1,3,2)  #N,C,T=75,V=6
    def interal_dct(self,x):  #组内DCT
        N,C,T,V=x.shape
        y=np.zeros((N,C,T,6))
        y[:, :, :, 0] = in_dct(x[:, :, :, self.left_leg],n_bases=2)[:,:,:,1]  # [N, C, T, V=1]  DCT保留前两个频率分量，但只取第二个(第一个等同于平均)
        y[:, :, :, 1] = in_dct(x[:, :, :, self.right_leg],n_bases=2)[:,:,:,1]  # [N, C, T, V=1]
        y[:, :, :, 2] = in_dct(x[:, :, :, self.head],n_bases=2)[:,:,:,1]  # [N, C, T, V=1]
        y[:, :, :, 3] = in_dct(x[:, :, :, self.torso],n_bases=2)[:,:,:,1] # [N, C, T, V=1]
        y[:, :, :, 4] = in_dct(x[:, :, :, self.left_arm],n_bases=2)[:,:,:,1]  # [N, C, T, V=1]
        y[:, :, :, 5] = in_dct(x[:, :, :, self.right_arm],n_bases=2)[:,:,:,1]  # [N, C, T, V=1]
        return y

def DCT_Base(n_frames,n_bases):
    x = np.arange(n_frames)
    fixed_bases = [np.ones(n_frames) * np.sqrt(1 / n_frames)]
    for i in range(1, n_bases):
        fixed_bases.append(np.sqrt(2 / n_frames) * np.cos(i * np.pi * ((x + 0.5) / n_frames)))
    fixed_bases = np.array(fixed_bases,dtype=np.float64)  #10,50
    return fixed_bases
def dct(K,J_pos,NBase):          #DCT和IDCT变换 J_pos: N,C,T,V
    fix_bases = DCT_Base(K, NBase)  # 10,50
    J_d = np.matmul(J_pos, fix_bases.T)  #N,C,V,T
    return J_d
def in_dct(x,n_bases):
    n_frames = x.shape[-1]
    fix_bases = DCT_Base(n_frames, n_bases)  # 10,50
    J_d = np.matmul(x, fix_bases.T)
    return J_d

data_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/val_data.npy'
label_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/val_label.pkl'

dataset=Feeder(data_path,label_path)
num_frame = np.array(dataset.numframe)
sample_name=dataset.sample_name
body=['leftleg', 'rightleg', 'head','torso', 'leftarm', 'rightarm']
max_body=2

#计算得分
out_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/score/xview'
subject_no='val'
fp = open_memmap('{}/{}_score.npy'.format(out_path, subject_no),
       dtype='float32',
       mode='w+',
       shape=(len(dataset),max_body, len(body)))  #N,M,V=6


for i, s in enumerate(sample_name):
    print_toolbar(i * 1.0 / len(num_frame),
                  '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                      i + 1, len(sample_name), 'xview', 'val'))
    data=np.array(dataset[i][0])
    numframe = num_frame[i]
    C, T, V, M = data.shape
    data = data.transpose(3,0,1,2)  # M,C,T,V
    S2J = AveargePart()
    s2x = S2J.forward(data)   #M,C,T=300,V=6
    for k in range(0,2):
        ditraj = np.zeros((s2x.shape[1], numframe-1, s2x.shape[3])) # C,numframe-1,V=6
        ditraj = s2x[k, :, 1:numframe, :] - s2x[k, :, 0:numframe-1, :]  # C,numframe-1,V=6
        ditraj = np.abs(ditraj)
        ditraj = np.sum(ditraj, axis=1)  # C,V=6
        ditraj = np.sum(ditraj, axis=0)  # V=6
        fp[i,k, :] = ditraj  #N,M,V=6
    sleep(0.01)
    end_toolbar()
sleep(0.1)




