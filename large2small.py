import numpy as np
import pickle
from numpy.lib.format import open_memmap
from tqdm import tqdm
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
        # x_body = np.stack((x_leftleg, x_rightleg, x_head,x_torso, x_leftarm, x_rightarm),
        #                   axis=-1)  # [N, C, T, V=1]), dim=-1)        # [N, C, T, 6]
        x_body = np.stack((x_head, x_leftarm, x_rightarm, x_torso, x_leftleg, x_rightleg),
                          axis=-1)  # [N, C, T, V=1]), dim=-1)        # [N, C, T, 6]
        return x_body

    def avg_pool(self,x):
        x = np.mean(x, axis=-1)
        return x

Joint_num=25
Part_num=6
def get_map():
    torso = [0, 1, 20]
    right_leg = [16, 17, 18, 19]
    left_leg = [12, 13, 14, 15]
    right_arm = [8, 9, 10, 11, 24, 23]
    left_arm = [4, 5, 6, 7, 22, 21]
    head = [2, 3]
    body=[left_leg, right_leg, head, torso, left_arm, right_arm]
    Mat=np.zeros((Joint_num,Part_num))   #25,6
    for i,s in enumerate(body):
        Mat[s,i]=1
    return Mat

def resmall(thre,data,score):   #data:N,C,T,V=25  score:N,V=6  thre:V=6 (对每个关节部分的阈值)
    S2J = AveargePart()
    s2x = S2J.forward(data)  #N,C,T,V=6
    Mat=get_map()  #25,6
    prodata=np.matmul(s2x,Mat.T)  #N,C,T,V=25  同一部分内关节点值相等
    score_bool=(score>thre)  #N,V=6
    Mat_bool=np.matmul(score_bool,Mat.T)  #N,V=25 恢复小尺度为1，else 0
    result=np.zeros(data.shape)  #N,C,T,V=25
    result=result.transpose(1,2,0,3) #C,T,N,V
    data=data.transpose(1,2,0,3)  #C,T,N,V=25
    prodata=prodata.transpose(1,2,0,3)  #C,T,N,V=25
    result[:,:,Mat_bool>0]=data[:,:,Mat_bool>0]  #小尺度 C,T,N,V
    result[:,:,Mat_bool<1]=prodata[:,:,Mat_bool<1]  #大尺度 C,T,N,V
    return result.transpose(2,0,1,3),Mat_bool  #N,C,T,V

def traj_dct(data,cr,length):  #data: C,V,T data的最后一维为时间，length为实际帧长度
    fix_bases = DCT_Base(data.shape[-1], int(length*cr))  # 10,50
    J_d = np.matmul(data, fix_bases.T)  # C,V,T=10(不同动作不一样)
    J_id=np.matmul(J_d,fix_bases)   #C,V,T=50
    return J_id

def DCT_Base(n_frames,n_bases):
    x = np.arange(n_frames)
    fixed_bases = [np.ones(n_frames) * np.sqrt(1 / n_frames)]
    for i in range(1, n_bases):
        fixed_bases.append(np.sqrt(2 / n_frames) * np.cos(i * np.pi * ((x + 0.5) / n_frames)))
    fixed_bases = np.array(fixed_bases,dtype=np.float64)  #10,50
    return fixed_bases  #n_bases,n_frames



#data_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/val_data.npy'
#label_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/val_label.pkl'
#score_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/score/xview/val_score.npy'
#dataset=Feeder(data_path,label_path,score_path)
#length=np.array(dataset.numframe)
#index=[0,20,30,40,50,60,70,80]
#data=np.array([dataset[i][0] for i in tqdm(range(0,len(dataset)))])  #N,C,T,V,M
#label=[dataset[i][1] for i in tqdm(range(0,len(dataset)))]
#score=np.array([dataset[i][2] for i in tqdm(range(0,len(dataset)))])  #N,M,V

#N,C,T,V,M=data.shape
#data=data.transpose(0,4,1,2,3) #N,M,C,T,V
#data=data.reshape(N*M,C,T,V)  #N*M,C,T,V=25
#score=score.reshape(N*M,6)    #N*M,C,T,V=6

#thre=np.array([2.1,2.1,1,1,2.1,2.1])

#result=resmall(thre,data,score)
#print(result)

#out_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-compressrate/xview'
#part_no='val_50%'


def gen_traj(thre,data,score,length,cr,out_path,part_no):  #data:N,C,T,V=25  score:N,V=6 thre:V=6, length:N/2  cr:压缩率
    result,_=resmall(thre,data,score)  #N,C,T,V=25
    result=result.transpose(0,1,3,2) #N,C,V,T=300
    N,C,V,T=result.shape
    fp0 = np.zeros((N,C, T, V))
    fp = open_memmap('{}/{}_data.npy'.format(out_path, part_no),
                     dtype='float32',
                     mode='w+',
                     shape=(int(N/2), C, T, V,2))  # N,M,V=6
    for i in tqdm(range(0,len(result))):
        leng=length[int(i/2)]
        J_data=result[i,:,:,0:leng]
        coeff=traj_dct(J_data, cr, leng)  #C,V,T=cr*leng
        l = coeff.shape[-1]
        coeff=coeff.transpose(0,2,1)  #C,T,V
        fp0[i,:,0:l,:]=coeff  #N,C,T,V
        #fp[i,:,0:l,:]=coeff  #N,C,T,V
    #fp=fp.tanspose(0,1,3,2)  #N,C,T=300,V
    fp0=fp0.reshape(int(N/2),2,C,T,V)
    fp[:,:,:,:,:]=fp0.transpose(0,2,3,4,1)[:,:,:,:,:] #N,C,T,V,M
    return

#gen_traj(thre,data,score,length,cr=0.5,out_path=out_path,part_no=part_no)


def gen_trajdata(data_path,score_path,label_path,outpath,thre,part,cr):
    dataset = Feeder(data_path, label_path, score_path)
    length = np.array(dataset.numframe)
    data = np.array([dataset[i][0] for i in tqdm(range(0, len(dataset)))])  # N,C,T,V,M
    score = np.array([dataset[i][2] for i in tqdm(range(0, len(dataset)))])  # N,M,V
    N, C, T, V, M = data.shape
    data = data.transpose(0, 4, 1, 2, 3)  # N,M,C,T,V
    data = data.reshape(N * M, C, T, V)  # N*M,C,T,V=25
    score = score.reshape(N * M, score.shape[-1])  # N*M,V=6
    #thre = np.array([2.1, 2.1, 1, 1, 2.1, 2.1])
    part_no = part+'_'+str(int(cr*100))+'%'
    gen_traj(thre, data, score, length, cr,outpath,part_no)


def gen_smalldata(data_path,score_path,label_path,outpath,thre,part):
    dataset = Feeder(data_path, label_path, score_path)
    length = np.array(dataset.numframe)
    data = np.array([dataset[i][0] for i in tqdm(range(0, len(dataset)))])  # N,C,T,V,M
    score = np.array([dataset[i][2] for i in tqdm(range(0, len(dataset)))])  # N,M,V
    N, C, T, V, M = data.shape
    data = data.transpose(0, 4, 1, 2, 3)  # N,M,C,T,V
    data = data.reshape(N * M, C, T, V)  # N*M,C,T,V=25
    score = score.reshape(N * M, score.shape[-1])  # N*M,V=6
    result,_=resmall(thre,data,score)  #N*M,C,T,V=25
    result=result.reshape(N,M,C,T,V)
    result=result.transpose(0,2,3,4,1) #N,C,T,V,M
    fp = open_memmap('{}/{}_data.npy'.format(outpath, part),
                   dtype='float32',
                    mode='w+',
                    shape=(N, C, T, V, M))  # N,C,T,V,M=2
    fp[:,:,:,:]=result[:,:,:,:]

def stat_multiscale(data_path,score_path,label_path,thre):
    dataset = Feeder(data_path, label_path, score_path)
    label=dataset.label #N
    data = np.array([dataset[i][0] for i in tqdm(range(0, len(dataset)))])  # N,C,T,V,M
    score = np.array([dataset[i][2] for i in tqdm(range(0, len(dataset)))])  # N,M,V
    judge = np.array(score < thre)  # N,M,V=6
    N, C, T, V, M = data.shape
    data = data.transpose(0, 4, 1, 2, 3)  # N,M,C,T,V
    data = data.reshape(N * M, C, T, V)  # N*M,C,T,V=25
    score = score.reshape(N * M, score.shape[-1])  # N*M,V=6
    _,matbool = resmall(thre, data, score)  # N*M,C,T,V=25, N*M,V=25
    matbool=matbool.reshape(N,M,V)  #N,M,V=25
    total=0
    sum=[]  #每个样本的小尺度关节点个数+大尺度部分个数
    joints=[]  #每个样本的小尺度关节个数
    labels=[]
    douper=[49,50,51,52,53,54,55,56,57,58,59]
    for i in tqdm(range(0,len(label))):
        if label[i] in douper:
            total=total+2
            s=np.sum(matbool[i,0,:])+np.sum(judge[i,0,:])
            sum.append(s)
            joints.append(np.sum(matbool[i,0,:]))
            labels.append(label[i])

            s=np.sum(matbool[i, 1, :])+np.sum(judge[i,1,:])
            sum.append(s)
            joints.append(np.sum(matbool[i,1,:]))
            labels.append(label[i])

        else:
            total=total+1
            s = np.sum(matbool[i, 0, :])+np.sum(judge[i,0,:])
            sum.append(s)
            joints.append(np.sum(matbool[i,0,:]))
            labels.append(label[i])
    return np.array(sum),np.array(joints),total,labels   #sum:每个样本小尺度的关节点个数+大尺度部分个数 total总共含有一个人的样本 labels样本标签



def gen_dctdata(data_path,num_path,out_path,p,cr):
    data=np.load(data_path, mmap_mode='r')
    data=np.array(data) #N,C,T,V,M
    numframe=np.load(num_path)
    numframe=np.array(numframe)  #N
    N,C,T,V,M=data.shape
    data=data.transpose(0,4,1,2,3)  #N,M,C,T,V
    data=data.transpose(0,1,2,4,3)  #N,M,C,V,T
    fp0 = np.zeros((N,M, C, T, V))
    part_no=p+'_'+'seg52.6_'+str(int(cr*100))+'%'
    fp = open_memmap('{}/{}_data.npy'.format(out_path, part_no),
                     dtype='float32',
                     mode='w+',
                     shape=(N, C, T, V, M))  # N,M,V=6

    for i in tqdm(range(0, len(numframe))):
        leng = numframe[i]
        J_d=data[i,:,:,:,:] #M,C,V,T
        coeff = traj_dct(J_d, cr, leng)  #M,C,V,T=cr*leng
        l = coeff.shape[-1]
        coeff = coeff.transpose(0, 1, 3,2)  # M,C,T=cr*leng,V
        fp0[i, :, :,0:l, :] = coeff  # M,C,T,V

    fp[:, :, :, :, :] = fp0.transpose(0, 2, 3, 4, 1)[:, :, :, :, :]  # N,C,T,V,M



if __name__ == '__main__':
    benchmark=['xview','xsub']
    part=['train','val']
    cr=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    ntupath='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-segment/xview'
    #outdata='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-small'
    outdata='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/idct'
    score='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/score'
    thre=np.array([1.7,1.7,1,1,2,2])

    for p in part:
        for comrate in cr:
            datapath=ntupath+'/'+p+'_seg52.6_data.npy'
            numpath='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/'+p+'_num_frame.npy'
            gen_dctdata(datapath, numpath, outdata, p, comrate)





    # for comrate in cr:
    #     for b in benchmark:
    #         for p in part:
    #             datapath=ntupath+'/'+b+'/'+p+'_data.npy'
    #             labelpath=ntupath+'/'+b+'/'+p+'_label.pkl'
    #             scorepath=score+'/'+b+'/'+p+'_score.npy'
    #             outpath=outdata+'/'+b
    #             gen_trajdata(datapath, scorepath, labelpath, outpath, thre, p, comrate)
                #sums,total,labels=stat_multiscale(datapath, scorepath, labelpath, thre)
                #out_path='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/stat-joints/xview'
                #with open('{}/{}_numjoints.pkl'.format(out_path, part), 'wb') as f:
                #   pickle.dump((list(sums),list(labels)), f)
                #print(sums)
                #tosum=np.sum(sums)
                #print(total)
                #print(tosum/(total*25))






