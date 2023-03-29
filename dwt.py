import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

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
    J_id = np.matmul(J_d, fix_bases)
    return J_id

# dwt = DWT1DForward(wave='db6', J=3)
# X = torch.randn(3, 25, 300)
# #X=torch.tensor([[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]]])
# yl, yh = dwt(X)
# print(yl.shape)
# #torch.Size([10, 5, 22])
# print(yh[0].shape)
# #torch.Size([10, 5, 55])
# print(yh[1].shape)
# #torch.Size([10, 5, 33])
# print(yh[2].shape)
# #torch.Size([10, 5, 22])
# idwt = DWT1DInverse(wave='db6')
# yl[0][0][int(yl.shape[-1]*0.1):]=torch.tensor(0)
# yh[0][0][0][int(yh[0].shape[-1]*0.1):]=torch.tensor(0)
# yh[1][0][0][int(yh[1].shape[-1]*0.1):]=torch.tensor(0)
# yh[2][0][0][int(yh[2].shape[-1]*0.1):]=torch.tensor(0)
# x = idwt((yl, yh))
# print(torch.sum(torch.abs(X-x)))

# Xt=X.numpy()
# #xt=np.zeros(Xt.shape)
# xt=in_dct(Xt,int(Xt.shape[-1]*0.9))
# Xt=torch.from_numpy(Xt)
# xt=torch.from_numpy(xt)
# print(torch.sum(torch.abs(Xt-xt)))
#
# print(torch.sum(torch.abs(Xt-X)))

def traj_dwt(J_d, cr, leng):
    dwt = DWT1DForward(wave='db6', J=3)
    idwt = DWT1DInverse(wave='db6')
    yl, yh = dwt(J_d)
    rate=yl.shape[-1]+yh[0].shape[-1]+yh[1].shape[-1]+yh[2].shape[-1]
    yl[0][0][int(yl.shape[-1]*(leng*cr)/rate):] = torch.tensor(0)
    yh[0][0][0][int(yh[0].shape[-1]*(leng*cr)/rate):] = torch.tensor(0)
    yh[1][0][0][int(yh[1].shape[-1]*(leng*cr)/rate):] = torch.tensor(0)
    yh[2][0][0][int(yh[2].shape[-1]*(leng*cr)/rate):] = torch.tensor(0)
    J_id = idwt((yl, yh))
    return J_id


def gen_dwtdata(data_path,num_path,out_path,p,cr):
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

    data=torch.from_numpy(data)
    for i in tqdm(range(0, len(numframe))):
        leng = numframe[i]
        J_d0=data[i,0,:,:,:] #M,C,V,T
        J_d1 = data[i, 1, :, :, :]  # M,C,V,T
        coeff0 = traj_dwt(J_d0, cr, leng)  #M,C,V,T=cr*leng
        coeff1 = traj_dwt(J_d1, cr, leng)  # M,C,V,T=cr*leng
        coeff0,coeff1=coeff0.numpy(),coeff1.numpy()
        l = coeff0.shape[-1]
        coeff0,coeff1 = coeff0.transpose(0, 2, 1),coeff1.transpose(0,2,1)    # M,C,T=cr*leng,V
        fp0[i, 0, :,0:l, :] = coeff0  # M,C,T,V
        fp0[i, 1, :, 0:l, :] = coeff1  # M,C,T,V
    fp[:, :, :, :, :] = fp0.transpose(0, 2, 3, 4, 1)[:, :, :, :, :]  # N,C,T,V,M

if __name__ == '__main__':
    benchmark=['xview','xsub']
    part=['val']
    cr=[0.9]
    ntupath='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-segment/xview'
    #outdata='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-small'
    #outdata='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/idct'
    outdata='/mnt/DataDrive5/yangjiahui/ntu-compress/xview/idwt'
    score='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/score'
    thre=np.array([1.7,1.7,1,1,2,2])

    for p in part:
        for comrate in cr:
            datapath=ntupath+'/'+p+'_seg52.6_data.npy'
            numpath='/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/'+p+'_num_frame.npy'
            gen_dwtdata(datapath, numpath, outdata, p, comrate)