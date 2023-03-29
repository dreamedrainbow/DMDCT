import numpy as np
from numpy.lib.format import open_memmap
import os
import pickle

def DCT_Base(n_frames,n_bases):
    x = np.arange(n_frames)
    fixed_bases = [np.ones(n_frames) * np.sqrt(1 / n_frames)]
    for i in range(1, n_bases):
        fixed_bases.append(np.sqrt(2 / n_frames) * np.cos(i * np.pi * ((x + 0.5) / n_frames)))
    fixed_bases = np.array(fixed_bases,dtype=np.float64)  #10,50
    return fixed_bases  #n_bases,n_frames

def idct(data,nframe,nbases): #data:C,V,M,T
    fixed_bases=DCT_Base(nframe,nbases)  #10,100
    J_pos=np.matmul(data,fixed_bases)  #C,V,M,100
    J_pos=J_pos.transpose(0,3,1,2)  #C,T,V,M
    return J_pos

# with open('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/enc_cha_dec_idct_data/val_52.6_10%_1800_label.pkl', 'rb') as f:
#     samplename00,label00=pickle.load(f)
#
# data0=np.load('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/enc_cha_dec_idct_data/val_52.6_20%_len_50_snr10dB_data.npy',mmap_mode='r')
# data0=np.array(data0)
# data0 = np.load('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/dct/val_seg52.6_10%_data.npy', mmap_mode='r')
# data0=np.array(data0)
data = np.load('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/idct/val_seg52.6_10%_data.npy', mmap_mode='r')
data=np.array(data)
N,C,T,V,M=data.shape
# with open('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/val_label0.pkl', 'rb') as f:
#     samplename0,label0=pickle.load(f)
data=data.transpose(0,4,1,2,3)  #N,M,C,T,V
leng=np.load('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-len/xview/val_num_frame.npy', mmap_mode='r')
leng=np.array(leng)
#b=data[1012,:,0:int(leng[1012]*0.1),:,:]
out_path='/mnt/DataDrive5/zhouhonghong'
fp = open_memmap('{}/{}_data.npy'.format(out_path, 'seg52.6_S1C2P1R1A12'),
                dtype='float32',
                mode='w+',
                shape=(C,T,V))  # N,M,V=6
fp[:,:,:]=data[11,0,:,:,:]
a=np.random.randint(low=0,high=18931,size=1893)
a=np.random.choice(18931, 1800, replace=False)
for s in a:
    data_len=data[s,:,0:int(leng[s]*0.2),:,:]
    data_len[:,:,:,:].tofile('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/dct_bin/val_52.6_20%_len_1800/'+str(s)+'.bin')  #生成bin文件
    print(s)


a=np.fromfile('/mnt/DataDrive5/zhouhonghong/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/dct_bin/3.bin',dtype='float32')  #bin转成numpy
a=a.reshape(C,int(leng[3]*0.1),V,M)
a=a.reshape(C,T,V,M)
print(data)


# snrfile='/mnt/DataDrive5/yangjiahui/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/rayleigh_ldpc_0.5_bit_snr_0dB_10%/val_52.6_10%_len_1800'
# out_path='/mnt/DataDrive5/yangjiahui/parse_ntu_skeleton/NTU-RGB-D-dct&idct/xview/enc_cha_dec_idct_data'
# filename=os.listdir(snrfile)
# filename.sort()
# label=[]
# numframe=[]
# samplename=[]
# fp0 = np.zeros((len(filename), C, T, V, M))
# fp = open_memmap('{}/{}_data.npy'.format(out_path, 'val_rayleigh_52.6_10%_len_1800_snr0dB'),
#                  dtype='float32',
#                  mode='w+',
#                  shape=fp0.shape)  # 1800,C,T=300,V,M
# for i,name in enumerate(filename):
#     index = int(name[:-4])
#     dedata = np.fromfile(snrfile+'/'+name,dtype='float32')
#     dedata = dedata.reshape(C, int(leng[index] * 0.1), V, M)  #C,T‘,V,M
#     dedata=dedata.transpose(0,2,3,1)  #C,V,M,T'
#     idcdata=idct(dedata,T,int(leng[index]*0.1))  #C,T=300,V,M
#     fp0[i,:,:,:,:]=idcdata[:,:,:,:]
#     label.append(label0[index])
#     numframe.append(leng[index])
#     samplename.append(samplename0[index])
# fp[:,:,:,:,:]=fp0[:,:,:,:,:]
# numframe=np.array(numframe,dtype='int')

# with open('{}/{}_label.pkl'.format(out_path,'val_52.6_20%_1800'), 'wb') as f:
#     pickle.dump((list(samplename),list(label)), f)
#
# np.save('{}/{}_num_frame.npy'.format(out_path,'val_52.6_20%_1800'),numframe)
