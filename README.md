# DMDCT
面向语义通信的3D骨骼点数据编码与压缩方法
Encoding and compression method of 3D skeleton data for semantic communication
[论文地址](https://journal.bupt.edu.cn/CN/abstract/abstract5092.shtml)
- 论文所用数据集——[NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- 论文所需数据预处理——[数据](https://pan.baidu.com/s/1irUhGiF0soSH9H8v-pAH7Q) 提取码：4ofa 
--来自百度网盘超级会员V6的分享
- 论文所对比的骨骼点压缩模型——[3D Human Skeleton Data Compression for Action Recognition](https://ieeexplore.ieee.org/abstract/document/8965920)
- 论文对比实验所用的网络——[HCN-pytorch](https://github.com/huguyuehuhu/HCN-pytorch)

## large2small.py:

get_map(): 对关节点进行分组，将小尺度分为6组大尺度，返回两个尺度的变换矩阵

AveargePart():一个类，其forward函数根据尺度间变换矩阵，将ground truth数据映射为关节点通道为6的大尺度数据

gen_smalldata(): 在确定的阈值下，根据样本得分(score)将上面生成的大尺度数据转换为小尺度(即压缩后的小尺度数据)，返回生成的小尺度数据(关节点通道维度6)

gen_dctdata(): 将上述小尺度数据在时间上使用DCT进一步压缩，可以生成某一压缩率下DCT/IDCT后的小尺度数据


## ntu-largescale-data.py:

该py文件计算每个ground truth数据样本的得分(score)，并生成得分文件


## adjust_thre.py：

gen_thre(): 根据样本得分(score)和设置的分位数生成得分阈值，得分大于该阈值的样本将使用ground truth，小于该阈值的样本将使用压缩后的小尺度


## dwt.py:

gen_dwtdata(): 将小尺度数据使用DWT进一步压缩，可以生成某一压缩率下DWT/IDWT后的小尺度数据
