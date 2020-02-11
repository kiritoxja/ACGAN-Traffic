# ACGAN-Traffic

整体思路：

数据处理  --  ACGAN进行分类 --  对比实验

CGAN进行扩充数据  同时判别器输出真假和类别  实现GAN扩充数据和分类结合在一起的全局优化

python3 + pytorch

## 1 数据处理

1. 下载IDS2017  入侵检测的原始pcap数据集

2. https://github.com/yungshenglu/USTC-TK2016  提供的工具   将pcap切分成会话的形式  组成会话矩阵

   如A类别  有一条 05AB····  这种16进制的会话txt数据   由于需要输入固定长度的数据   因此可以采用长的截断，短的补0（最后固定长度  如756   或者P**acketCGAN: Exploratory Study of Class Imbalance for Encrypted Traffic Classification Using CGAN IEEE T RANSACTIONS ON N ETWORK AND S ERVICE M ANAGEMENT**  第4页提到的1480     总之这可以设置为一个参数 后面实验取最佳也行 ，上篇论文里的参考文献 **C. X. Wang Pan, “Encrypted traffic identification method based on stacked automatic encoder,” 2018, pp. 1–8.** 中也提到了固定长度的问题   截断和补0的可以分开存储  因为最后不一定都会用上所有数据集  因此可以只用截断的也行)

3. 归一化  每个会话由16进制的字节组成（0-255） 因此可以归一化到[0,1] 加快模型收敛

4. 最后的目录结构

   /data/className（如BOT）/ truncate(padding)/这里可以采取如csv文件来存储  n* m 的流量文件  n 代表了该类别的会话数  m代表了该会话的长度

5. 最后要统计每个类别的样本数对少样本进行后续的扩充，大样本随机取样？（还没想好）平衡数据后进行学习

   https://link.springer.com/article/10.1007/s00779-019-01332-y#Abs1中的table1 描述了该数据集的数据分布 我们也要根据自己的处理结果统计分布 

   ​	

## 2 ACGAN进行分类

https://zhuanlan.zhihu.com/p/44177576

有介绍ACGAN

![img](https://pic3.zhimg.com/80/v2-2ad47451ac82a66f5edbf868e3afe69a_hd.jpg)

   总之就是![[公式]](https://www.zhihu.com/equation?tex=D) 不仅需要判断每个样本的真假，还需要完成一个分类任务即预测 ![[公式]](https://www.zhihu.com/equation?tex=C) ，通过增加一个辅助分类器实现。

其中判别器和生成器可以考虑简单的多层全连接层

在**PacketCGAN: Exploratory Study of Class Imbalance for Encrypted Traffic Classification Using CGAN IEEE T RANSACTIONS ON N ETWORK AND S ERVICE M ANAGEMENT**

的参考文献中也提到了5中不同的分类器 都可以作为实验 (也可以考虑别的)

 MLP/CNN/SAE

 **P. Wang, F. Ye, X. Chen, and Y. Qian, “Datanet: Deep learning based encrypted network traffic classification in sdn home gateway,” IEEE Access, vol. 6, pp. 55380–55391, 2018.**

1D-CNN/2D-CNN

**W. Wang, M. Zhu, J. Wang, X. Zeng, and Z. Yang, “End-to-end encrypted traffic classification with one-dimensional convolution neural networks,” 2017 IEEE International Conference on Intelligence and Security Informatics (ISI), pp. 43–48, 2017.**

 **“Malware traffic classification using convolutional neural network for representation learning,” in 2017 International Conference on Information Networking (ICOIN), Jan 2017, pp. 712–717.**

   

## 3 对比实验

   既然是结合GAN的全局优化的分类 那么我想可以有三种不同的对比实验

### 没有任何数据平衡的

直接用原始数据输入2中的分类器进行分类

### 采用其他数据扩充方式进行数据平衡 再分类

如https://dl.acm.org/doi/pdf/10.1145/3155133.3155175?download=true中提到的

SMOTE采样

**Ly Vu Thi, Dong Van Tra, and Quang Uy Nguyen. 2016. Learning from Imbalanced Data for Encrypted Traffic Identification Problem. SoICT, Ho Chi Minh, Vietnam.**

 ensemble BalanceCascade technique

**A. More. 2016. Survey of resampling techniques for improving classification performance in unbalanced datasets. ArXiv e-prints (Aug. 2016). arXiv:stat.AP/1608.06048**

### 采用了GAN但是扩充数据和分类是分开的 不是全局的

如**PacketCGAN: Exploratory Study of Class Imbalance for Encrypted Traffic Classification Using CGAN IEEE T RANSACTIONS ON N ETWORK AND S ERVICE M ANAGEMENT**

中采用CGAN扩充数据 然后进行分类



## 4. 暂时就想到这么多 等待补充