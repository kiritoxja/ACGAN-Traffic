# 500字节数据集
## 处理过程
---
1. 将原始数据集做标签；
2. 将分类好的数据集中的流量数据转为会话；
3. 过滤掉由受害者IP发出的会话；
4. 结合攻击特点及分布特点对会话数据集进行过滤；
5. 将会话数据集按字节位，将一个字节码转为2位16进制，并2位16进制转化为0-255的整数；
6. 将0-255整数归一化为0-1的整数；
7. 取前500个字节码。过程中如果大小超过500个字节，截断处理，不够500个字节码的补零处理。相应的分别存在Padding和Truncate文件中。
## 数据集分布
---
![810hJe.png](https://s1.ax1x.com/2020/03/15/810hJe.png)
## 统计信息
---
## Benign
- Number:10000
- Padding:4185
- Truncate:5815
## Botnet-ARES
- Number:1226
- Padding:490
- Truncate:736
## BruteForce
- Number:70
- Padding:0
- Truncate:70
## DDos
- Number:10000
- Padding:0
- Truncate:10000
## DoS-GoldenEye
- Number:7472
- Padding:0
- Truncate:7472
## Dos-Slowhttptest
- Number:1103
- Padding:0
- Truncate:1103
## DoS-Slowloris
- Number:1784
- Padding:0
- Truncate:1784
## FTP-Patator
- Number:3912
- Padding:0
- Truncate：3912
## SSH-Patator
- Number:2404
- Padding:0
- Truncate:2404
## Hulk
- Number:10000
- Padding:0
- Truncate:10000
## PortScan
- Number:10000
- Padding:9985
- Truncate:15
## WebAttack-Sql Injection
- Number:9
- Padding:0
- Truncate:9
## WebAttack-Xss
- Number:18
- Padding:0
- Truncate:18
## Infiltration
- Number:6
- Padding:0
- Truncate:6