# 因为每一个原始数据集体积过于庞大，因此不进行批量化处理
#输入目录：.\InputSet
#输出目录：.\OutputSet
#流程 ：  1、将待处理Pcap文件放入.\InputSet目录
#         2、设置OutputName
#         3、在IPSet中设置 IP地址
#         4、Go


from scapy.all import *
import os

def GenerateLabel(InputDir,IpSet,OutPcapPath):

    #================时间设置=============================
    StartTime = '2017-7-3 10:00:00'
    OverTime = '2017-7-3 10:10:00'
    #=====================================================

    StartTimeArray = time.strptime(StartTime, "%Y-%m-%d %H:%M:%S")
    OverTimeArray = time.strptime(OverTime, "%Y-%m-%d %H:%M:%S")
    StartTimeStamp = float(time.mktime(StartTimeArray))
    OverTimeStamp = float(time.mktime(OverTimeArray))



    OutPcap = PcapWriter(OutPcapPath)
    SkipCount = 0

    Pcaps = os.listdir(InputDir)
    print(len(Pcaps))
    for Pcap in Pcaps :
        PcapPath = InputDir + '\\'+ Pcap
        print('Now :' + PcapPath)
        Packets = rdpcap(PcapPath)
        for Packet in Packets:
            TempStamp = repr(Packet.time)
            TimeStamp = float(TempStamp)
            if TimeStamp >= StartTimeStamp and TimeStamp <= OverTimeStamp:
                try:
                    # if (Packet['IP'].src in IpSet) and  (Packet['IP'].dst in IpSet):
                    #     print(repr(Packet))
                    #     OutPcap.write(Packet)

                    #print(repr(Packet))
                    OutPcap.write(Packet)
                except:
                    SkipCount+=1
                    continue
    print(SkipCount)

if __name__ == '__main__':
    InputDir = os.getcwd() + '\InputSet'  #当前输入目录
    OutputDir = os.getcwd() + '\OutputSet' #当前输出目录

    OutputName = "Benign3.pcap"

    OutPcapPath = OutputDir +'\\' + OutputName    #输出文件名
    IpSet=['192.168.10.51', '172.16.0.1']        #填入需获取的IP地址
    GenerateLabel(InputDir, IpSet, OutPcapPath)
