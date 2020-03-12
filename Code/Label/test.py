from scapy.all import *
import time, datetime

def GenerateLabel(InPcapPath):

    StartTime = '2017-7-8 21:00:00'
    OverTime = '2017-7-8 00:00:00'
    StartTimeArray = time.strptime(StartTime, "%Y-%m-%d %H:%M:%S")
    OverTimeArray = time.strptime(OverTime, "%Y-%m-%d %H:%M:%S")
    StartTimeStamp = float(time.mktime(StartTimeArray))
    OverTimeStamp = float(time.mktime(OverTimeArray))

    print(StartTimeStamp)
    print(OverTimeStamp)

    print('Now :'+InPcapPath)
    Packets = rdpcap(InPcapPath)
    count = 0
    for Packet in Packets:
        TempStamp = repr(Packet.time)

        TimeStamp = float(TempStamp)
        print(TimeStamp)
        if TimeStamp>=StartTimeStamp and TimeStamp<=OverTimeStamp:
            count+=1
    print(count)

    #print(SkipCount)

if __name__ == '__main__':
    InputDir = os.getcwd() + '\InputSet'  #当前输入目录
    #OutputDir = os.getcwd() + '\OutputSet' #当前输出目录
    InputName = "Friday.10001300_00000_20170707210000.pcap"  #设置为当前要处理的文件名
    InPcapPath = InputDir +'\\'+ InputName
    GenerateLabel(InPcapPath)
