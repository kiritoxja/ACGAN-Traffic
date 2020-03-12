#----------------------------------------------------------------------------------
#功能：1、以二进制读取Pcap文件，将其转为十六进制输出或将每字节标准化输出
#      2、过滤掉所有的PcapHeader和PacketHeader
#      3、PcapHeader共占24字节，其中前四字节决定大小端机
#      4、PacketHeader共占16字节，其中后四位表示该数据包的总大小


from __future__ import print_function
import os
import random

def HandlePacketHeader(f):
    PacketHeader, length = f.read(16), 0
    if not PacketHeader:
        return None
    for i, byte in enumerate(PacketHeader[-4:]):
        length += int(byte) * 256 ** (3 - i)
    return length

def HandlePacket(PacketSize,f,OutFile,Count):
    while PacketSize!=0:
        date = f.read(1)
        byte = ord(date)

        #-----------------------------
        #若采用十六进制输出
        # print('%02x '%(byte),end='')
        # Count += 1
        # if Count % 16 == 0:
        #     print('')
        # PacketSize -= 1
        #-----------------------------

        #-----------------------------
        #若将每一字节采用0——1整数输出，即将数字标准化
        NormalDate = (float)(byte /255 )
        print('%f,'%(NormalDate),end='')
        OutFile.write(str(NormalDate))
        OutFile.write(',')
        PacketSize -= 1
        #-----------------------------

    return Count

def PrintPacket(PcapPath,OutFilePath):
    with open(OutFilePath,'w') as OutFile:
        with open(PcapPath,'rb') as ReadFile:
            ReadFile.read(24)         #前24字节为PcapHeader
            Count = 0
            while True:
                PacketSize = HandlePacketHeader(ReadFile)          #处理PacketHeader,返回数据包大小
                if PacketSize:
                    Count = HandlePacket(PacketSize, ReadFile,OutFile,Count)
                else:
                    break
        print('')




if __name__ == '__main__':
    InputDir = 'E:\MyProject\ACGAN-Traffic\DateSet\SessionDateFinal\Final-Benign'
    OutputDir = 'E:\MyProject\ACGAN-Traffic\DateSet\SessionDateNormaling\Benign'
    Pcaps = os.listdir(InputDir)
    #---------------------------
    #如果需要随机选择一定量的样本
    random.shuffle(Pcaps)
    n = 10000
    #---------------------------
    for Pcap in Pcaps:
        PcapPath = InputDir + '\\' + Pcap
        OutFilePath = OutputDir + '\\' + Pcap[:-4]+'txt'
        PrintPacket(PcapPath,OutFilePath)

        #--------------------
        #如果需要随机选择一定量的样本
        n -= 1
        if n==0 :
            break
