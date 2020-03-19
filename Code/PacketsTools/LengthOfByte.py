import os
import csv

# 使用说明
# 在InputDir中输入读取文件所在文件夹
# 在OutputDir中输入目标文件夹
# Length中设置最终提取的长度
# 输出结果会依据是否截断（Truncate）或 补零（PaddingHandle）将数据写入相应的文件中


def HandleFile( InputDir ,OutputDir ,Length):
    Files = os.listdir(InputDir)
    for File in Files:
        InputFilePath = InputDir + '\\' + File
        LengthOfByte(InputFilePath,OutputDir,Length,)

def TruncateHandle(DateList , OutputDir , Length  ):
    OutputFile = OutputDir + '//' + 'Truncate.csv'
    with open(OutputFile,'a',newline='') as CsvFile :
        writer = csv.writer(CsvFile)
        writer.writerow(DateList[0:Length])

def PaddingHandle(DateList , OutputDir ,Length ):
    OutputFile = OutputDir + '//' + 'Padding.csv'
    with open(OutputFile,'a',newline='') as CsvFile :
        PaddingNumber = Length - len(DateList)
        for i in range(PaddingNumber):
            DateList.append("0.0")
        writer = csv.writer(CsvFile)
        writer.writerow(DateList)


def LengthOfByte(InputFilePath , OutputDir , Length ):

    with open(InputFilePath,'r') as InputFile:
        Date = InputFile.read()

        DateList = Date.split(',')
        del(DateList[-1])          #之前输入中，最后一个数据为' '
        if len(DateList) >= Length :
            TruncateHandle(DateList,OutputDir,Length)
        else :
            PaddingHandle(DateList,OutputDir,Length)



def Test(Path):
    with open(Path) as f:
        Date = f.read()
        DateList = Date.split(',')
        print(len(DateList))


if __name__ == '__main__':
    InputDir = 'E:\MyProject\ACGAN-Traffic\DateSet\SessionDateNormaling\WebAttack-Xss'
    OutputDir = 'E:\MyProject\ACGAN-Traffic\DateSet\\500\WebAttack-Xss'
    Length = 500
    HandleFile(InputDir,OutputDir,Length)
    # Path = 'E:\MyProject\ACGAN-Traffic\DateSet\\1480\Botnet ARES\Friday-Botnet ARES-h10m00-h13m00.pcap.TCP_192-168-10-5_50054_205-174-165-73_8080.txt'
    # Test(Path)
    # Input = 'E:\MyProject\ACGAN-Traffic\DateSet\SessionDateNormaling\DoS-GoldenEye\Wednesday-DoS-GoldenEye-h11m10-h11m23.pcap.TCP_172-16-0-1_59284_192-168-10-50_80.txt'
    # Output = 'E:\MyProject\ACGAN-Traffic\DateSet\SessionDateNormaling\\test.txt'
    # LengthOfByte(Input,Output,Length=1480)
