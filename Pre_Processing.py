# -*- coding: utf-8 -*-
#根据MIT16的格式提取并生成图像序列数据集和ground truth
import os
from glob import glob
from tools_image_video import *
import pandas as pd

#1 创建图像序列文件夹
def createFolder(originalBasePath, newBasePath):
    filelist = glob(originalBasePath + '/*ch.avi')
    print(len(filelist))
    #遍历并创建文件夹
    for f in filelist:
        fname = os.path.basename(f).split('.')[0]
        #print(fname)
        #创建图像集文件夹和子文件夹
        newpath = os.path.join(newBasePath, fname)
        if not os.path.exists(newpath):
            os.mkdir(newpath)

        imgPath = os.path.join(newBasePath, fname, "img1")
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)

        detPath = os.path.join(newBasePath, fname, "det")
        if not os.path.exists(detPath):
            os.mkdir(detPath)

        gtPath = os.path.join(newBasePath, fname, "gt")
        if not os.path.exists(gtPath):
            os.mkdir(gtPath)

#2 从视频中提取图像序列
def video2image(videoBasePath, imgBasePath):
    filelist = glob(videoBasePath + '/*ch.avi')
    # 遍历文件
    for f in filelist:
        videoname = os.path.basename(f).split('.')[0]
        #创建图像帧文件夹
        imgPath =os.path.join(imgBasePath, videoname, "img1")
        #video2Images(f, imgPath)   #执行视频转换

        # 获取视频的图像宽度, 高度, 帧数量, 生成说明文件
        w, h, d = getVideoSize(f)
        filename =os.path.join(imgBasePath,videoname, "seqinfo.ini")
        with open(filename, "w") as f:
            f.writelines("[Sequence]\n")
            f.writelines("name="+videoname+"\n")          #MOT16-02  图像序列名称
            f.writelines("imDir=img1\n")    #       图像文件夹
            f.writelines("frameRate=30\n")    #30    帧频率(每秒30帧)
            f.writelines("seqLength="+str(int(d)) +"\n")    #600   图像序列的长度
            f.writelines("imWidth="+str(w) + "\n")      #1920   图像宽度
            f.writelines("imHeight="+str(h) +"\n")     #1080     图像高度
            f.writelines("imExt=.jpg")
            f.close()
        print(filename)


#3 生成ground truth
def generateGroundTruth(gtFileBasePath, saveBasePath):
    #gtFileBasePath="../Datasets/D1_AVJ_AI_Original/1/2ch"

    filelist = glob(gtFileBasePath +"/lat/*.xls")
    #遍历文件
    for f in filelist:
        df_detections = pd.DataFrame( columns= ["frameID", "TrajectoryID", "x","y","width", "height",  "Score", "UnName1", "UnName2", "UnName3"])
        df_groundtruth = pd.DataFrame( columns = ["frameID", "TrajectoryID", "x","y","width", "height",  "Score", "Class", "Visibility"])
        casename = os.path.basename(f).split('.')[0][:-4] #文件名为5_2ch_lat.xls, 取"5_2ch",舍去"_lat.xls"或者"_sep.xls"
        print("Processing:", casename)
        #遍历两个点
        pointfiles =[f, f.replace("lat","sep")] #分别是lat和sep的文件路径
        #print("pointfiles:", pointfiles)

        for pointfile in pointfiles:
            #读取excel文件 (x轴和y轴)
            df_x = pd.read_excel(pointfile, sheet_name='Sheet1', header=None, index_col=False)  #将是4行30列的文件
            df_y = pd.read_excel(pointfile, sheet_name='Sheet2', header=None, index_col=False)

            #转置(变成行是frame, 列是坐标位置)
            df_x = df_x.T
            df_y = df_y.T
            df_x = df_x[0]  #只需要第一列,即起始x和y
            df_y = df_y[0]
            #print(df_x.head())
            #print(df_y.head())

            df_det = pd.concat([df_x, df_y], axis= 1)   #axis = 1表示列拼接(向右) #将x和y拼接
            #设置DataFrame名并根据MOT16的det文件格式设置 "1 -1 241  295 20 20 100.0 -1 -1 -1"
            df_det.columns = ["x", "y"]  #设置标题
            df_det["frameID"] = df_det.index+1 #将索引号(即frame ID) 从1开始
            df_det["width"]=20
            df_det["height"] = 20
            df_det["TrajectoryID"] =-1
            df_det["Score"] = 100.0
            df_det["UnName1"] = -1
            df_det["UnName2"] = -1
            df_det["UnName3"] = -1
            colName = ["frameID", "TrajectoryID", "x","y","width", "height",  "Score", "UnName1", "UnName2", "UnName3"]
            df_det = df_det[colName]
            #print(df_det.head())
            #生成det文件
            df_detections = df_detections.append(df_det, ignore_index=True)

            df_gt = pd.DataFrame(df_det, columns=["frameID", "TrajectoryID", "x","y","width", "height",  "Score"])
            df_gt["TrajectoryID"] = 1 if "lat" in pointfile else 2
            df_gt["Class"] = df_gt["TrajectoryID"]
            df_gt["Visibility"] = 1.0
            #print(df_gt.head())
            # 生成gt文件
            df_groundtruth = df_groundtruth.append(df_gt, ignore_index=True)

        if os.path.exists(os.path.join(saveBasePath, casename)):
            #print(df_detections)
            df_detections = df_detections.sort_values(by=["frameID", "x"], ascending=[True, True])  #两列都升序排列
            savePath_det =os.path.join(saveBasePath, casename, "det", "det.txt")
            df_detections.to_csv(savePath_det, header=False, index=False)

            df_groundtruth = df_groundtruth.sort_values(by=["frameID", "TrajectoryID"], ascending=[True, True])  # 两列都升序排列
            savePath_gt =os.path.join(saveBasePath, casename, "gt", "gt.txt")
            df_groundtruth.to_csv(savePath_gt, header=False, index=False)
        else:  #有些图像集不存在,但有相应的坐标文件
            print("case not exists:", casename)


if __name__ == "__main__":
    # 1 创建图像序列文件夹
    newBasePath = "../Datasets/D2_AVJ_AI_v2"
    for i in ["1","2","3"]:
        originalBasePath = "../Datasets/D1_AVJ_AI_Original/"+i  #换成1, 2, 3分别执行一次
        createFolder(originalBasePath, newBasePath)

    #2 从视频中提取图像序列
    video2image(originalBasePath, newBasePath)

    # 3 生成ground truth (分别执行各个文件夹)
    dirs = ["1/2ch", "1/3ch", "1/4ch", "2/2ch", "2/3ch", "2/4ch", "3/2ch", "3/3ch", "3/4ch"]
    for dir in dirs:
        gtFileBasePath = "../Datasets/D1_AVJ_AI_Original/" + dir
        generateGroundTruth(gtFileBasePath, newBasePath)
