# -*- coding: utf-8 -*-
#将Task 1 (Point Tracking)的预测结果存储为det.txt文件并生成detection文件,用于Task 2的输入

import pandas as pd
import os
import csv

def splitPreResults(resultFile):
    df = pd.read_csv(resultFile, header=0, sep=',')
    #ID, X1, X2, Y1, Y2
    #s2_21_2ch_frame0, 273.09488, 288.01666, 282.8082, 232.95195
    
    # 收集所在sampleID_charm ID
    sidList = []
    for index, row in df.iterrows():
        id = df.iloc[index]["ID"]
        idlist = id.split('_')   #['s2', '21', '2ch', 'frame0']
        sid = idlist[1]+"_"+idlist[2]
        sidList.append(sid)
        sidList = list(set(sidList))
        #print(sidList)
        
        #分别提取每个sid的所有帧的预测结果
        for sid in sidList:
            #savePath = "../Datasets/D2_AVJ_AI_v2/"+ sid + "/det/det.txt" 
            savePath="../Datasets/D2_AVJ_AI_v2/" +sid+ "/det/det.txt"
            
            #print(savePath)
            df2=[]
            for index, row in df.iterrows():
                idlist = df.iloc[index]["ID"].split('_')   #['s2', '21', '2ch', 'frame0']
                if(sid == idlist[1]+"_"+idlist[2]):
                    frameid = int(idlist[3].replace("frame",""))+1 #将frame0先去掉frame,再加1
                    #[frameID, x1, x2, y1, y2]  -> [frameId, tracyID, x, y, width, height]
                    #因为给定的框的宽和高是20,因此预测的点默认在框中间,坐标各减10
                    df2.append([frameid, "1", float(df.iloc[index]["X1"])-10, float(df.iloc[index]["Y1"])-10, "20", "20", "-1", "-1","-1","-1"])
                    df2.append([frameid, "2", float(df.iloc[index]["X2"])-10, float(df.iloc[index]["Y2"])-10, "20", "20", "-1", "-1", "-1", "-1"])
            df2 = pd.DataFrame(df2)
            #print(df2.head())
            if (os.path.exists("../Datasets/D2_AVJ_AI_v2/" + sid) == False):
                print("path not exists: " + savePath)
            else:
                df2.to_csv(savePath, header=False, index=False)
                print("complete: "+sid)
                                        
                        
resultFile = "../Datasets/pred_234.csv"
splitPreResults(resultFile)

