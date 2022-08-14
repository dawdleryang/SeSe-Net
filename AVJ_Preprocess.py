# -*- coding: utf-8 -*-
import glob, cv2, os
import pandas as pd
import numpy as np

Data_Root = '/home/xulei/projects/nhcs/AVJ_updata/data'
Sub_Dirs = ['training', 'testing']


#ch234 = [] #pd.DataFrame(columns=["frameID","X1","Y1","X2", "Y2"])
#IDs = []


def video2Images(videofile, images_dir):
    print (videofile, images_dir)
    basename =  os.path.basename(videofile)
    # read avi file
    videoCapture = cv2.VideoCapture(videofile)
    
    # extract frames,success TRUE otherwise FALSE
    success, frame = videoCapture.read()
    i = 0
    while success:
        frame_id = basename.replace('.avi','_frame{:d}.jpg'.format(i)) #keep the same to IDs
        imgname = os.path.join(images_dir, frame_id)  
        cv2.imwrite(imgname, frame)
        success, frame = videoCapture.read()
        print("save frame: ",imgname)
        i = i + 1

for Sub_Dir in Sub_Dirs:
    Cur_Dir = os.path.join(Data_Root, Sub_Dir) 
    avi_files = glob.glob(os.path.join(Cur_Dir,'*.avi'))
    #print(avi_files)
    avi_files.sort()
    ch234 = []
    IDs = []
    for fl in avi_files:
        basename = os.path.basename(fl)
        print(basename, fl)
        pointfile1 = fl.replace('.avi','_lat.xls')
        df_y1 = pd.read_excel(pointfile1, sheet_name='Sheet1', header=None, index_col=False)  #将是4行30列的文件
        df_x1 = pd.read_excel(pointfile1, sheet_name='Sheet2', header=None, index_col=False)
        pointfile2 = fl.replace('.avi','_sep.xls')
        df_y2 = pd.read_excel(pointfile2, sheet_name='Sheet1', header=None, index_col=False)  #将是4行30列的文件
        df_x2 = pd.read_excel(pointfile2, sheet_name='Sheet2', header=None, index_col=False)
        video2Images(fl, os.path.join(Cur_Dir, "frames")) 

        len_p = df_y1.shape[1] 
        print(len_p)
        for i in range(len_p):
            x1, y1, x2, y2 = np.mean(df_x1[i]), np.mean(df_y1[i]), np.mean(df_x2[i]), np.mean(df_y2[i])
            x1, y1, x2, y2 = "{:.2f}".format(x1), "{:.2f}".format(y1),"{:.2f}".format(x2),"{:.2f}".format(y2)
            print(x1,y1,x2,y2)
            frame_id = basename.replace('.avi','_frame{:d}'.format(i)) 
            IDs.append(frame_id)
            ch234.append([x1,y1,x2,y2])
#combine 21 and 22
    df_id = pd.DataFrame(data=IDs, columns=["ID"])
    df_ch = pd.DataFrame(data=ch234, columns=["X1","Y1","X2","Y2"])
    df_comb = pd.concat([df_id, df_ch], axis=1)
    df_comb.to_csv(os.path.join(Cur_Dir, 'ch234.csv'), index=False)    
