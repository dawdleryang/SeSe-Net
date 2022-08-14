# -*- coding: utf-8 -*-
import glob, cv2, os
import pandas as pd
import numpy as np

Data_Root = '/home/xulei/projects/nhcs/AVJ_updata'
Sub_Dirs = ['1', '2']

TRN_DIR = './data/training/'
TST_DIR = './data/testing/'
RATIO = 0.8 # training sample percentage 

seed = 2022
np.random.seed(seed)

for Sub_Dir in Sub_Dirs:
    Cur_Dir = os.path.join(Data_Root, Sub_Dir)
    avi_files = glob.glob(os.path.join(Cur_Dir,'*.avi'))
    total = len(avi_files) 
    perm = np.random.permutation(total)
    trn_num = int(total*RATIO)
    trn_idx = perm[:trn_num]
    tst_idx = perm[trn_num:]
    print("total {:d}, train {:d}, test {:d}".format(total, trn_num, total-trn_num))
    for idx in trn_idx:
        fl = avi_files[idx]
        basename = os.path.basename(fl)
        dirname = os.path.dirname(fl)
        latname = basename.replace('.avi','_lat.xls')
        sepname = basename.replace('.avi','_sep.xls')
        cmd = "cp " + fl + ' ' + TRN_DIR
        os.system(cmd) 
        chname = basename[-7:-4]
        latfile = os.path.join(dirname,chname,'lat',latname)
        sepfile = os.path.join(dirname,chname,'sep',sepname)
        cmd = "cp " + latfile + ' ' + TRN_DIR
        os.system(cmd)
        cmd = "cp " + sepfile + ' ' + TRN_DIR
        os.system(cmd)    

    for idx in tst_idx:
        fl = avi_files[idx]
        basename = os.path.basename(fl)
        dirname = os.path.dirname(fl)
        latname = basename.replace('.avi','_lat.xls')
        sepname = basename.replace('.avi','_sep.xls')
        cmd = "cp " + fl + ' ' + TST_DIR
        os.system(cmd)
        chname = basename[-7:-4]
        latfile = os.path.join(dirname,chname,'lat',latname)
        sepfile = os.path.join(dirname,chname,'sep',sepname)
        cmd = "cp " + latfile + ' ' + TST_DIR
        os.system(cmd)
        cmd = "cp " + sepfile + ' ' + TST_DIR
        os.system(cmd)


