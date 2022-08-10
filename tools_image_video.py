# -*- coding: utf-8 -*-
# 用于将JPG/png图像合成MP4格式视频,以及将MP4视频中的每一帧提取为JPG图像
import cv2
import os
import numpy as np


# 将jpg/npg图像合成mp4格式视频
def image2Video(images_dir, video_file):
    # no glob, need number-index increasing
    imglist = os.listdir(images_dir)  # 获取该目录下的所有文件名
    print(imglist)
    if len(imglist) > 0:
        imgname = images_dir + "/" + imglist[0]  # 获取第一个文件
        img = cv2.imread(imgname)  # 读取第一张图片
        size = (img.shape[1], img.shape[0])  # 获取图片宽高度信息
        print("Image size:", size)

        fps = 10  # 30  #每秒传输帧数(Frames Per Second)
        # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        video_writer = cv2.VideoWriter(video_file, fourcc, fps, size)

        for img in imglist:

            if img.endswith('.png') or img.endswith('.PNG') or img.endswith('.jpg') or img.endswith(
                    '.JPG') or img.endswith('.jpge') or img.endswith('.JPGE'):
                imgname = images_dir + "/" + img
                frame = cv2.imread(imgname)
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
                video_writer.write(frame)

        video_writer.release()


# 获取视频的图像宽度, 高度, 帧数量
def getVideoSize(videofile):
    # 获取视频
    vin = cv2.VideoCapture(videofile)
    # 获取视频的帧数量
    d = vin.get(cv2.CAP_PROP_FRAME_COUNT)
    # 获取每一帧图像的宽度和高度
    w, h = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #print("video name:", videofile, ", video width:", w, ", height: ", h, ", frame count: ", d)
    return w, h, d


# 将视频定义保存图片函数
def video2Images(videofile, images_dir):
    # 读取视频文件
    videoCapture = cv2.VideoCapture(videofile)
    # 通过摄像头的方式
    # videoCapture=cv2.VideoCapture(1)

    # 读帧,success为TRUE或者FALSE
    success, frame = videoCapture.read()
    i = 0
    while success:
        i = i + 1
        imgname = os.path.join(images_dir, ("%06d" % i) + '.jpg')  # 将i变为六位数: 1-->000001
        cv2.imwrite(imgname, frame)
        success, frame = videoCapture.read()
        #print('save image:', i)


# --------测试-------------------------------#

if __name__ == '__main__':
    # imagepath="../data/Originial_CT_Images/CP/1133/3351"
    # videofile = "../data/Temp/Videos/CP_1133_3351v2.mp4"

    # 将jpg/npg图像合成mp4格式视频
    # image2Video(imagepath, videofile)

    # 获取视频的图像宽度, 高度, 帧数量
    videofile = "../data/Temp/interpolation_Videos/CP_1133_3351.mp4"
    # videofile = "../data/Temp/Videos/CP_1133_3351.mp4"
    w, h, d = getVideoSize(videofile)

    imagepath_save = "../data/Temp/Images/CP_1133_3351v2/"
    video2Images(videofile, imagepath_save)

