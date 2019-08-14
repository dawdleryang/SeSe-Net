import cv2,os,glob
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy.random import permutation
from keras.models import load_model

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.models import Model, model_from_json
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.engine.training import Model as KerasModel

from keras.losses import binary_crossentropy
import keras.backend as K

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Set training data set as car/mri/dog", type=str)
parser.add_argument("--unetsse_epochs", help="Set number of epochs for unet sse", type=int, default=20)

args = parser.parse_args()

data_path = '../input/'+args.dataset+'_data/'
unetsse_ep = args.unetsse_epochs

unet_h, unet_w, unet_c = 128, 128, 3
resnet_h, resnet_w, resnet_c = 224, 224, 4
batch_size = 16
threshold = 0.5

df_train = pd.read_csv(data_path +'train_masks.csv')
ids_all = df_train['img'].map(lambda s: s.split('.')[0])
ids_test = ids_all[900:1000]
ids_train = ids_all[1000:]
print ("train samples: ", len(ids_train))



def get_unet_128(input_shape=(unet_h, unet_w, unet_c),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    #down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    #down4 = BatchNormalization()(down4)
    #down4 = Activation('relu')(down4)
    #down4 = Conv2D(512, (3, 3), padding='same')(down4)
    #down4 = BatchNormalization()(down4)
    #down4 = Activation('relu')(down4)
    #down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(512, (3, 3), padding='same')(down3_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    #up4 = UpSampling2D((2, 2))(center)
    #up4 = concatenate([down4, up4], axis=3)
    #up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    #up4 = Activation('relu')(up4)
    #up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    #up4 = Activation('relu')(up4)
    #up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    #up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(center)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=0.001), loss=sse_bce_dice_loss, metrics=[sse_dice_coeff])

    return model












def load_data_resnet():
    directory = './submit/unet/'
    path = os.path.join(directory, '*.png')
    files = glob.glob(path)

    train_x, train_y = [], []
    #import pdb; pdb.set_trace()
    for fl in files:
        msk = cv2.imread(fl,0)
        flbase = os.path.basename(fl)
        id = flbase[:-11]
        img = cv2.imread(data_path +'train_hq/{}.jpg'.format(id))
        img = cv2.resize(img,(resnet_w,resnet_h))
        msk = cv2.resize(msk,(resnet_w,resnet_h))
        msk = np.expand_dims(msk, axis = 2)
        img = np.concatenate((img,msk),axis = 2)
        lab = float(flbase[-10:-4])
        train_x.append(img)
        train_y.append(lab)
    train_x = np.array(train_x, np.float32) / 255
    train_y = np.array(train_y, np.float32)
    return train_x, train_y


def load_data_unet(ids_input):
    #import pdb; pdb.set_trace()
    train_x, train_y = [], []
    for id in ids_input:#files[:100]:
        img = cv2.imread(data_path +'train_hq/{}.jpg'.format(id))
        msk = cv2.imread(data_path +'train_masks/{}_mask.png'.format(id),0)
        img = cv2.resize(img,(unet_w,unet_h))
        msk = cv2.resize(msk,(unet_w,unet_h))
        msk = np.expand_dims(msk, axis = 2)

        train_x.append(img)
        train_y.append(msk)
    train_x = np.array(train_x, np.float32) / 255
    train_y = np.array(train_y, np.float32) / 255

    return train_x, train_y


def sse_dice_coeff(y_true, y_pred):
    #import pdb; pdb.set_trace()
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def sse_dice_loss(y_true, y_pred):
    loss = 1.0 - sse_dice_coeff(y_true, y_pred)
    return loss

def sse_bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + sse_dice_loss(y_true, y_pred)
    return loss


def simple_loss(y_true, y_pred):
    loss = 1-simple_dice(y_true,y_pred)
    return loss

def simple_dice(y_true, y_pred):
    epsilon = 1e-30
    loss = K.mean(y_true, axis=-1) + K.mean(y_pred, axis=-1)*epsilon
    return loss

def sse_loss(pred):
    def loss(y_true, y_pred):
        #X = model.layers[0].input
        #X = K.concatenate((X,y_pred),axis=3)
        #print(X.shape, y_pred.shape)
        #fun = K.function([model1.layers[0].input],[model1.layers[-1].output])
        #pred = fun([X])
        ##pred = model1.predict(X, batch_size=batch_size)
        return pred

    return loss


def fake_gt(model1,model2,X):

    y1 = model2.predict(X)

    imgs = []
    for i in range(len(y1)):
        img = X[i]
        msk = y1[i] > threshold
        msk = msk * 1.0
        img = cv2.resize(img, (resnet_w, resnet_h))
        msk = cv2.resize(msk, (resnet_w, resnet_h))
        msk = np.expand_dims(msk,axis=2)
        img = np.concatenate((img,msk),axis = 2)
        imgs.append(img)
    imgs = np.array(imgs)
    msks = []
    pred = model1.predict(imgs, batch_size=16)

    print(pred[:10])
    for i in range(len(pred)):
        msk = np.ones((unet_w, unet_h, unet_c),np.float32)*pred[i]
        msks.append(msk)
    return np.array(msks)


model1 = load_model('weights/weights_1.hdf5')
#model2 = load_model('weights/weights_2.hdf5', {'sse_bce_dice_loss':sse_bce_dice_loss, 'sse_dice_coeff': sse_dice_coeff})
model2 = get_unet_128()

def train_model_unet_sse(epochs):
    train_data,train_target = load_data_unet(ids_train)    
       
    best_model_file = "./weights/weights_4.hdf5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True, mode = 'min')

    # this is the augmentation configuration we will use for training
    datagen = ImageDataGenerator(
            #rescale=1./255,
            shear_range=0.0,
            zoom_range=0.1,
            rotation_range=0.0,
            width_shift_range=0.0625,
            height_shift_range=0.0625,
            vertical_flip = True,
            horizontal_flip = False)
    model2.compile(optimizer=Adam(lr=0.001), loss=simple_loss, metrics=[simple_dice])
        
    for epoch in range(epochs):
        print('epoch {} over total {}'.format(epoch, epochs))
        train_target = fake_gt(model1, model2, train_data)
        nVal = int(len(train_target)*0.2)
        perm = permutation(len(train_target))
        X_train = train_data[perm[nVal:]]
        Y_train = train_target[perm[nVal:]]
        X_val = train_data[perm[:nVal]]
        Y_val = train_target[perm[:nVal]]

        print 'Training and Validation Samples: '
        print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape

        # fits the model on batches with real-time data augmentation:
        model2.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
                    samples_per_epoch=len(X_train),
                    nb_epoch=1,
                    verbose=1,
                    validation_data = (X_val, Y_val),
                    #validation_data = datagen.flow(X_val, Y_val, batch_size=16),
                    #nb_val_samples = len(X_val),
                    callbacks = [best_model])

        # predict on test data
        vIoU, vDice, vAcc = unet_predict(model2, ids_test)
        print("epoch {:d} prediction results:".format(epoch))
        print('mean IoU = {:.4f}, mean Dice = {:.4f}, mean Acc = {:.4f}'.format(np.mean(vIoU), np.mean(vDice), np.mean(vAcc)))


def unet_predict(model, ids_input):
    X, y = load_data_unet(ids_input)
    y1 = model.predict(X)
    arrIoU = []
    arrDice = []
    arrAcc = []

    for i in range(len(y1)):
        mask = y1[i] > threshold
        mask = mask * 1.0
        label = y[i]
        comb = mask + label
        L2 = np.sum(comb==2)
        L1 = np.sum(comb==1)
        IoU = L2*1.0 / (L1+L2)
        Dice = L2*2.0 / (L1+2*L2)
        arrIoU.append(IoU)
        arrDice.append(Dice)

        #acc
        L0 = np.sum(comb==0)
        Acc = (L0 + L2)*1.0 / (L0 + L1 + L2)
        arrAcc.append(Acc)

    arrIoU = np.array(arrIoU)
    arrDice = np.array(arrDice)
    #acc
    arrAcc = np.array(arrAcc)
    print('mean IoU = {:.4f}, mean Dice = {:.4f}, mean Acc = {:.4f}'.format(np.mean(arrIoU), np.mean(arrDice), np.mean(arrAcc)))
    return np.mean(arrIoU), np.mean(arrDice), np.mean(arrAcc)

train_model_unet_sse(unetsse_ep)
