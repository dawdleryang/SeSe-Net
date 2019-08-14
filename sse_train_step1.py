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
parser.add_argument("--train_percent", help="Set the percentage of data used for training", type=int, default=100)
parser.add_argument("--unet_epochs", help="Set number of epochs for unet", type=int, default=20)
parser.add_argument("--resnet_epochs", help="Set number of epochs for densenet", type=int, default=1000)
parser.add_argument("--res_refine", help="Refining from previous model", type=bool, default=False)

args = parser.parse_args()

res_refine = args.res_refine
data_path = '../input/'+args.dataset+'_data/'
unet_ep = args.unet_epochs
resnet_ep = args.resnet_epochs
train_no = int(800 / 100 * args.train_percent)

unet_h, unet_w, unet_c = 128, 128, 3
resnet_h, resnet_w, resnet_c = 224, 224, 4
batch_size = 16
threshold = 0.5

df_train = pd.read_csv(data_path + 'train_masks.csv')
ids_all = df_train['img'].map(lambda s: s.split('.')[0])
ids_train = ids_all[:train_no]
ids_val = ids_all[800:900]
ids_test = ids_all[900:1000]

#ids_train1 = ids_all[:800]
#ids_train2 = ids_all[1000:]
#ids_train = np.concatenate((ids_train1, ids_train2),axis=-1)





def load_data_resnet():
    directory = './output/unet/'
    path = os.path.join(directory, '*.png')
    files = glob.glob(path)
    # for balance distribution, do selection
    files = get_unet_selected_output(files)

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
    print(np.max(train_x),np.max(train_y))
    return train_x, train_y


def get_unet_selected_output(files):

    selected_files = []

    for i in range(12):
        # case 0 and case 1
        if i == 10:
            dice_str = '0.0000'
        elif i == 11:
            dice_str = '1.0000'
        else:
            dice_str = str(i/10.0)

        # case 0.0 < dice < 0.1
        if i == 0:
            file = [mask for mask in files if mask.find(dice_str) != -1 and mask.find('0.0000') == -1]
        # other case
        else:
            file = [mask for mask in files if mask.find(dice_str) != -1]
            
        # get 2000 for each case
        selected_files.extend(file[:2000])

    return selected_files


def load_data_unet(ids_input):

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
    loss = 1 - sse_dice_coeff(y_true, y_pred)
    return loss

def sse_bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + sse_dice_loss(y_true, y_pred)
    return loss

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


def build_model_resnet50():

    base_model = ResNet50(include_top=False, weights = None, input_tensor=None, input_shape=(resnet_h, resnet_w, resnet_c))

    x = base_model.output

    x = Flatten()(x)

    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    predictions  = Dense(1, activation='linear', name = 'regression')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    model.compile(optimizer=Adam(lr=0.001), loss='mse')

    return model


def train_model_resnet(nfolds=1, nb_epoch=10):
    print("loading dataset:")
    batch_size = 32
    random_state = 51
    train_data, train_target = load_data_resnet()

    print("loaded samples: ", train_data.shape, train_target.shape)

    num_fold = 0

    for i in range(nfolds):
        nVal = int(len(train_target)*0.2)
        perm = permutation(len(train_target))
        X_train = train_data[perm[nVal:]]
        Y_train = train_target[perm[nVal:]]
        X_val = train_data[perm[:nVal]]
        Y_val = train_target[perm[:nVal]]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_val), len(Y_val))

        print 'Training and Validation Samples: '
        print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape

        if res_refine == True:
		model = load_model('weights/weights_1.hdf5')
	else:
		model = build_model_resnet50()
	
        best_model_file = "./weights/weights_1.hdf5"
        best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)

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

        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
                    samples_per_epoch=len(X_train),
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_data = (X_val, Y_val),
                    #validation_data = datagen.flow(X_val, Y_val, batch_size=16),
                    #nb_val_samples = len(X_val),
                    callbacks = [best_model])


def train_model_unet(epochs):

    model = get_unet_128()
    #train_data,train_target = load_data_unet(ids_train)
    X_train, Y_train = load_data_unet(ids_train)
    X_val, Y_val = load_data_unet(ids_val)

    best_model_file = "./weights/weights_2.hdf5"
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
	
    for epoch in range(epochs):
        print 'Training and Validation Samples: '
        print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape

        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
                    samples_per_epoch=len(X_train),
                    nb_epoch=1,
                    verbose=1,
                    validation_data = (X_val, Y_val),
                    #validation_data = datagen.flow(X_val, Y_val, batch_size=16),
                    #nb_val_samples = len(X_val),
                    callbacks = [best_model])

        
        #generate various masks for resnet training
        print('Predicting on {} samples with batch_size = {}...'.format(len(ids_train), batch_size))
        for start in tqdm(range(0, len(ids_train), batch_size)):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train))
            ids_test_batch = ids_train[start:end]
            for id in ids_test_batch.values:
                img = cv2.imread(data_path +'train_hq/{}.jpg'.format(id))
                img = cv2.resize(img, (unet_w, unet_h))
                msk = cv2.imread(data_path +'train_masks/{}_mask.png'.format(id),0)
                msk = cv2.resize(msk, (unet_w, unet_h))
                x_batch.append(img)
                y_batch.append(msk)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            preds = model.predict_on_batch(x_batch)
            preds = np.squeeze(preds, axis=3)
            for k in range(len(preds)):
                #import pdb; pdb.set_trace()
                pred = preds[k]
                prob = cv2.resize(pred, (unet_w, unet_h))
                mask = prob > threshold
                mask = mask * 1.0
                label = y_batch[k]
                comb = mask + label
                L2 = np.sum(comb>=2)
                L1 = np.sum(comb>=1)
                Dice = L2*2.0/(L1+L2)
                save = './output/unet/'+ids_test_batch.values[k] + '_{:.4f}'.format(Dice) +'.png'
                cv2.imwrite(save, mask*255)

                mask = 1.0 - mask
                comb = mask + label
                L2 = np.sum(comb>=2)
                L1 = np.sum(comb>=1)
                Dice1 = L2*2.0/(L1+L2)
                save = './output/unet/'+ids_test_batch.values[k] + '_{:.4f}'.format(Dice1) +'.png'
                cv2.imwrite(save, mask*255)

                #print(Dice, Dice1)
                 
        print("epoch {:d} saving masks finished".format(epoch))
        vIoU, vDice, vAcc = unet_predict(model, ids_test)
        print("epoch {:d} prediction results:".format(epoch))
        print('mean IoU = {:.4f}, mean Dice = {:.4f}, mean Acc = {:.4f}'.format(np.mean(vIoU), np.mean(vDice), np.mean(vAcc)))

    # lastly generate worst and perfect masks
    gen_0_1(ids_train)

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


def gen_0_1(ids_input):
    for k in range(len(ids_input.values)):
        id = ids_input.values[k]
        msk = cv2.imread(data_path + 'train_masks/{}_mask.png'.format(id),0)
        msk = cv2.resize(msk, (resnet_w, resnet_h))
        Dice = 1.000000
        save = './output/unet/'+ id + '_{:.4f}'.format(Dice) +'.png'
        cv2.imwrite(save,msk)
        msk = 255 - msk
        Dice = 0.000000
        save = './output/unet/'+ id + '_{:.4f}'.format(Dice) +'.png'
        cv2.imwrite(save,msk)


if res_refine == False:
	# train a unet model and generate the masks for resnet
	train_model_unet(unet_ep)
        model = ''
        unet_predict(model,

# train a resent for regression
train_model_resnet(1,resnet_ep)

