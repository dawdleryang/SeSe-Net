import os
import numpy as np
import skimage.io as skio
import skimage.transform as skt

from pandas import DataFrame
from pandas.io.parsers import read_csv
import scipy.ndimage as ndi

from keras import backend as K
from keras.engine.training import Model as KerasModel
from keras.layers import Input, Dense, Activation, Flatten, Dropout, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img, transform_matrix_offset_center, flip_axis, random_channel_shift#,  apply_transform
from keras.callbacks import Callback, EarlyStopping
from keras.models import load_model


INPUT_HEIGHT = 128
INPUT_WIDTH = 128
OUTPUTS = 4
LR = 0.001
EPOCHS = 20001
BATCH_SIZE = 192


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=3, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x




class LRDecay(Callback):
    def __init__(self, start=0.001, stop=0.0001, max_epochs=200):
        super(LRDecay, self).__init__()
        self.start, self.stop = start, stop
        self.ls = np.linspace(self.start, self.stop, max_epochs)

    def on_epoch_begin(self, epoch, logs={}):
        new_value = self.ls[epoch]
        K.set_value(self.model.optimizer.lr, new_value)


class CheckpointCallback(Callback):
    def __init__(self, start_index, save_periodic=True, period=500):
        super(CheckpointCallback, self).__init__()
        self.start_index = start_index
        self.save_periodic = save_periodic
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if self.save_periodic:
            if (self.start_index + epoch) % self.period == 0:
                fname = os.path.join('saved_{}.model'.format(self.start_index + epoch))
                self.model.save(fname)


class MyGenerator(ImageDataGenerator):
    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.2,
                 height_shift_range=0.2):

        super(MyGenerator, self).__init__(rotation_range=rotation_range,
                                          width_shift_range=width_shift_range,
                                          height_shift_range=height_shift_range,
                                          fill_mode='constant',
                                          cval=0.,
                                          dim_ordering='th')

    # override to return params
    def my_random_transform(self, x):
        #import pdb; pdb.set_trace()
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_axis - 1
        img_col_index = self.col_axis - 1
        img_channel_index = self.channel_axis - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        mat_rotation = np.array([[np.cos(theta), np.sin(theta), 0],
                                 [-np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        # tx is height, ty is width
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        mat_translation = np.array([[1, 0, -tx],
                                    [0, 1, -ty],
                                    [0, 0, 1]])
        if self.shear_range:
            raise RuntimeError('not implemented')
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])


        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            raise RuntimeError('not implemented')
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
        mat_transform = np.dot(mat_translation, mat_rotation)
        
        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        mat_transform = transform_matrix_offset_center(mat_transform, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        
        return x, mat_transform, tx, ty, theta


    def flow_from_imglist(self, X, y=None, 
                          target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                          batch_size=BATCH_SIZE, shuffle=True, seed=None,
                          save_to_dir=None, save_prefix='', save_format='jpeg'):
        return ImgListIterator(X, y, self,
                               target_size=target_size,
                               batch_size=batch_size, shuffle=shuffle, seed=seed,
                               dim_ordering=self.dim_ordering,
                               save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


    
class ImgListIterator(Iterator):
   
    def __init__(self, X, y, image_data_generator,
                 target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                 batch_size=BATCH_SIZE, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        #import pdb; pdb.set_trace() 
        if y is not None and len(X) != len(y):
            raise Exception('X (images) and y (labels) '
                            'should have the same length. '
                            'Found: X : %s, y : %s' % (len(X), len(y)))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering
        self.X = X # list of images 
        self.y = y # list of tuples of points
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImgListIterator, self).__init__(len(X), batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        # default to th ordering here
        batch_x = np.zeros((current_batch_size, 1,) + self.target_size)
        batch_y = np.zeros((current_batch_size,) + (OUTPUTS, ))
        # build batch of image data
        for i, j in enumerate(index_array):
            height, width = self.X[j].shape
            x = skt.resize(self.X[j], self.target_size)
            x = np.expand_dims(x, axis=0)
            x, transform_matrix, tx, ty, theta = self.image_data_generator.my_random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.y is not None:
                x1, y1 = self.y[j][0]
                x2, y2 = self.y[j][1]
                #offset_x = width / 2
                #offset_y = height / 2
                x1 = x1
                x2 = x2
                y1 = y1
                y2 = y2
                mat = np.array([[y1, y2], [x1, x2], [1, 1]])
                mat = np.dot(transform_matrix, mat)
                batch_y[i, 0] = mat[1, 0]
                batch_y[i, 1] = mat[0, 0]
                batch_y[i, 2] = mat[1, 1]
                batch_y[i, 3] = mat[0, 1]

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                gray_img = batch_x[i, 0, :, :]
                img = array_to_img(gray_img, dim_ordering='th', scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))


        if 1==1:#self.dim_ordering == 'tf':
            batch_tmp = np.zeros((current_batch_size,) + self.target_size + (1,))
            for i in range(current_batch_size):
                batch_tmp[i] = np.transpose(batch_x[i, 0:1, :, :], (1, 2, 0))

            batch_x = batch_tmp
            
                
        if self.y is None:
            return batch_x
        else:
            return batch_x, batch_y



    
def load_data(data_dir, data_csv, load_pts=True):
    
    df = read_csv(data_csv)  # load pandas dataframe
    img_ids = df['ID']
    hR = 96    
    xCnt = 128
    yCnt = 128
    imgs = []
    for img_name in img_ids:
        # read in as grey img [0, 1]
        img = skio.imread('%s/%s.jpg' % (data_dir, img_name), as_gray=True)
        #img1 = img[yCnt-hR:yCnt+hR,xCnt-hR:xCnt+hR]      
        height, width = np.shape(img)[0:2]     
        img = skt.resize(img, (INPUT_HEIGHT, INPUT_WIDTH)) 
        imgs.append(img)
    #import pdb; pdb.set_trace()
    fScale = INPUT_HEIGHT*1.0/height
    if load_pts:
        # pts are not normalized
        x1 = df['X1'].values * fScale
        y1 = df['Y1'].values * fScale
        x2 = df['X2'].values * fScale
        y2 = df['Y2'].values * fScale
    
        pts1 = np.array(zip(x1, y1))
        pts2 = np.array(zip(x2, y2))

    print('Num of images: {}'.format(len(imgs)))

    if load_pts:
        return img_ids, imgs, pts1, pts2
    else:
        return img_ids, imgs


def build_model():
    if K.image_dim_ordering() == 'th':
        inp = Input(shape=(1, INPUT_HEIGHT, INPUT_WIDTH), name='input')
    elif K.image_dim_ordering() == 'tf':
        inp = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 1), name='input')

    x = Convolution2D(32, 3, 3, border_mode='same')(inp)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.1)(x)
    
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(128, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.3)(x)

    #x = Convolution2D(128, 3, 3, border_mode='same')(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    #x = Dropout(0.3)(x)

    x = Flatten()(x)

    x = Dense(1000)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(OUTPUTS)(x)
    outp = Activation('relu')(x)

    learning_method = Adam(lr=LR)
    model = KerasModel(inp, outp)
    model.compile(loss='mean_squared_error', optimizer=learning_method)#optimizer='rmsprop')# optimizer=learning_method)

    print(model.summary())

    return model


def prepare_data(imgs, pts1=None, pts2=None):
    #import pdb; pdb.set_trace()
    X = np.zeros((len(imgs), 1, INPUT_HEIGHT, INPUT_WIDTH))
    y = np.zeros((len(imgs), OUTPUTS))
    #y = np.hstack([pts1, pts2])
    
    for i, img in enumerate(imgs):
        #height, width = img.shape
        height, width = 1, 1  #down sample to 128 128
        X[i] = skt.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
        if pts1 is not None and pts2 is not None:
            y[i, 0] = pts1[i][0] / width
            y[i, 1] = pts1[i][1] / height
            y[i, 2] = pts2[i][0] / width
            y[i, 3] = pts2[i][1] / height

    if pts1 is not None and pts2 is not None:
        return X, y
    else:
        return X
    
    
def train(model, imgs, pts1, pts2, start_index=0):
    y = zip(pts1, pts2)
    train_iter = create_generator().flow_from_imglist(imgs, y, target_size=(INPUT_HEIGHT, INPUT_WIDTH), batch_size=BATCH_SIZE, shuffle=True)
    
#    model.fit({'input': X}, y,
#              batch_size=BATCH_SIZE,
#              nb_epoch=EPOCHS,
#              verbose=1)
    model.fit_generator(train_iter,
                        samples_per_epoch=len(imgs),
                        nb_epoch=EPOCHS,
                        verbose=1,
                        #nb_worker=2,
                        callbacks=[CheckpointCallback(start_index), LRDecay(LR, LR/100, EPOCHS)])
    

def train_model(data_dir, data_csv, save_model_fname, prev_fname=None, start_index=0):
    #import pdb; pdb.set_trace()
    _, imgs, pts1, pts2 = load_data(data_dir, data_csv)
    #import pdb; pdb.set_trace()
    model = build_model()
    if prev_fname is not None:
        model.load_weights(prev_fname)
    #import pdb; pdb.set_trace()
    train(model, imgs, pts1, pts2, start_index)
    model.save(save_model_fname)    


def test_model(model_fname, test_dir, test_csv):
    _, imgs, pts1, pts2 = load_data(test_dir, test_csv)
    X, y = prepare_data(imgs, pts1, pts2)

    model = load_model(model_fname)
    print(model.evaluate(X, y))


def predict(model_fname, data_dir, data_csv, out_csv):
    #import pdb; pdb.set_trace()
    img_ids, imgs, pts1, pts2 = load_data(data_dir, data_csv)
    height, width = imgs[0].shape
    X = prepare_data(imgs)
    
    model = load_model(model_fname)
    p = model.predict(X)

    df = DataFrame({'ID': img_ids,
                    'X1': p[:, 0],
                    'Y1': p[:, 1],
                    'X2': p[:, 2],
                    'Y2': p[:, 3]})
    
    df.to_csv(out_csv, index=False)
    
    
    #display(data_dir, out_csv)

    #import pdb; pdb.set_trace()
    arrA1 = np.zeros(len(img_ids))
    arrA2 = np.zeros(len(img_ids))
    dx = np.zeros(len(img_ids))
    dy = np.zeros(len(img_ids))


    for i in range(len(img_ids)):
        pt1 = [p[i, 0],p[i, 1]]
        pt2 = [p[i, 2],p[i, 3]] 
        arrA1[i], arrA2[i], dx[i], dy[i] = show_img_4(imgs[i], pt1, pt2, pts1[i], pts2[i], img_ids[i])
    print(arrA1-arrA2) 
    print(np.max(arrA1-arrA2), np.min(arrA1-arrA2), np.mean(abs(arrA1-arrA2)), np.std(abs(arrA1-arrA2)))  
    dif = (dx+dy)/2.0
    print(np.max(dif), np.min(dif), np.mean(abs(dif)), np.std(abs(dif)))
    #dif = arrA1-arrA2  
    #np.savez('dif_point', dif_point)

    #npzfile = np.load('dif_angle.npz')
    #dif_angle = npzfile['arr_0']

    #dif = (dif_angle + dif_point)/2.0
    #print dif
    #print np.max(dif), np.min(dif), np.mean(abs(dif)), np.std(abs(dif))
    


def display(data_dir, data_csv):
    #import pdb; pdb.set_trace()
    img_ids, imgs, pts1, pts2 = load_data(data_dir, data_csv)

    for i, img_name in enumerate(img_ids):
        img = imgs[i]#skio.imread('%s/%s.tif' % (data_dir, img_name), as_grey=True)
        
        print(img_name)
        if(i%1==0):  
            if data_csv == 'pred.csv':
                show_img(img, pts1[i]*2, pts2[i]*2)  #the result already downsampled, load_data redo downsampl, here have to upsample
            else:
                show_img(img, pts1[i], pts2[i])
        
def show_img(img, pts1, pts2):
    import sys
    import matplotlib.pyplot as plt
    #import matplotlib.patches as patches

    def press(event):
        if event.key == 'q':
            print('Terminated by user')
            sys.exit()
        elif event.key == 'c':
            plt.close()


    x1, y1 = pts1
    x2, y2 = pts2

    plt.ioff()
    fig = plt.figure(frameon=False)
    fig.canvas.set_window_title('Image')
    fig.canvas.mpl_connect('key_press_event', press)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
    ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
    print(x1, y1, x2, y2)
    plt.scatter([x1, x2], [y1, y2], s=20)
    plt.show()

import math
def show_img_4(img, pts1, pts2, gt1, gt2, title='Image'):
    import sys
    import matplotlib.pyplot as plt
    #import matplotlib.patches as patches

    def press(event):
        if event.key == 'q':
            print('Terminated by user')
            sys.exit()
        elif event.key == 'c':
            plt.close()


    x1, y1 = pts1
    x2, y2 = pts2

    gx1, gy1 = gt1
    gx2, gy2 = gt2

    diff = np.arctan2(y2-y1, x2-x1) * 180/math.pi - np.arctan2(gy2-gy1, gx2-gx1) * 180/math.pi

    plt.ioff()
    fig = plt.figure(frameon=False)
    fig.canvas.set_window_title(title)
    fig.canvas.mpl_connect('key_press_event', press)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
    ax.plot([x1, x2], [y1, y2], 'r-', lw=1)
    print(np.arctan2(y2-y1, x2-x1) * 180/math.pi, np.arctan2(gy2-gy1, gx2-gx1) * 180/math.pi)
    plt.scatter([x1, x2], [y1, y2], s=20)
    ax.plot([gx1, gx2], [gy1, gy2], 'b-', lw=2) 
    plt.scatter([gx1, gx2], [gy1, gy2], s=20)
    #plt.show()
    plt.title(title+'_'+str(diff))
    img_name = os.path.join('./output',title + '.tif') 
    #cv2.imwrite(img_name,img) 
    plt.savefig(img_name)

    return np.arctan2(y2-y1, x2-x1) * 180/math.pi, np.arctan2(gy2-gy1, gx2-gx1) * 180/math.pi, abs((x1+x2)*1.0/2 - (gx1+gx2)*1.0/2), abs((y1+y2)*1.0/2 - (gy1+gy2)*1.0/2) 


        
def create_generator():
    datagen = MyGenerator(rotation_range= 30,  # randomly rotate images in the range (degrees, 0 to 180)
                          width_shift_range= 0.05,  # randomly shift images horizontally (fraction of total width)
                          height_shift_range= 0.05)  # randomly shift images vertically (fraction of total height)

    return datagen



def test_iter(data_dir, data_csv):
    img_ids, imgs, pts1, pts2 = load_data(data_dir, data_csv)
    y = zip(pts1, pts2)
    iter = create_generator().flow_from_imglist(imgs, y, target_size=(INPUT_HEIGHT, INPUT_WIDTH), batch_size=1, shuffle=False)

    for i, (X, y) in enumerate(iter):
        img = X[0, 0, :, :]
        pts1 = y[0, 0], y[0, 1]
        pts2 = y[0, 2], y[0, 3]
        show_img(img, pts1, pts2)
    
    
if __name__ == '__main__':

    import time
    start = time.time()

    #from checkpoint
    #train_model('../../21/frames', '../../21/S21_234ch.csv', 'saved_final.model', 'saved_1500.model', 1501)
    #from scratch
    train_model('./data/training/frames', './data/training/ch234.csv', 'saved_234ch.model', 'saved_15000.model', 15001)
    #framename = basename.replace('.avi','_frame{:d}'.format(i))

    end = time.time()
    print('running time: ', (end - start))
