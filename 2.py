from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import *
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.applications.vgg19 import VGG19
from tqdm import tqdm
import tensorflow.keras.backend as K
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from functools import partial
import random
import matplotlib as plt
import os  
import glob
import math
import cv2
import h5py
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1'

class DataGenerator(Sequence):
    def __init__(self, imgs_dir, batch_size = 8, img_size = 512, shuffle = True):
        self.indexes = os.listdir(imgs_dir)
        self.len = int(np.floor(len(self.indexes) / 2))
        self.size = int(img_size / 2)
        self.imgs_dir = imgs_dir
        self.shuffle = shuffle
        self.batch_size = batch_size

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def num(self, number):
        if number < 10:
            return '00'+str(number)
        return '0'+str(number)

    def __data_generation__(self, folder):
        location = os.path.join(self.imgs_dir, folder)
        imgs = []
        hints = []

        img_list = os.listdir(location)
        number = int(np.floor(len(img_list) / 3))

        if number < self.batch_size:
            return self.__data_generation__(random.choice(self.indexes))

        img_list = list(range(number))
        np.random.shuffle(img_list)

        for i in range(self.batch_size):

            img_num = img_list[i] + 1
            img_name = self.num(img_num)
            dir = os.path.join(location, img_name)
            
            img = cv2.imread(dir + '.jpg', cv2.IMREAD_COLOR) / 255.0
            line = cv2.imread(dir + 'line.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
            line = np.expand_dims(line, axis = 2)
            color = cv2.imread(dir + 'hint.jpg', cv2.IMREAD_COLOR) / 255.0
            hint = np.concatenate((line, color), axis = 2)

            imgs.append(img)
            hints.append(hint)

        imgs = np.array(imgs)
        hints = np.array(hints)

        return imgs, hints

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        a_full, a_hint = self.__data_generation__(self.indexes[index])
        b_full, b_hint = self.__data_generation__(self.indexes[self.len + index])

        return a_full, a_hint, b_full, b_hint

def conv2d(layer, filters = 128, kernel_size = (3,3), strides = (1,1), padding = 'same'):
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, 
           kernel_initializer = 'he_normal', bias_initializer = 'zeros')(layer)
    return x

def crop(imgs, crop_size = 256):
    img_size = imgs.shape[2]
    crops = []

    for mat in imgs:
        t = np.random.randint(img_size - crop_size, size = 2)
        crop = mat[t[0] : t[0] + crop_size, t[1] : t[1] + crop_size]
        crops.append(crop)

    crops = np.array(crops)
    return crops

def normalize(layer, epsilone = 1e-5):
    layer_shape = K.int_shape(layer)
    axis = list(range(len(layer_shape)))
    del axis[-1]
    del axis[0]

    mean = K.mean(layer, axis, keepdims = True)
    std = K.std(layer, axis, keepdims = True) + epsilone

    return mean, std

def AdaIn(layer, latent):
    
    mean, std = normalize(layer)
    norm = (layer - mean) / std

    mean, std = normalize(latent)

    return norm * std + mean

class GAN():
    def __init__(self, dir, hint = 'hint', img_size = 512, style_size = 256, gen_dep = 4, batch_size = 4, lr = 0.0001):
        self.gen_dep = gen_dep
        self.batch_size = batch_size
        self.model_dir = os.path.join(dir, 'model')
        self.data_dir = os.path.join(dir, 'data')
        self.test_dir = os.path.join(dir, 'test')
        self.img_size = img_size
        self.test_num = 0
        self.lr = lr
        self.len = int(np.floor(len(os.listdir(self.data_dir)) / 2))
        self.datagenerator = DataGenerator(imgs_dir = self.data_dir, batch_size = self.batch_size)

        self.img_shape = [self.img_size, self.img_size, 4]

        self.encs = []
        self.decs = []
        self.gens = []

        self.style_shape = [style_size, style_size, 3]
        self.encoder = VGG19(weights = 'imagenet', include_top = False, input_shape = self.style_shape)
        self.slayer_name = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']
        self.slayer = {}
        for i in range(4):
            self.slayer[(1<<(i+6))] = Model(inputs = self.encoder.input, outputs = self.encoder.get_layer(self.slayer_name[i]).output)
        for l in self.encoder.layers:
            l.trainable = False

        self.build_discriminator()
        self.build_generator()

        self.loss_d = []
        self.loss_g = []

        print(self.gens)

    def train(self, epoch = 10):

        ones = np.ones((self.batch_size, 1, 1, 1))
        zeros = np.zeros((self.batch_size, 1, 1, 1))
        mixes = np.ones((self.batch_size, 1, 1, 1)) / 2.0

        pbar = tqdm(total = self.len * epoch, position = 0, leave = True)
        step_size = int(np.floor(epoch / (2 * self.gen_dep)))

        model = None

        for e in range(epoch):
            if e % step_size == 0:
                temp = int(e / step_size)
                
                if temp < self.gen_dep:

                    hint = Input(shape = self.img_shape)
                    style = Input(shape = self.style_shape)
                    full = Input(shape = [self.img_size, self.img_size, 3])

                    gen = self.gens[temp]

                    gen.trainable = True
                    self.disc.trainable = False

                    fake = gen([hint, style])
                    d_fake = self.disc(fake)

                    g_model = Model(inputs = [hint, style], outputs = fake)
                    g_model.compile(loss = mse, optimizer = self.Adamas())

                    model = Model(inputs = [hint, style], outputs = [d_fake, fake])
                    model.compile(loss = [mse, self.custom_loss], optimizer = self.Adamas())
                    
                    gen.trainable = False
                    self.disc.trainable = True

                    fake = gen([hint, style])
                    mix = self.mix(full, fake, self.batch_size)

                    d_real = self.disc(full)
                    d_fake = self.disc(fake)
                    d_mix = self.disc(mix)
                    
                    partial_gp_loss = partial(self.gp_loss, mix = mix)
                    partial_gp_loss.__name__ = 'gradient_penalty'

                    d_model = Model(inputs = [hint, style, full], outputs = [d_real, d_fake, d_mix])
                    d_model.compile(loss = ['mse', 'mse', partial_gp_loss], optimizer = self.Adamas(), loss_weights = [1, 1, 10])
                
            for i, (a_full, a_hint, b_full, b_hint) in enumerate(self.datagenerator):

                a_style = crop(a_full)
                b_style = crop(b_full)

                loss_disc = d_model.train_on_batch([a_hint, b_style, a_full], [ones, zeros, mixes])
                loss_disc += d_model.train_on_batch([a_hint, a_style, b_full], [ones, zeros, mixes])
                loss_disc += d_model.train_on_batch([b_hint, b_style, a_full], [ones, zeros, mixes])
                loss_disc += d_model.train_on_batch([b_hint, a_style, b_full], [ones, zeros, mixes])

                loss_disc = np.sum(loss_disc)
                loss_disc /= 4
                self.loss_d.append(loss_disc)

                rand = np.random.randint(100)

                if e < self.gen_dep * step_size:
                    g_model.train_on_batch([a_hint, a_style], a_full)
                    g_model.train_on_batch([b_hint, b_style], b_full)

                loss_gen = model.train_on_batch([a_hint,a_style], [ones, a_full])
                loss_gen += model.train_on_batch([a_hint,b_style], [ones, b_full])
                loss_gen += model.train_on_batch([b_hint,a_style], [ones, a_full])
                loss_gen += model.train_on_batch([b_hint,b_style], [ones, b_full])
                
                loss_gen = np.sum(loss_gen)
                loss_gen /= 4
                self.loss_g.append(loss_gen)
                
                pbar.update(1)

            self.eval(gen, e, style_name = 'meri')
            self.eval(gen, e, style_name = 'renian')

        pbar.close()

    def eval(self, model, num, hint_name = '001', style_name = '005'):
        size = int(self.img_size / 2)

        line = cv2.imread(os.path.join(self.test_dir, hint_name) + 'line.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
        color = cv2.imread(os.path.join(self.test_dir, hint_name) + 'hint.jpg', cv2.IMREAD_COLOR) / 255.0
        style = cv2.imread(os.path.join(self.test_dir, style_name) + '.jpg', cv2.IMREAD_COLOR) / 255.0

        tx = int(np.random.randint(size, size = 1))
        ty = int(np.random.randint(size, size = 1))
        style = style[tx : tx + size, ty : ty + size]
        
        line = np.expand_dims(line, axis = 0)
        line = np.expand_dims(line, axis = 3)
        color = np.expand_dims(color, axis = 0)
        style = np.expand_dims(style, axis = 0)

        hint = np.concatenate((line, color), axis = 3)

        result = model.predict([hint, style])
        result = result[0]
        result = result * 255.0
        self.test_num += 1
        cv2.imwrite(os.path.join(self.test_dir, style_name + '_' + str(num) + '.jpg'), result)

    def build_generator(self):

        filters = 64

        def down_block(filter):

            input = Input(shape = self.img_shape)

            if len(self.encs) > 1:
                layer = self.encs[-1](input)
            else:
                layer = input

            layer = conv2d(layer, filters = filter, strides = (2,2))
            layer = LeakyReLU(0.01)(layer)
            layer = conv2d(layer, filters = filter)
            layer = LeakyReLU(0.01)(layer)
            
            model = Model(inputs = input, outputs = layer)

            self.encs.append(model)
        
        def up_block(filter):

            input = Input(shape = self.img_shape)

            t_shape = K.int_shape(self.encs[-1].output)
            t_shape = t_shape[1:]
            t_in = Input(shape = t_shape)

            style_in = Input(shape = self.style_shape)
            style = self.slayer[filter](style_in)

            layer = UpSampling2D(size = (2,2))(t_in)
            layer = Concatenate(axis = 3)([layer, self.encs[-2](input)])

            layer = conv2d(layer, filters = filter)
            layer = LeakyReLU(0.01)(layer)

            layer = AdaIn(layer, style)
            layer = LeakyReLU(0.01)(layer)

            layer = conv2d(layer, filters = filter)
            layer = LeakyReLU(0.01)(layer)

            if self.decs != []:
                layer = self.decs[-1]([input, layer, style_in])
            
            model = Model(inputs = [input, t_in, style_in], outputs = layer)

            self.decs.append(model)

        input = Input(shape = self.img_shape)
        x = conv2d(input, filters = filters)
        model = Model(inputs = input, outputs = x)
        self.encs.append(model)

        for i in range(self.gen_dep):

            filters *= 2
            down_block(filters)
            up_block(int(filters / 2))

            if i == 0:
                t_shape = K.int_shape(self.decs[0].output)
                t_shape = t_shape[1:]
                input = Input(shape = t_shape)
                x = conv2d(input, 3)

                model = Model(inputs = input, outputs = x)
                self.model_end = model

            input = Input(shape = self.img_shape)
            style_in = Input(shape = self.style_shape)

            layer = self.encs[-1](input)
            layer = self.decs[-1]([input, layer, style_in])
            img = self.model_end(layer)

            model = Model(inputs = [input, style_in], outputs = img)
            self.gens.append(model)

    def build_discriminator(self):

        filters = 16
        img = Input(shape = [512, 512, 3])
        x = conv2d(img, filters = filters)

        img_size = K.int_shape(x)
        img_size = img_size[1]
        
        while img_size > 3:

            x = conv2d(x, filters, strides = (2, 2), padding = 'valid')
            x = LeakyReLU(0.01)(x)
            x = conv2d(x, filters, padding = 'valid')
            x = LeakyReLU(0.01)(x)

            img_size = K.int_shape(x)
            img_size = img_size[1]
            if img_size > 8:
                filters *=2

        x = conv2d(x, filters, kernel_size = (img_size, img_size), padding = 'valid')

        while filters > 1:

            filters = int(filters / 2)
            x = conv2d(x, filters, kernel_size = (1,1), padding = 'valid')

        self.disc = Model(img, x)
    
    def gp_loss(self, y_true, y_pred, mix):
    
        gradients = K.gradients(y_pred, mix)[0]
        gradients = K.square(gradients)
        gradients = K.sum(gradients, axis = np.arange(1, len(gradients.shape)))
        gradients = K.sqrt(gradients)
        gradients = K.square(1 - gradients)

        return K.mean(gradients)

    def custom_loss(self, y_true, y_pred):

        t_mean, t_std = normalize(y_true)
        f_mean, f_std = normalize(y_pred)

        mean_loss = mean_squared_error(t_mean, f_mean)
        std_loss = mean_squared_error(t_std, f_std)

        norm_true = (y_true - t_mean) / t_std
        norm_pred = (y_pred - f_mean) / f_std

        content_loss = mean_squared_error(norm_true, norm_pred)

        return mean_loss + std_loss + content_loss
    
    def mix(self, real, fake, batch_size):
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * real) + ((1 - alpha) * fake)

    def Adamas(self, beta_1 = 0, beta_2 = 0.99, decay = 0.00001):
        return Adam(self.lr, beta_1 = beta_1, beta_2 = beta_2, decay = decay)

if __name__ == '__main__':
    model = GAN('/home/him030107/pix2pix')
    model.train(epoch = 100)

    nohup python3 -u main.py &
    tail -f nohup.out