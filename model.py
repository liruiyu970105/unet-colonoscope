import numpy as np
import os
import skimage.io as io
import skimage.transform as trans

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size=(3072, 3072, 3)):
    inputs = Input(input_size)
    print('-------------------------------------unet-------------------------------------')
    print('1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
    print(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    # print('-------------------------------conv1---------------------------------------------------')
    # print(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    # print('-------------------------------conv1---------------------------------------------------')
    # print(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print('-------------------------------pool1---------------------------------------------------')
    # print(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    # print('-------------------------------conv2---------------------------------------------------')
    # print(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    # print('-------------------------------conv2---------------------------------------------------')
    # print(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print('-------------------------------pool2---------------------------------------------------')
    # print(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    # print('-------------------------------conv3---------------------------------------------------')
    # print(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    # print('-------------------------------conv3---------------------------------------------------')
    # print(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print('-------------------------------pool3---------------------------------------------------')
    # print(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    # print('-------------------------------conv4---------------------------------------------------')
    # print(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    # print('-------------------------------conv4---------------------------------------------------')
    # print(conv4)
    drop4 = Dropout(0.5)(conv4)
    # print('-------------------------------drop4---------------------------------------------------')
    # print(drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # print('-------------------------------pool4---------------------------------------------------')
    # print(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    # print('-------------------------------up8---------------------------------------------------')
    # print(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    # print('-------------------------------merge8---------------------------------------------------')
    # print(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    # print('-------------------------------conv9---------------------------------------------------')
    # print(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print('-------------------------------conv10---------------------------------------------------')
    print(conv10)
    model = Model(input=inputs, output=conv10)
    # 模型执行之前必须要编译
    model.compile(
        optimizer=Adam(
            lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
