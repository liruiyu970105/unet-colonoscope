from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
import re

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky,
                       Building,
                       Pole,
                       Road,
                       Pavement,
                       Tree,
                       SignSymbol,
                       Fence,
                       Car,
                       Pedestrian,
                       Bicyclist,
                       Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it
            # into one-hot vector
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(
            new_mask,
            (new_mask.shape[0],
             new_mask.shape[1] *
             new_mask.shape[2],
             new_mask.shape[3])) if flag_multi_class else np.reshape(
            new_mask,
            (new_mask.shape[0] *
             new_mask.shape[1],
             new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
    
        # global count
        # count += 1
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def trainGenerator(
        batch_size,
        train_path,
        image_folder,
        mask_folder,
        aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        flag_multi_class=False,
        num_class=2,
        target_size=(None,None),
        seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    print('\n\n\n----------------------------------trainGenerator--------------------------------')
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    print('------------------------image generator------------------------')
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    print('------------------------mask generator------------------------')
    
    train_generator = zip(image_generator, mask_generator)
    #print(train_generator)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # (samples, height, width, channels)
    for img, mask in train_generator:
        print('\n\n--------------------start data augmentation------------------------------')
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        print('========================finished data autmentation====================================')
        yield img, mask


def testGenerator(
        test_path,
        #target_size=(3072, 3072),
        flag_multi_class=False,
        as_gray=False):
    print('\n\n\n--------------------testGenerator-----------------------------------------')
    img_mask_list = os.listdir('./data/tissue-test-pos0/')
    #img_list = []
    for item in img_mask_list:
        if re.match('.*mask.jpg', item):
            print(item)
            pass
        else:
            img = io.imread(item, as_gray=as_gray)
            img = img / 255
            #img = trans.resize(img, target_size)
            img = np.reshape(img, (1,) + img.shape)
            yield img


# create .npy data
# If your computer has enough memory, you can create npy files containing
# all your images and masks, and feed your DNN with them.
def geneTrainNpy(
        image_path,
        mask_path,
        flag_multi_class=False,
        num_class=2,
        image_prefix="image",
        mask_prefix="mask",
        image_as_gray=True,
        mask_as_gray=True):
    print('\n\n\n----------------------geneTrainNpy----------------------')
    image_name_arr = glob.glob(
        os.path.join(
            image_path,
            "%s*.png" %
            image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(
            item.replace(
                image_path,
                mask_path).replace(
                image_prefix,
                mask_prefix),
            as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    print('\n\n\n----------------------label visualize--------------------------------')
    print("img shape is :", img.shape)
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    print("img_out shape is :", img_out.shape)
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT,
                             item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
