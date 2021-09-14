import numpy as np
import sys
from functools import partial
import random
import matplotlib as plt
import os  
import glob
import math
import cv2
import h5py
import time

def num(number):
    if number < 10:
        return '00'+str(number)
    return '0'+str(number)

loc = '/home/him030107/pix2pix/data'

f_list = ['0000292']

for folder in f_list:

    print(folder)
    c = True
    location = os.path.join(loc, folder)
    img_list = os.listdir(location)
    number = int(np.floor(len(img_list) / 3))

    for i in img_list:
        if i == '001H.jpg':
            c = False

    if c == True:

        for i in range(number):

            img_num = i + 1
            img_name = num(img_num)
            dir = os.path.join(location, img_name)

            line = cv2.imread(dir + 'line.jpg', cv2.IMREAD_COLOR) / 255.0
            color = cv2.imread(dir + 'hint.jpg', cv2.IMREAD_COLOR) / 255.0
            hint = color * line

            hint = np.array(hint * 255.0)
            cv2.imwrite(dir + 'H.jpg', hint)