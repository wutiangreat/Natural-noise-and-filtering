#!/usr/bin/env python
# coding: utf-8

import os

img_path = '..\\multiplicative_noise'
imgs = os.listdir(img_path)
imgNum = len(imgs)
for j in range (imgNum):
    os.system('python D:\\YWX\\filter\\code\\wiener.py %s' % imgs[j])