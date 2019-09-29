import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


for j in range(27, 28):
    print(j)
    img = nib.load('.\\image\\volume-' + str(j) + '.nii')
    img_arr = img.get_fdata()
    img_arr = np.squeeze(img_arr)  # 三维图片
    os.system('mkdir .\\image\\v' + str(j))
    for i in range(img_arr.shape[2]):
        img_name = '.\\image\\v'+ str(j) +'\\' + 'Patient_' + str(j) + '_' + str(i) + '_mask.png'
        image = img_arr[:, :, i] * 255 / 4 + 0.3
        cv2.imwrite(img_name, np.transpose(image))  # 要旋转一下
    img = nib.load('.\\label\\segmentation-' + str(j) + '.nii')
    img_arr = img.get_fdata()
    img_arr = np.squeeze(img_arr)  # 三维图片
    os.system('mkdir .\\label\\s' + str(j))
    for i in range(img_arr.shape[2]):
        img_name = '.\\label\\s'+ str(j) +'\\' + 'Patient_' + str(j) + '_' + str(i) + '_mask.png'
        image = img_arr[:, :, i] * 255 / 4 + 0.3
        cv2.imwrite(img_name, np.transpose(image))  # 要旋转一下
