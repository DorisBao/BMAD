import os
import SimpleITK
import pydicom
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import csv


filename = 'stage_2_detailed_class_info.csv'
image_file = '/home/jinan/Datasets/Medical-datasets/chest-rsna/stage_2_train_images/'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    count =0
    for i in reader:
        image_id = i[0]
        img_name = image_file + image_id +'.dcm'
        print(img_name)
        ds = pydicom.read_file(img_name)
        img = Image.fromarray(ds.pixel_array)
        if i [1] =='Lung Opacity':
            save_file = './abnormal'+'/Lung_Opacity_'+i[0]+'.png'

        elif i[1] == 'Normal':
            save_file = './normal'+'/'+i[1]+'-'+i[0]+'.png'

        elif i[1] == 'No Lung Opacity / Not Normal':
            save_file = './abnormal'+'/'+'No-Not-Normal_'+i[0]+'.png'

        img.save(save_file)


