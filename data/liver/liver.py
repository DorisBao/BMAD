import os
import numpy as np
from PIL import Image

#path = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/train'

path = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label'

test_label = os.listdir(path)

path = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img'

test = os.listdir(path)

prefix = []

for i in range(len(test_label)):
    prefix.append(test_label[i].split('_')[1]+'_'+test_label[i].split('_')[2])

for i in range(len(test)):
    test[i] = test[i].split('_')[1]+'_'+test[i].split('_')[2]

print(set(test_label) == set(test))
k=0
j = 0




os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/'))
os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/'))
os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/'))
os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/'))
os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/'))
os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/'))


for i in range(len(prefix)):
    test_label_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label/' + 'liver_' + prefix[i] + '_label.png'
    test_img_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img/' + 'liver_' + prefix[i] + '_segmentation_DIY_hist.png'

    test_label_idx = np.array(Image.open(test_label_idx))
    test_img_idx = np.array(Image.open(test_img_idx))

    #print(np.max(img))
    if np.max(test_label_idx) >= 1:
        isgood = 'Ungood' # 733 valid is 73 img
        k = k+1
    else:
        isgood = 'good' # 926 valid is 93 img
        j = j+1

    if isgood == 'Ungood' and k > 73:
        test_label_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label/' + 'liver_' + prefix[i] + '_label.png'
        test_img_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img/' + 'liver_' + prefix[i] + '_segmentation_DIY_hist.png'
        if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, )) == False:
            os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, ))
        if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood, )) == False:
            os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood, ))
        target_img = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, prefix[i] + '.png')
        target_label = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood,prefix[i] + '.png')

    elif isgood == 'good' and j > 93:
        test_label_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label/' + 'liver_' + prefix[i] + '_label.png'
        test_img_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img/' + 'liver_' + prefix[i] + '_segmentation_DIY_hist.png'
        if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, )) == False:
            os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, ))
        if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood, )) == False:
            os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood, ))
        target_img = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, prefix[i] + '.png')
        target_label = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood,prefix[i] + '.png')
    
    else:
        test_label_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label/' + 'liver_' + prefix[i] + '_label.png'
        test_img_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img/' + 'liver_' + prefix[i] + '_segmentation_DIY_hist.png'
        if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/',isgood, )) == False:
            os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/',isgood))
        if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/',isgood, )) == False:
            os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/',isgood))
        target_img = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/',isgood, prefix[i] + '.png')
        target_label = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/',isgood,prefix[i] + '.png')
    
    Image.open(test_img_idx).save(target_img)
    Image.open(test_label_idx).save(target_label)

print(k, j)
print(k+j)

print('test img good',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/good')))
print('test label good',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/good')))
print('test img Ungood',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/Ungood')))
print('test label Ungood',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/Ungood')))
print('valid img good',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/good')))
print('valid label good',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/good')))
print('valid img Ungood',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/Ungood')))
print('valid label Ungood',len(os.listdir('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/Ungood')))

for i in range(166, len(prefix)):
    test_label_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label/' + 'liver_' + prefix[i] + '_label.png'
    test_img_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img/' + 'liver_' + prefix[i] + '_segmentation_DIY_hist.png'

    test_label_idx = np.array(Image.open(test_label_idx))
    test_img_idx = np.array(Image.open(test_img_idx))

    #print(np.max(img))
    if np.max(test_label_idx) >= 1:
        isgood = 'Ungood' # 733
        k = k+1
    else:
        isgood = 'good' # 926
        j = j+1
    
    test_label_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_label/' + 'liver_' + prefix[i] + '_label.png'
    test_img_idx = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test_img/' + 'liver_' + prefix[i] + '_segmentation_DIY_hist.png'
    if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, )) == False:
        os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, ))
    if os.path.exists(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood, )) == False:
        os.mkdir(os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood, ))
    target_img = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/img/',isgood, prefix[i] + '.png')
    target_label = os.path.join('/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/test/label/',isgood,prefix[i] + '.png')
    
    Image.open(test_img_idx).save(target_img)
    Image.open(test_label_idx).save(target_label)

print(k, j)
print(k+j)
print(len(test_label))
