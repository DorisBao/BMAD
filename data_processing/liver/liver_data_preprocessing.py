import os
import nibabel as nib
import numpy as np
import cv2

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import cv2
import matplotlib.image as img
import matplotlib

def get_img_labels(files):
    label_list = []
    root_dir = './data/liver_dataset'
    for img_name in files:
        gt_name = img_name.replace('liver', 'liver_gt')
        gt_path = os.path.join(root_dir, 'image_label', gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.sum() > 0:
            label = 1
        else:
            label = 0
            
        label_list.append(label)
        
    return label_list

def hist_equalization(img_array):

	#flatten image array and calculate histogram via binning
	histogram_array = np.bincount(img_array.flatten(), minlength=256)
 
	histogram_array[0] = 0
	histogram_array = histogram_array/np.linalg.norm(histogram_array)

	#normalize
	num_pixels = np.sum(histogram_array)
	histogram_array = histogram_array/num_pixels

	#normalized cumulative histogram
	chistogram_array = np.cumsum(histogram_array)


	"""
	STEP 2: Pixel mapping lookup table
	"""
	transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
 
	for i in range(len(transform_map)):
		if transform_map[i] > 0 and transform_map[i] < 150:

			transform_map[i] = np.floor((transform_map[i]/150*120 + 30))  

	"""
	STEP 3: Transformation
	"""
	# flatten image array into 1D list
	img_list = list(img_array.flatten())

	# transform pixel values to equalize
	eq_img_list = [transform_map[p] for p in img_list]

	# reshape and write back into img_array
	eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
 
	return eq_img_array

# def load_Altas(dir_path, save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     filenames = os.listdir(dir_path)
#     filenames.sort()
#     # delete the element 'DS_Store'
#     for j, x in enumerate(filenames):
#         if x.startswith('.'):
#             filenames.pop(j)
        
#     for file in filenames:
#         file_path = os.path.join(dir_path, file)
#         img = nib.load(file_path)          # read nii
#         img_fdata = img.get_fdata()
#         fname = file.replace('.nii.gz', '')
        
#         # img_fdata = np.clip(img_fdata, -200, 250)
        
#         number = int(fname.replace('img', ''))
#         number_string = str(number).zfill(4)
#         label_name = file.replace('img', 'label')
#         label_path = os.path.join(dir_path.replace('img', 'label'), label_name)
#         label_data = nib.load(label_path)
#         label_fdata = label_data.get_fdata()
            
#         (x,y,z) = img.shape
#         for i in range(z):
#             gray = img_fdata[:,:,i]
#             label = label_fdata[:,:,i]
            
#             # liver mask
#             liver_mask = np.where(label == 6, 1, 0)
                        
#             if liver_mask.sum() == 0:
#                 continue
            
#             # gray = (gray + 200)/(450/255)
#             gray_liver = gray * liver_mask
#             flipped_gray = cv2.flip(gray_liver, 0)
            
#             # Histogram equalization
#             hist_gray = hist_equalization(flipped_gray.astype(np.uint8))
#             cv2.imwrite(os.path.join(save_dir, 'img_{}_{}_hist'.format(number_string, i) + '.png'), hist_gray)
            
#         print(number, '    done')


def load_Altas(dir_path, save_dir):
    save_raw_dir = os.path.join(dir_path, 'raw')
    if not os.path.exists(save_raw_dir):
        os.makedirs(save_raw_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = os.listdir(dir_path)
    filenames.sort()
    # delete the element 'DS_Store'
    for j, x in enumerate(filenames):
        if x.startswith('.'):
            filenames.pop(j)
        
    for file in filenames:
        file_path = os.path.join(dir_path, file)
        img = nib.load(file_path)          # read nii
        img_fdata = img.get_fdata()
        fname = file.replace('.nii.gz', '')
        
        # img_fdata = np.clip(img_fdata, -200, 250)
        
        number = int(fname.replace('img', ''))
        number_string = str(number).zfill(4)
        label_name = file.replace('img', 'label')
        label_path = os.path.join(dir_path.replace('img', 'label'), label_name)
        label_data = nib.load(label_path)
        label_fdata = label_data.get_fdata()
            
        (x,y,z) = img.shape
        for i in range(z):
            gray = img_fdata[:,:,i]
            label = label_fdata[:,:,i]
            
            # liver mask
            liver_mask = np.where(label == 6, 1, 0)
                        
            if liver_mask.sum() == 0:
                continue
            
            # gray = (gray + 200)/(450/255)
            save_raw_path = os.path.join(save_raw_dir, 'liver_{}_{}'.format(number_string, i) + '.png')
            matplotlib.image.imsave(save_raw_path, gray, cmap='gray')
            
            # Apply liver mask and flipping
            img = cv2.imread(save_raw_path, cv2.IMREAD_GRAYSCALE)
            gray_liver = img * liver_mask
            flipped_gray = cv2.flip(gray_liver, 0)
            
            # Histogram equalization
            
            hist_img = hist_equalization(flipped_gray.astype(np.uint8))
            
            cv2.imwrite(os.path.join(save_dir, 'img_{}_{}_hist'.format(number_string, i) + '.png'), hist_img)
            
        print(number, '    done')        
      
def load_LiTs(dir_path, save_dir):
    save_raw_dir = os.path.join(dir_path, 'raw')
    if not os.path.exists(save_raw_dir):
        os.makedirs(save_raw_dir)

    
    save_img_dir = os.path.join(save_dir, 'image')
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    save_gt_dir = os.path.join(save_dir, 'image_label')
    if not os.path.exists(save_gt_dir):
        os.makedirs(save_gt_dir)

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".nii"):
                file_path = os.path.join(root, file)

                if 'volume' in file:
                    img = nib.load(file_path)          # read nii
                    img_fdata = img.get_fdata()
                    
                    folder_path = os.path.dirname(file_path)
                    fname = os.path.basename(file_path).replace('.nii', '')

                    img_fdata = np.clip(img_fdata, -200, 250)
                    
                    number = int(fname.replace('volume-', ''))
                    label_name = file.replace('volume-', 'segmentation-')
                    label_path = os.path.join(folder_path, label_name)
                    label_data = nib.load(label_path)
                    label_fdata = label_data.get_fdata()
                else:
                    continue
                    
                (x,y,z) = img.shape
                for i in range(z):
                    gray = img_fdata[:,:,i]
                    label = label_fdata[:,:,i]
                    
                    # liver mask
                    liver_mask = np.where(label > 0, 1, 0)
                    lesion_mask = np.where(label == 2, 1, 0)
                        
                    if liver_mask.sum() == 0:
                        continue
                    # gray = (gray + 200)/(450/255)
                    
                    # # Retrieve the liver region
                    # gray_liver = gray * liver_mask
                    
                    # gray_liver_hist = hist_equalization(gray_liver.astype(np.uint8))
                    
                    save_raw_path = os.path.join(save_raw_dir, 'liver_{}_{}'.format(number, i) + '.png')
                    matplotlib.image.imsave(save_raw_path, gray, cmap='gray')
                    
                    # Apply liver mask and flipping
                    img = cv2.imread(save_raw_path, cv2.IMREAD_GRAYSCALE)
                    gray_liver = img * liver_mask
                    flipped_gray = cv2.flip(gray_liver, 0)
                    flipped_lesion = cv2.flip(lesion_mask, 0)
                    
                    # Histogram equalization
                    
                    hist_img = hist_equalization(flipped_gray.astype(np.uint8))
                            
                    cv2.imwrite(os.path.join(save_img_dir, 'liver_{}_{}'.format(number, i) + '.png'), hist_img)
                    cv2.imwrite(os.path.join(save_gt_dir, 'liver_gt_{}_{}'.format(number, i) + '.png'), flipped_lesion*255)
                    
                print(number, '    done')
            
def cross_fold_split(root_dir):
    image_dir = os.path.join(root_dir, 'image')
    files = os.listdir(image_dir)
    files.sort()
    files_np = np.array(files)
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(files)

    for i, (train_index, test_index) in enumerate(kf.split(files)):
        # print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = files_np[train_index], files_np[test_index]
        
        train_labels = get_img_labels(X_train)
        test_labels = get_img_labels(X_test)
        
        data = {'Train_images': X_train,
                'Train_labels': train_labels,
                'Test_images': X_test,
                'Test_labels': test_labels}
        
        with open(os.path.join(root_dir,'fold_{}.npy'.format(i)), 'wb') as f:
            np.save(f, data)
        
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train', 'test'], action='store')
    args = parser.parse_args()
    
    base_dir = './data'
    join = os.path.join
    
    if args.phase == 'train':
        source_dir = join(base_dir, 'Altas/Training/img')
        save_dir = join(base_dir, 'liver_dataset/train/good')
        load_Altas(source_dir, save_dir)
        # load_Altas_test(source_dir, save_dir)
    else:    
        source_dir = join(base_dir, 'LiTs')
        save_dir = join(base_dir, 'liver_dataset')
        load_LiTs(source_dir, save_dir)
        
        cross_fold_split('./data/liver_dataset')
        
        
        
    
   