import os

RESC_train_normal = '/home/jinan/Datasets/Medical-datasets/RESC/Train/train/good'

RESC_val_abnormal = '/home/jinan/Datasets/Medical-datasets/RESC/Val/val/Ungood'
RESC_val_abnormal_mask = '/home/jinan/Datasets/Medical-datasets/RESC/Val/val_label/Ungood'
RESC_val_normal = '/home/jinan/Datasets/Medical-datasets/RESC/Val/val/good'

RESC_test_abnormal = '/home/jinan/Datasets/Medical-datasets/RESC/Test/test/Ungood'
RESC_test_abnormal_mask = '/home/jinan/Datasets/Medical-datasets/RESC/Test/test_label/Ungood'
RESC_test_normal = '/home/jinan/Datasets/Medical-datasets/RESC/Test/test/good'

OCT2017_train_normal = '/home/jinan/Datasets/Medical-datasets/OCT2017/train/good'

OCT2017_val_abnormal = '/home/jinan/Datasets/Medical-datasets/OCT2017/val/Ungood'
OCT2017_val_normal = '/home/jinan/Datasets/Medical-datasets/OCT2017/val/good'

OCT2017_test_abnormal = '/home/jinan/Datasets/Medical-datasets/OCT2017/test/Ungood'
OCT2017_test_normal = '/home/jinan/Datasets/Medical-datasets/OCT2017/test/good'

def count_imgs(path):
    files = os.listdir(path)
    return len(files)

# Train: 4297, Val: 115 (1086+834), Test: 1805 (1086+834); All: 8137
print("========================RESC===========================")
print('RESC (train, normal):', count_imgs(RESC_train_normal)) # 4297

print('RESC (val, normal):', count_imgs(RESC_val_normal))   # 1086
assert count_imgs(RESC_val_abnormal) == count_imgs(RESC_val_abnormal_mask)
print('RESC (val, abnormal):', count_imgs(RESC_val_abnormal)) # 834

print('RESC (test, normal):', count_imgs(RESC_test_normal))   # 1086
assert count_imgs(RESC_test_abnormal) == count_imgs(RESC_test_abnormal_mask)
print('RESC (test, abnormal):', count_imgs(RESC_test_abnormal)) # 834


#Train: 26315, Valid: 32 (8+24), Test: 968 (242+726); All: 27315
print("========================OCT2017===========================")
print('OCT2017 (train, normal):', count_imgs(OCT2017_train_normal)) # 26315

print('OCT2017 (test, normal):', count_imgs(OCT2017_val_normal))   # 250
print('OCT2017 (test, abnormal):', count_imgs(OCT2017_val_abnormal)) # 750

print('OCT2017 (test, normal):', count_imgs(OCT2017_test_normal))   # 250
print('OCT2017 (test, abnormal):', count_imgs(OCT2017_test_abnormal)) # 750

