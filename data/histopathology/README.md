# FW-RD
## Prerequisites
### Environment
* Python (3.8)
* Numpy (1.20.3)
* Scipy (1.7.1)
* [PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/)
* torchvision (0.2.0)
* PIL (8.3.2)
* scikit-image (0.13.1)
* [OpenSlide 3.4.1](https://openslide.org/)
* matplotlib (2.2.2)
* sklearn (1.0)

### Dataset
The main data are the whole slide images (WSI) in `*.tif` format from the [Camelyon16](https://camelyon17.grand-challenge.org/) challenge. You may also download the dataset at [GigaDB](http://gigadb.org/dataset/100439). There are 400 WSis in total, together about 700GB+. Once you download all the slides, please put all the tumor slides and normal slides for training under one same directory.

The Camelyon16 dataset also provides pixel level annotations of tumor regions for each tumor slide in xml format. You can use them to generate tumor masks for each WSI.

## Preprocess
In order to train WSIs with deep learning models, we need to crop WSIs into 256*256 patches. We provided the patch coordinates that selected by us, so you can directly use it to generate patches.

### Annotations
Generate patches for training and testing use
```shell
python preprocess/patch_gen.py /WSI_TRAIN/ coords/train_good.txt /PATCHES_NORMAL_TRAIN/
python preprocess/patch_gen.py /WSI_TEST/ coords/valid_good.txt /PATCHES_NORMAL_VALID/
python preprocess/patch_gen.py /WSI_TEST/ coords/valid_bad.txt /PATCHES_TUMOR_VALID/
python preprocess/patch_gen.py /WSI_TEST/ coords/test_good.txt /PATCHES_NORMAL_TEST/
python preprocess/patch_gen.py /WSI_TEST/ coords/test_bad.txt /PATCHES_TUMOR_TEST/
```