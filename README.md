# BMAD: Benchmarks for Medical Anomaly Detection
Jinan Bao, Hanshi Sun, Hanqiu Deng, Yinsheng He, Zhaoxiang Zhang, Xingyu Li†

Our paper can be accessed at [arvix](https://arxiv.org/abs/2306.11876) .

![](imgs/BMAD.png)

## Overview
In medical imaging, AD is especially vital for detecting and diagnosing anomalies that may indicate rare diseases or conditions. However, there is a lack of a universal and fair benchmark for evaluating AD methods on medical images, which hinders the development of more generalized and robust AD methods in this specific domain. To bridge this gap, we introduce a comprehensive evaluation benchmark for assessing anomaly detection methods on medical images. This benchmark encompasses six reorganized datasets from five medical domains (i.e. brain MRI, liver CT, retinal OCT, chest X-ray, and digital histopathology) and three key evaluation metrics, and includes a total of fourteen state-of-the-art AD algorithms. This standardized and well-curated medical benchmark with the well-structured codebase enables comprehensive comparisons among recently proposed anomaly detection methods. It will facilitate the community to conduct a fair comparison and advance the field of AD on medical imaging.
      
## BMAD
### Our datasets

BMAD includes six medical datasets from five different domains for medical anomaly detection as summerized in Table 1. Within these datasets, three supports pixel-level evaluation of anomaly detection, while the remaining three is for sample-level assessment only. Note, the validation set is specifically designed for model hyper-parameter tuning and training strategy selection, while the test set should be remained untouched until the final evaluation stage. One can download the our datasets from [Google drive](https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing
) 
![](imgs/Table2.png)

![](imgs/whole-vision.png)


Take the Histopathology dataset (anomaly detection) as an example, the structure is as follows:

```text
camelyon16
├── train
    ├── good
        ├── 1000.png
        ├── 1001.png
        ├── ...
├── valid
    ├── good
        ├── 1080.png
        ├── 1081.png
        ├── ...
    ├── Ungood
        ├── 1000.png
        ├── 1001.png
        ├── ...
    
├── test
    ├── good
        ├── 1000.png
        ├── 1001.png
        ├── ...
    ├── Ungood
        ├── 100.png
        ├── 101.png
        ├── ...
```

Take the Brain dataset (anomaly detection and localization) as an example, the structure is as follows:

```text
Brain
├── train
    ├── good
        ├── img
            ├── 00003_60.png
            ├── 00003_61.png
            ├── ...
├── valid
    ├── good
        ├── img
            ├── 00025_99.png
            ├── 00100_60.png
            ├── ...
    ├── Ungood
        ├── img
            ├── 00124_60.png
            ├── 00124_70.png
            ├── ...
        ├── anomaly_mask
            ├── 00124_60.png
            ├── 00124_70.png
            ├── ...
├── test
    ├── good
        ├── img
            ├── 00000_96.png
            ├── 00000_97.png
            ├── ...
    ├── Ungood
        ├── img
            ├── 00002_60.png
            ├── 00002_68.png
            ├── ...
        ├── anomaly_mask
            ├── 00002_60.png
            ├── 00002_68.png
            ├── ...
```
### Our codebase
We support 14 algorithms in our BMAD and the results are shown in Table 2. Our trained checkpoints can be downloaded from [Google drive](https://drive.google.com/drive/folders/105s6IzMO-Y5P_a_YkQ_dzL0YgbmGPhd8?usp=sharing) 

![](imgs/new-table.JPG)

#### Train
You can train the model by running `main.py` with args. For example, if you want to train a RD4AD model on RESC dataset, you can run the following command:

```bash
python main.py --mode train --data RESC --model RD4AD
```
#### If change the hyperparameters
You can change the hyperparameters by modifying the config file in `config/` folder. Take the `cflow` model as an example, you can change the hyperparameters in `config/camelyon_cflow.yaml` file for cflow model on the camelyon dataset.

```yaml
...
coupling_blocks: 8
clamp_alpha: 1.9
fiber_batch_size: 64
lr: 0.0001
...
```
#### Test
You can test the model by running `main.py` with args. For example, if you want to test a PaDiM model on liver dataset with weight file `results/padim/liver/run/weights/model.ckpt`, you can run the following command:

```bash
python main.py --mode test --data liver --model padim --weight results/padim/liver/run/weights/model.ckpt
```
## References 

### Original resource access links
Brain MRI Anomaly Detection and Localization Benchmark : [BraTS2021 Dataset](http://braintumorsegmentation.org/)  

Liver CT Anomaly Detection and Localization Benchmark :  [BTCV Dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217753) + [LiTS Dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)    

Retinal OCT Anomaly Detection and Localization Benchmark : [RESC Dataset](https://github.com/CharlesKangZhou/P_Net_Anomaly_Detection) + [OCT2017 Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) 

Chest X-ray Anomaly Detection Benchmark : [RSNA Dataset](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview)  

Digital Histopathology Anomaly Detection Benchmark : [Camelyon16 Dataset](https://camelyon17.grand-challenge.org/Data/)      

### Support alogorithms
[RD4AD](https://arxiv.org/abs/2201.10703),[PatchCore](https://arxiv.org/abs/2106.08265),[DRAEM](https://arxiv.org/abs/2108.07610),[DeepSVDD](https://proceedings.mlr.press/v80/ruff18a.html),[MKD](https://arxiv.org/abs/2011.11108),[PaDIM](https://arxiv.org/abs/2011.08785),[CFLOW](https://arxiv.org/abs/2107.12571),[CS-Flow](https://arxiv.org/abs/2110.02855),[CutPaste](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf) [GANomaly](https://arxiv.org/abs/1805.06725),[UTRAD](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004810),[STFPM](https://arxiv.org/abs/2111.15376),[f-AnoGAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640),[CFA](https://arxiv.org/abs/2206.04325)

## Thanks
Our orgianl resources and support alogorithms are come from the above references, thanks their splendid works!

