# BMAD: Benchmarks for Medical Anomaly Detection
Jinan Bao, Hanshi Sun, Hanqiu Deng, Yinsheng He, Zhaoxiang Zhang, Xingyu Li†

(† Corresponding authors)

Our paper is summitted arVix. 

![](imgs/F1.png)

## Overview
In medical imaging, AD is especially vital for detecting and diagnosing anomalies that may indicate rare diseases or conditions. However, there is a lack of a universal and fair benchmark for evaluating AD methods on medical images, which hinders the development of more generalized and robust AD methods in this specific domain. To bridge this gap, we introduce a comprehensive evaluation benchmark for assessing anomaly detection methods on medical images. This benchmark encompasses six reorganized datasets from five medical domains (i.e. brain MRI, liver CT, retinal OCT, chest X-ray, and digital histopathology) and three key evaluation metrics, and includes a total of fourteen state-of-the-art AD algorithms. This standardized and well-curated medical benchmark with the well-structured codebase enables comprehensive comparisons among recently proposed anomaly detection methods. It will facilitate the community to conduct a fair comparison and advance the field of AD on medical imaging.

## Original Resource Access links

| Dataset        | Download Link                                                |
| -------------- | ------------------------------------------------------------ |
| Brain MRI Anomaly Detection and Localization Benchmark | [BraTS2021 Dataset](http://braintumorsegmentation.org/)                   |
| Liver CT Anomaly Detection and Localization Benchmark | [BTCV Dtaset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217753) + [LiTS Dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)   |
| Retinal OCT Anomaly Detection and Localization Benchmark | [RESC](https://github.com/CharlesKangZhou/P_Net_Anomaly_Detection) + [OCT2017](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| Chest X-ray Anomaly Detection Benchmark                  | [RSNA dataset](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview) |
| Digital Histopathology Anomaly Detection Benchmark       | [Camelyon16 Dataset](https://camelyon17.grand-challenge.org/Data/)                        |

| Support AD algorithm        | Access Link                                                |
| -------------- | ------------------------------------------------------------ |
| Anomalib       | [link](https://github.com/openvinotoolkit/anomalib) |

 
## Our BMAD datasets
To download the reorganization: https://drive.google.com/drive/folders/1La5H_3tqWioPmGN04DM1vdl3rbcBez62?usp=sharing
To download all trained checkpoints: 

## Intoruction for medical domains
## How to use our work
### Train
You can train the model by running `main.py` with args. For example, if you want to train a RD4AD model on RESC dataset, you can run the following command:

```bash
python main.py --mode train --data RESC --model RD4AD
```

### Test
You can test the model by running `main.py` with args. For example, if you want to test a PaDiM model on liver dataset with weight file `results/padim/liver/run/weights/model.ckpt`, you can run the following command:

```bash
python main.py --mode test --data liver --model padim --weight results/padim/liver/run/weights/model.ckpt
```

### Change the hyperparameters
You can change the hyperparameters by modifying the config file in `config/` folder. Take the `cflow` model as an example, you can change the hyperparameters in `config/camelyon_cflow.yaml` file for cflow model on the camelyon dataset.

```yaml
...
coupling_blocks: 8
clamp_alpha: 1.9
fiber_batch_size: 64
lr: 0.0001
...
```
### Thanks
Our orgianl datasets and support alogorithms are come from the above resources, thanks their splendid works!
