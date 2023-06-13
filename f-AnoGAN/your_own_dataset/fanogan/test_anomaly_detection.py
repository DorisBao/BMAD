import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm
from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
def test_anomaly_detection(opt, generator, discriminator, encoder,
                           dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator_camelyon.pth"))
    discriminator.load_state_dict(torch.load("results/discriminator_camelyon.pth"))
    encoder.load_state_dict(torch.load("results/encoder_camelyon.pth"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss()

    labels = []
    img_distances = []
    anomaly_scores = []
    z_distances = []
    for i, sample in enumerate(dataloader):
        img = sample['image'].cuda()
        label = sample['label']

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature

        z_distance = criterion(fake_z, real_z)
        #print(label.tolist(), img_distance.tolist(), anomaly_score.tolist())
        labels.append(label.cpu().numpy().tolist()[0])
        img_distances.append(img_distance.tolist())
        anomaly_scores.append(anomaly_score.tolist())
        z_distances.append(z_distance.tolist())
    
    fpr, tpr, thresholds = roc_curve(labels, img_distances, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    print('roc_auc (img_distance): ', roc_auc)

    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    print('roc_auc (anomaly_score): ', roc_auc)

    fpr, tpr, thresholds = roc_curve(labels, z_distances, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    print('roc_auc (z_distance): ', roc_auc)