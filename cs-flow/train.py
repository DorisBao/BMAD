import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import config as c
from scipy.ndimage.filters import gaussian_filter
from model import get_cs_flow_model, save_model, FeatureExtractor, nf_forward
from utils import *
import cv2
import torch.nn.functional as F
from numpy import ndarray
from skimage import measure
import pandas as pd
from statistics import mean
from sklearn.metrics import auc
import sklearn.metrics as metrics

def train(train_loader, valid_loader, test_loader):
    model = get_cs_flow_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    c.pre_extracted = False
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    z_obs = Score_Observer('AUROC')

    for epoch in range(c.meta_epochs):
        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            # for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
            for i, sample in enumerate(train_loader):
                data = sample['image'].cuda()
                optimizer.zero_grad()

                inputs = data.cuda()  # move to device and reshape
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)

                loss = get_loss(z, jac)
                train_loss.append(t2np(loss))

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if c.verbose and epoch == 0 and sub_epoch % 4 == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
        
        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()
        dice_list = []
        aupro_list = []
        gt_list_px = []
        pr_list_px = []

        with torch.no_grad():
            #for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            for i, sample in enumerate(valid_loader):
                #data = sample['image']
                inputs, labels = preprocess_batch(sample)
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)
                loss = get_loss(z, jac)

                z_concat = t2np(concat_maps(z))
                score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
                test_z.append(score)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))

                # z_grouped = list()
                # likelihood_grouped = list()
                # for i in range(len(z)):
                #     z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                #     likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
                # map = likelihood_grouped[0][0]
                # # print(map.unsqueeze(dim=0).unsqueeze(dim=0).shape)
                # # print(sample['image'].shape[2:])
                # map_to_viz = t2np(F.interpolate(map.unsqueeze(dim=0).unsqueeze(dim=0), size=sample['image'].shape[2:], mode='bilinear', align_corners=False))[0][0]
                # map_to_viz = (map_to_viz - min(map_to_viz.flatten())) / (
                # max(map_to_viz.flatten()) - min(map_to_viz.flatten()))
                # map = gaussian_filter(map_to_viz, sigma=4)
                # gt_list_px.extend(sample['mask'].cpu().numpy().astype(int).ravel())
                # #print(mask.cpu().numpy().astype(int).ravel().shape)
                # pr_list_px.extend(map.ravel())

                # if sample['label'].item()!=0:
                #     mask = sample['mask']
                #     dice_list.append(compute_dice(mask.cpu().numpy().astype(int)[0], map.reshape(sample['image'].shape[0],256,256)))
                #     aupro_list.append(compute_pro(mask.cpu().numpy().astype(int)[0], map.reshape(sample['image'].shape[0],256,256)))

        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                     print_score=c.verbose or epoch == c.meta_epochs - 1)

        # auroc_px = round(metrics.roc_auc_score(gt_list_px, pr_list_px), 5)
        # aupro_px = round(np.mean(aupro_list), 5)
        # print('auroc_px: ', auroc_px, ',', 'aupro_px', aupro_px)
        # print("dice_px: ", (round(np.mean(dice_list), 5)))

        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        import time
        t1=time.time()
        test_loss = list()
        test_z = list()
        test_labels = list()
        dice_list = []
        aupro_list = []
        gt_list_px = []
        pr_list_px = []

        with torch.no_grad():
            #for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            for i, sample in enumerate(test_loader):
                #data = sample['image']
                inputs, labels = preprocess_batch(sample)
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)
                loss = get_loss(z, jac)

                z_concat = t2np(concat_maps(z))
                score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
                test_z.append(score)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))

                # z_grouped = list()
                # likelihood_grouped = list()
                # for i in range(len(z)):
                #     z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                #     likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
                # map = likelihood_grouped[0][0]
                # print(map.unsqueeze(dim=0).unsqueeze(dim=0).shape)
                # print(sample['image'].shape[2:])
                # map_to_viz = t2np(F.interpolate(map.unsqueeze(dim=0).unsqueeze(dim=0), size=sample['image'].shape[2:], mode='bilinear', align_corners=False))[0][0]
                # map_to_viz = (map_to_viz - min(map_to_viz.flatten())) / (
                # max(map_to_viz.flatten()) - min(map_to_viz.flatten()))
                # map = gaussian_filter(map_to_viz, sigma=4)
                # gt_list_px.extend(sample['mask'].cpu().numpy().astype(int).ravel())
                #print(mask.cpu().numpy().astype(int).ravel().shape)
                # pr_list_px.extend(map.ravel())

                # if sample['label'].item()!=0:
                #     mask = sample['mask']
                #     dice_list.append(compute_dice(mask.cpu().numpy().astype(int)[0], map.reshape(sample['image'].shape[0],256,256)))
                #     aupro_list.append(compute_pro(mask.cpu().numpy().astype(int)[0], map.reshape(sample['image'].shape[0],256,256)))

        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                     print_score=c.verbose or epoch == c.meta_epochs - 1)
        t2=time.time()
        print(t2-t1, len(test_loader), len(test_loader)/(t2-t1))
        # auroc_px = round(metrics.roc_auc_score(gt_list_px, pr_list_px), 5)
        # aupro_px = round(np.mean(aupro_list), 5)
        # print('auroc_px: ', auroc_px, ',', 'aupro_px', aupro_px)
        # print("dice_px: ", (round(np.mean(dice_list), 5)))

    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)

    return z_obs.max_score, z_obs.last, z_obs.min_loss_score


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def compute_dice(pred, gt):
    dice_score = []
    for i in range(pred.shape[0]):
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        dice = 2 * intersection / union
        dice_score.append(dice)
    return dice_score

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {
        0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr,
                       "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc