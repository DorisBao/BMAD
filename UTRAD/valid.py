import argparse
import os
import numpy as np
import time
import datetime
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
from statistics import mean
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib
import wandb
import pandas as pd
from sklearn.metrics import auc
from skimage import measure
from scipy.ndimage.filters import gaussian_filter
from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms, TestDataset
import yaml

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def compute_dice(pred, gt):
    dice_score = []
    for i in range(pred.shape[0]):
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        dice = 2 * intersection / union
        dice_score.append(dice)
    return dice_score

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def compute_pro(masks, amaps, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = TrainOptions().parse()
    config_path = f"config/{args.dataset_name}_UTRAD.yaml"
    print(f"reading config {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    torch.manual_seed(config['seed'])
    if args.weight == "None" or args.weight == None:
        save_dir = '%s-%s/%s/%s' % (config['exp_name'], args.dataset_name, config['model_result_dir'], 'checkpoint.pth')
    else:
        save_dir = args.weight
    
    print(f"Loading ckpt {save_dir}...")
    
    start_epoch = 0
    transformer = Create_nets(args)
    transformer = transformer.to(device)
    transformer.cuda()
    '''
    if os.path.exists('resnet18_pretrained.pth'):
        backbone = models.resnet18(pretrained=False).to(device)
        backbone.load_state_dict(torch.load('resnet18_pretrained.pth'))
    else:
        backbone = models.resnet18(pretrained=True).to(device)
    '''

    backbone = models.resnet18(pretrained=True).to(device)

    if os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        transformer.load_state_dict(checkpoint['transformer'])
        start_epoch = checkpoint['start_epoch']

    backbone.eval()
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)
    #backbone.layer4[-1].register_forward_hook(hook)
    layer = 3

    valid_dataset = ValidDataset(data=args.dataset_name, transform=get_valid_transforms())
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=8)
    test_dataset = TestDataset(data=args.dataset_name, transform=get_valid_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8)


    def embedding_concat(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z


    img_save_dir='%s-%s/%s' % (config['exp_name'], args.dataset_name, config['validation_image_dir'])
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    score_map = []
    gt_list = []
    gt_mask_list = []
    dice_list = []
    aupro_list = []
    if True:
        transformer.eval()
        #for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader)
        for i, sample in enumerate(valid_dataloader):
            with torch.no_grad():
                img = sample['image'].cuda()
                ground_truth = sample['mask'].cuda()
                gt = sample['label'].cuda()
                
                num = 4
                norm_range = [(0.9,1.3),(0.9,1.3),(0.9,1.3),(0.9,1.3),(1.1,1.5),]
                outputs = []
                _ = backbone(img)
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
                recon, std = transformer(outputs)
                batch_size, channels, width, height = recon.size()

                dist = torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).div(std.abs())
                dist = dist.view(batch_size, 1, width, height)

                patch_normed_score = []
                for j in range(4):
                    patch_size = pow(4, j)
                    patch_score = F.conv2d(input=dist, 
                        weight=(torch.ones(1,1,patch_size,patch_size) / (patch_size*patch_size)).to(device), 
                        bias=None, stride=patch_size, padding=0, dilation=1)
                    patch_score = F.avg_pool2d(dist,patch_size,patch_size)
                    patch_score = F.interpolate(patch_score, (width,height), mode='bilinear')
                    patch_normed_score.append(patch_score)
                score = torch.zeros(batch_size,1,64,64).to(device)
                for j in range(4):
                    score = embedding_concat(score, patch_normed_score[j])
                
                score = F.conv2d(input=score, 
                        weight=torch.tensor([[[[0.0]],[[0.25]],[[0.25]],[[0.25]],[[0.25]]]]).to(device), 
                        bias=None, stride=1, padding=0, dilation=1)

                score = F.interpolate(score, (ground_truth.size(2),ground_truth.size(3)), mode='bilinear')
                #heatmap = score.repeat(1,3,1,1)
                score_map.append(score.cpu())
                gt_mask_list.append(ground_truth.cpu())
                gt_list.append(gt)
                # if gt.item()!=0:
                #     #print(mask.cpu().numpy().astype(int)[0].shape, anomaly_map.reshape(sample['image'].shape[0],256,256).shape)
                #     aupro_list.append(compute_pro(ground_truth.cpu().numpy().astype(int)[0], score.cpu().reshape(sample['image'].shape[0],256,256)))
                #     dice_list.append(compute_dice(ground_truth.cpu().numpy().astype(int)[0], score.cpu().numpy().reshape(sample['image'].shape[0],256,256)))
                # if gt.item() != 0: 
                #         aupro_list.append(compute_pro(ground_truth.squeeze(0).cpu().numpy().astype(int), score[0].cpu()))
    
    score_map = torch.cat(score_map,dim=0)
    gt_mask_list = torch.cat(gt_mask_list,dim=0)
    gt_list = torch.cat(gt_list,dim=0)
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    
    # calculate image-level ROC AUC score
    img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
    gt_list = gt_list.cpu().numpy()
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(len(gt_list))
    print('image ROCAUC: %.5f' % (img_roc_auc))

    # # calculate per-pixel level ROCAUC
    # gt_mask = gt_mask_list.numpy().astype('int')
    # scores = scores.numpy().astype('float32')
    # #fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten()) 
    # per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten()) 
    # print('pixel ROCAUC: %.5f' % (per_pixel_rocauc))

    # aupro_px = round(np.mean(aupro_list), 5)
    # print('aupro_px', aupro_px)
    # print("dice_px: ", (round(np.mean(dice_list), 5)))


    img_save_dir='%s-%s/%s' % (config['exp_name'], args.dataset_name, config['validation_image_dir'])
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    score_map = []
    gt_list = []
    gt_mask_list = []
    dice_list = []
    aupro_list = []
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=args.dataset_name,
    #     name = 'UTRAD',
    # )
    if True:
        transformer.eval()
        from tqdm import tqdm
        #for i,(name ,batch, ground_truth, gt) in enumerate(tqdm(test_dataloader))
        for i, sample in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                img = sample['image'].cuda()
                ground_truth = sample['mask'].cuda()
                gt = sample['label'].cuda()
                
                num = 4
                norm_range = [(0.9,1.3),(0.9,1.3),(0.9,1.3),(0.9,1.3),(1.1,1.5),]
                outputs = []
                _ = backbone(img)
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
                recon, std = transformer(outputs)
                batch_size, channels, width, height = recon.size()

                dist = torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).div(std.abs())
                dist = dist.view(batch_size, 1, width, height)

                patch_normed_score = []
                for j in range(4):
                    patch_size = pow(4, j)
                    patch_score = F.conv2d(input=dist, 
                        weight=(torch.ones(1,1,patch_size,patch_size) / (patch_size*patch_size)).to(device), 
                        bias=None, stride=patch_size, padding=0, dilation=1)
                    patch_score = F.avg_pool2d(dist,patch_size,patch_size)
                    patch_score = F.interpolate(patch_score, (width,height), mode='bilinear')
                    patch_normed_score.append(patch_score)
                score = torch.zeros(batch_size,1,64,64).to(device)
                for j in range(4):
                    score = embedding_concat(score, patch_normed_score[j])
                
                score = F.conv2d(input=score, 
                        weight=torch.tensor([[[[0.0]],[[0.25]],[[0.25]],[[0.25]],[[0.25]]]]).to(device), 
                        bias=None, stride=1, padding=0, dilation=1)

                score = F.interpolate(score, (ground_truth.size(2),ground_truth.size(3)), mode='bilinear')
                heatmap = score.repeat(1,3,1,1)
                score_map.append(score.cpu())
                gt_mask_list.append(ground_truth.cpu())
                gt_list.append(gt)
                # if gt.item()!=0:
                #     #print(mask.cpu().numpy().astype(int)[0].shape, anomaly_map.reshape(sample['image'].shape[0],256,256).shape)
                #     aupro_list.append(compute_pro(ground_truth.cpu().numpy().astype(int)[0], score.cpu().reshape(sample['image'].shape[0],256,256)))
                #     dice_list.append(compute_dice(ground_truth.cpu().numpy().astype(int)[0], score.cpu().numpy().reshape(sample['image'].shape[0],256,256)))
                # if gt.item() != 0: 
                #         aupro_list.append(compute_pro(ground_truth.squeeze(0).cpu().numpy().astype(int), score[0].cpu()))
                # '''
                # visualization using wandb
                # '''
                # anomaly_map = gaussian_filter(score.cpu(), sigma=4)
                # anomaly_map = min_max_norm(anomaly_map)  # 0~1 mapping
                # ano_map = (anomaly_map*255).astype(np.uint8)
                # ano_map = cvt2heatmap(ano_map[0][0])
                # path = sample['path']
                # image = cv2.imread(path[0])
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                # img = np.uint8(min_max_norm((img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8))*255)
                # ano_map = cv2.cvtColor(ano_map, cv2.COLOR_BGR2RGB)
                # ano_map = cv2.addWeighted(ano_map, 0.4, image, (1 - 0.4), 0)
                
                # wandb.log({
                #     "img | gt | heatmap | pred": [wandb.Image(image), wandb.Image(ground_truth), wandb.Image(ano_map), wandb.Image(anomaly_map[0])]
                # })

    score_map = torch.cat(score_map,dim=0)
    gt_mask_list = torch.cat(gt_mask_list,dim=0)
    gt_list = torch.cat(gt_list,dim=0)
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    
    # calculate image-level ROC AUC score
    img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
    gt_list = gt_list.cpu().numpy()
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(len(gt_list))
    print('image ROCAUC: %.5f' % (img_roc_auc))

    # # calculate per-pixel level ROCAUC
    # gt_mask = gt_mask_list.numpy().astype('int')
    # scores = scores.numpy().astype('float32')
    # #fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten()) 
    # per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten()) 
    # print('pixel ROCAUC: %.5f' % (per_pixel_rocauc))

    # aupro_px = round(np.mean(aupro_list), 5)
    # print('aupro_px', aupro_px)
    # print("dice_px: ", (round(np.mean(dice_list), 5)))

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
