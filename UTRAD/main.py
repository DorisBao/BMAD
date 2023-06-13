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
from models import Create_nets, print_network
from datasets import Get_dataloader
from options import TrainOptions
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label
from collections.abc import Sequence
import pandas as pd
from statistics import mean
from sklearn.metrics import auc
import sklearn.metrics as metrics
from skimage import measure
def compute_dice(pred, gt):
    dice_score = []
    for i in range(pred.shape[0]):
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        dice = 2 * intersection / union
        dice_score.append(dice)
    return dice_score

def compute_pro(masks, amaps, num_th: int = 200) -> None:

    # assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    # assert isinstance(masks, ndarray), "type(masks) must be ndarray"
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
def get_shape(lst, shapelst=()):
    if not isinstance(lst, Sequence):
        return shapelst
    if isinstance(lst[0], Sequence):
        innerlen = len(lst[0])
        if not all(len(item) == innerlen for item in lst):
            raise ValueError("All lists dont have same length")
    shapelst += (len(lst), )
    shapelst = get_shape(lst[0], shapelst)
    return shapelst
def main():
    args = TrainOptions().parse()
    with open("./%s-%s/args.log" % (args.exp_name,  args.dataset_name) ,"a") as args_log:
        for k, v in sorted(vars(args).items()):
            print('%s: %s ' % (str(k), str(v)))
            args_log.write('%s: %s \n' % (str(k), str(v)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir = '%s-%s/%s/%s' % (args.exp_name, args.dataset_name, args.model_result_dir, 'checkpoint.pth')
    start_epoch = 0
    transformer = Create_nets(args)
    transformer = transformer.to(device)
    transformer.cuda()

    optimizer = torch.optim.Adam( transformer.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    best_loss = 1e10

    backbone = models.resnet18(pretrained=True).to(device)

    if os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        transformer.load_state_dict(checkpoint['transformer'])
        start_epoch = checkpoint['start_epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        del checkpoint

    backbone.eval()
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)
    #backbone.layer4[-1].register_forward_hook(hook)
    layer = 3

    criterion = nn.MSELoss()

    train_dataset = TrainDataset(data=args.dataset_name, transform=get_train_transforms())
    valid_dataset = ValidDataset(data=args.dataset_name, transform=get_valid_transforms())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=8)

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

    flag = 0
    torch.set_printoptions(profile="full")

    for epoch in range(0, 150):
        print("start evaluation on test set!")
        transformer.eval()
        score_map = []
        gt_list = []
        gt_mask_list = []
        dice_list = []
        aupro_list = []
        for i, sample in enumerate(valid_dataloader):
            with torch.no_grad():
                img = sample['image'].cuda()
                ground_truth = sample['mask'].cuda()
                gt = sample['label'].cuda()

                outputs = []
                _ = backbone(img)
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
                # print(outputs.shape)# torch.Size([2, 448, 128, 64])
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
                    patch_score = F.interpolate(patch_score, (width,height), mode='bilinear', align_corners=False)
                    patch_normed_score.append(patch_score)
                score = torch.zeros(batch_size,1,64,64).to(device)
                for j in range(4):
                    score = embedding_concat(score, patch_normed_score[j])
                
                score = F.conv2d(input=score, 
                        weight=torch.tensor([[[[0.0]],[[0.25]],[[0.25]],[[0.25]],[[0.25]]]]).to(device), 
                        bias=None, stride=1, padding=0, dilation=1)

                score = F.interpolate(score, (ground_truth.size(2),ground_truth.size(3)), mode='bilinear', align_corners=False)
                heatmap = score.repeat(1,3,1,1)
                score_map.append(score.cpu())
                gt_mask_list.append(ground_truth.cpu())
                gt_list.append(gt)
                if args.dataset_name == 'OCT2017' or args.dataset_name == "chest" or args.dataset_name == 'camelyon':
                    continue
                if gt.item()!=0:
                    #print(mask.cpu().numpy().astype(int)[0].shape, anomaly_map.reshape(sample['image'].shape[0],256,256).shape)
                    aupro_list.append(compute_pro(ground_truth.cpu().numpy().astype(int)[0], score.cpu().reshape(sample['image'].shape[0],256,256)))
                dice_list.append(compute_dice(ground_truth.cpu().numpy().astype(int)[0], score.cpu().numpy().reshape(sample['image'].shape[0],256,256)))

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
        # print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        # aupro_px = round(np.mean(aupro_list), 5)
        # print('aupro_px', aupro_px)
        # print("dice_px: ", (round(np.mean(dice_list), 5)))
        avg_loss = 0
        avg_loss_scale = 0
        total = 0
        transformer.train()
        for i, sample in enumerate(train_dataloader):
            img = sample['image'].cuda()
            #inputs = batch.to(device) # [2, 3, 256, 256]
            outputs = []
            #print(len(outputs))
            optimizer.zero_grad()
            with torch.no_grad():
                _ = backbone(img)#change output
                #outputs = outputs[layer-1]
                #outputs = embedding_concat(embedding_concat(embedding_concat(inputs,outputs[0]),outputs[1]),outputs[2])
                #print(embedding_concat(outputs[0],outputs[1]).shape)
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
                #print(outputs.shape) # torch.size([2, 448, 64, 64])

            recon, std = transformer(outputs)
            torch.cuda.empty_cache()

            loss = criterion(recon, outputs)
            loss_scale = criterion(std, torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).detach())

            (loss+loss_scale).backward()

            optimizer.step()
            torch.cuda.empty_cache()

            avg_loss += loss * img.size(0)
            avg_loss_scale += loss_scale * img.size(0)
            total += img.size(0)
            print(("\r[Epoch%d/%d]-[Batch%d/%d]-[Loss:%f]-[Loss_scale:%f]" %
                                                            (epoch+1, args.epoch_num,
                                                            i, len(train_dataloader),
                                                            avg_loss / total,
                                                            avg_loss_scale / total)))

        # with open("./%s-%s/args.log" % (args.exp_name,  args.dataset_name) ,"a") as train_log:
        #     train_log.write("\r[Epoch%d]-[Loss:%f]-[Loss_scale:%f]" %
        #                                                     (epoch+1, avg_loss / total, avg_loss_scale / total))
        loss = 0
        if best_loss > avg_loss and best_loss > loss:
            best_loss = avg_loss
            state_dict = {
                        'start_epoch':epoch,
                        #'optimizer':optimizer.state_dict(),
                        'transformer':transformer.state_dict(),
                        'args':args,
                        'best_loss':best_loss
                }
            torch.save(state_dict, save_dir)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()