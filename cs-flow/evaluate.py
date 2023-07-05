import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from model import load_model, FeatureExtractor
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from copy import deepcopy
import argparse
import yaml
from model import get_cs_flow_model, save_model, FeatureExtractor, nf_forward
from data import TrainDataset, ValidDataset, TestDataset, get_train_transforms, get_valid_transforms
from skimage import measure
import pandas as pd
from statistics import mean
from sklearn.metrics import auc
import sklearn.metrics as metrics

localize = True
upscale_mode = 'bilinear'
score_export_dir = join('./viz/scores/', c.modelname)
os.makedirs(score_export_dir, exist_ok=True)
map_export_dir = join('./viz/maps/', c.modelname)
os.makedirs(map_export_dir, exist_ok=True)


def compare_histogram(scores, classes, thresh=2.5, n_bins=64):
    classes = deepcopy(classes)
    scores = deepcopy(scores)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(r'$-log(p(z))$')
    plt.ylabel('Count (normalized)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(join(score_export_dir, 'score_histogram.png'), bbox_inches='tight', pad_inches=0)


def viz_roc(values, classes, class_names):
    def export_roc(values, classes, export_name='all'):
        # Compute ROC curve and ROC area for each class
        classes = deepcopy(classes)
        classes[classes > 0] = 1
        fpr, tpr, _ = roc_curve(classes, values)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class ' + c.class_name)
        plt.legend(loc="lower right")
        plt.axis('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(join(score_export_dir, export_name + '.png'))

    export_roc(values, classes)
    for cl in range(1, classes.max() + 1):
        filtered_indices = np.concatenate([np.where(classes == 0)[0], np.where(classes == cl)[0]])
        classes_filtered = classes[filtered_indices]
        values_filtered = values[filtered_indices]
        export_roc(values_filtered, classes_filtered, export_name=class_names[filtered_indices[-1]])


def viz_maps(maps, name, label):
    img_path = img_paths[c.viz_sample_count]
    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    map_to_viz = t2np(F.interpolate(maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    if label > 0:
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
        plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
        plt.savefig(join(map_export_dir, name + '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return


def viz_map_array(maps, labels, n_col=8, subsample=4, max_figures=-1):
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)

    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        if labels[i] == 0:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count > 0:
        plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
    return


def evaluate(model, test_loader):
    model.to(c.device)
    model.eval()
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    print('\nCompute maps, loss and scores on test set:')
    anomaly_score = list()
    test_labels = list()
    c.viz_sample_count = 0
    all_maps = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            if not c.pre_extracted:
                inputs = fe(inputs)
            z = model(inputs)

            z_concat = t2np(concat_maps(z))
            nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
            anomaly_score.append(nll_score)
            test_labels.append(t2np(labels))

            if localize:
                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
                all_maps.extend(likelihood_grouped[0])
                for i_l, l in enumerate(t2np(labels)):
                    # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
                    c.viz_sample_count += 1

    anomaly_score = np.concatenate(anomaly_score)
    test_labels = np.concatenate(test_labels)

    compare_histogram(anomaly_score, test_labels)

    class_names = [img_path.split('/')[-2] for img_path in img_paths]
    viz_roc(anomaly_score, test_labels, class_names)

    test_labels = np.array([1 if l > 0 else 0 for l in test_labels])
    auc_score = roc_auc_score(test_labels, anomaly_score)
    print('AUC:', auc_score)

    if localize:
        viz_map_array(all_maps, test_labels)

    return

parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
parser.add_argument('--data', default="camelyon",
                    help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

args = parser.parse_args()

config_path = f"config/{args.data}_csflow.yaml"
print(f"reading config {config_path}...")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
print(config)

train_dataset = TrainDataset(data=args.data, transform=get_train_transforms())
valid_dataset = ValidDataset(data=args.data, transform=get_valid_transforms())
test_dataset = TestDataset(data=args.data, transform=get_valid_transforms())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8)


model = load_model(config['modelname'])
model.to(config['device'])
config['pre_extracted'] = False
if not config['pre_extracted']:
    fe = FeatureExtractor()
    fe.eval()
    fe.to(config['device'])
    for param in fe.parameters():
        param.requires_grad = False
# evaluate
model.eval()
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
        if not config['pre_extracted']:
            inputs = fe(inputs.cuda())

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

test_labels = np.concatenate(test_labels)
is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

anomaly_score = np.concatenate(test_z, axis=0)
print(roc_auc_score(is_anomaly, anomaly_score))
# z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
#                 print_score=c.verbose or epoch == c.meta_epochs - 1)

# auroc_px = round(metrics.roc_auc_score(gt_list_px, pr_list_px), 5)
# aupro_px = round(np.mean(aupro_list), 5)
# print('auroc_px: ', auroc_px, ',', 'aupro_px', aupro_px)
# print("dice_px: ", (round(np.mean(dice_list), 5)))

# evaluate
model.eval()
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
        if not config['pre_extracted']:
            inputs = fe(inputs.cuda())

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
# if c.verbose:
#     print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

test_labels = np.concatenate(test_labels)
is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

anomaly_score = np.concatenate(test_z, axis=0)
print(roc_auc_score(is_anomaly, anomaly_score))
# z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
#                 print_score=config['verbose'] or epoch == config['meta_epochs'] - 1)
t2=time.time()
# print(t2-t1, len(test_loader), len(test_loader)/(t2-t1))
# auroc_px = round(metrics.roc_auc_score(gt_list_px, pr_list_px), 5)
# aupro_px = round(np.mean(aupro_list), 5)
# print('auroc_px: ', auroc_px, ',', 'aupro_px', aupro_px)
# print("dice_px: ", (round(np.mean(dice_list), 5)))
