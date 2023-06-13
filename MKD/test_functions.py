from torch import nn
from sklearn.metrics import roc_curve, auc
from utils import morphological_process, convert_to_grayscale, max_regarding_to_abs
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch
from torch.autograd import Variable
from copy import deepcopy
from torch.nn import ReLU
import wandb
from tqdm import tqdm
import cv2
from numpy import ndarray
from skimage import measure
import pandas as pd
from statistics import mean
from sklearn.metrics import auc
import sklearn.metrics as metrics
def detection_test(model, vgg, test_dataloader, config):
    normal_class = config["normal_class"]
    lamda = config['lamda']
    dataset_name = config['dataset_name']
    direction_only = config['direction_loss_only']

    similarity_loss = torch.nn.CosineSimilarity()
    label_score = []
    model.eval()
    for i, sample in enumerate(test_dataloader):
        img = sample['image'].cuda()
        mask = sample['mask'].cuda()
        Y = sample['label'].cuda()
        X = Variable(img).cuda()
        output_pred = model.forward(X)
        output_real = vgg(X)
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

        if direction_only:
            loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
            loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
            loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
            total_loss = loss_1 + loss_2 + loss_3
        else:
            abs_loss_1 = torch.mean((y_pred_1 - y_1) ** 2, dim=(1, 2, 3))
            loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
            abs_loss_2 = torch.mean((y_pred_2 - y_2) ** 2, dim=(1, 2, 3))
            loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
            abs_loss_3 = torch.mean((y_pred_3 - y_3) ** 2, dim=(1, 2, 3))
            loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
            total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)

        label_score += list(zip(Y.cpu().data.numpy().tolist(), total_loss.cpu().data.numpy().tolist()))
    
    target_class = normal_class
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    indx1 = labels == target_class
    indx2 = labels != target_class
    #print(indx1)
    #print(indx2)
    labels[indx1] = 1
    labels[indx2] = 0
    scores = np.array(scores)
    #print(labels.shape)
    #print(scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    return roc_auc


def localization_test(model, vgg, test_dataloader, config):
    localization_method = config['localization_method']
    if localization_method == 'gradients':
        grad = gradients_localization(model, vgg, test_dataloader, config)
    if localization_method == 'smooth_grad':
        grad = smooth_grad_localization(model, vgg, test_dataloader, config)
    if localization_method == 'gbp':
        new_gbp_localization(model, vgg, test_dataloader, config)

def new_gbp_localization(model, vgg, test_dataloader, config):
    gt_list_px = []
    pr_list_px = []
    aupro_list = []
    dice_list = []
    tpr = []
    fpr = []

    model.eval()
    print("GBP Method:")

    grad1 = None
    for _, sample in enumerate(tqdm(test_dataloader)):
        img = sample['image'].cuda()
        mask = sample['mask'].cuda()
        #print(mask.shape)
        Y = sample['label'].cuda()
        X = Variable(img).cuda()
        grad1 = np.zeros((X.shape[0], 1, 224, 224), dtype=np.float32)
        data = X.view(1, 3, 224, 224)
        GBP = GuidedBackprop(model, vgg, 'cuda:0')
        gbp_saliency = abs(GBP.generate_gradients(data, config))
        gbp_saliency = (gbp_saliency - min(gbp_saliency.flatten())) / (
                max(gbp_saliency.flatten()) - min(gbp_saliency.flatten()))
        saliency = gbp_saliency

        saliency = gaussian_filter(saliency, sigma=4)
        grad1 = saliency

        """
        save images
        """

        mask[mask > 0.1] = 1
        mask[mask <= 0.1] = 0
        #for i in range(ano_region_mask.shape[0]):

        anomaly_map = gaussian_filter(grad1, sigma=4)
        anomaly_map = min_max_norm(anomaly_map)  # 0~1 mapping
        ano_map = (anomaly_map*255).astype(np.uint8)
        # print(ano_map)
        ano_map = cvt2heatmap(ano_map[0])
        #img = image.repeat(1,3,1,1)
        #img = cv2.cvtColor((img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        path = sample['path']
        image = cv2.imread(path[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = np.uint8(min_max_norm((img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8))*255)
        #ano_map = show_cam_on_image(image, ano_map)
        ano_map = cv2.cvtColor(ano_map, cv2.COLOR_BGR2RGB)
        ano_map = cv2.addWeighted(ano_map, 0.4, image, (1 - 0.4), 0)
        #print(mask.cpu().numpy().astype(int).shape, anomaly_map.shape)
        # if Y.item()!=0:
        #     #print(mask.cpu().numpy().astype(int)[0].shape, anomaly_map.reshape(sample['image'].shape[0],224,224).shape)
        #     aupro_list.append(compute_pro(mask.cpu().numpy().astype(int)[0], anomaly_map.reshape(sample['image'].shape[0],224,224)))
        gt_list_px.extend(mask.cpu().numpy().astype(int).ravel())
        #print(mask.cpu().numpy().astype(int).ravel().shape)
        pr_list_px.extend(anomaly_map.ravel())
        #print(anomaly_map.ravel().shape)
        anomaly_map[anomaly_map >= 0.5] = 1 
        anomaly_map[anomaly_map < 0.5] = 0
        if Y.item()!=0:
            dice_list.append(compute_dice(mask.cpu().numpy().astype(int)[0], anomaly_map.reshape(sample['image'].shape[0],224,224)))
        #print(img.shape, ano_map.shape, anomaly_map.shape)
        wandb.log({
            "img | gt | heatmap | pred": [wandb.Image(image), wandb.Image(mask), wandb.Image(ano_map), wandb.Image(anomaly_map[0])]
        })


    # auroc_px = round(metrics.roc_auc_score(gt_list_px, pr_list_px), 5)
    # aupro_px = round(np.mean(aupro_list), 5)
    # print('auroc_px: ', auroc_px, ',', 'aupro_px', aupro_px)
    print("dice_px: ", (round(np.mean(dice_list), 5)))
    #print(gt_list), pred_list

def gbp_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("GBP Method:")

    grad1 = None
    i = 0
    t = 1

    for _, sample in enumerate(test_dataloader):
        img = sample['image'].cuda()
        mask = sample['mask'].cuda()
        Y = sample['label'].cuda()
        X = Variable(img).cuda()
        #print(X.shape)
        grad1 = np.zeros((X.shape[0], 1, 224, 224), dtype=np.float32)
        #print(grad1.shape) # [1,1,256,256]
        for x in X:
            data = x.view(1, 3, 224, 224)
            GBP = GuidedBackprop(model, vgg, 'cuda:0')
            gbp_saliency = abs(GBP.generate_gradients(data, config))
            gbp_saliency = (gbp_saliency - min(gbp_saliency.flatten())) / (
                    max(gbp_saliency.flatten()) - min(gbp_saliency.flatten()))
            saliency = gbp_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[i] = saliency
            i += 1
        t+=1

    grad1 = grad1.reshape(-1, 224, 224)
    #print(grad1.shape)
    return grad1, X, mask


def grad_calc(inputs, model, vgg, config):
    inputs = inputs.cuda()
    inputs.requires_grad = True
    temp = torch.zeros(inputs.shape)
    lamda = config['lamda']
    criterion = nn.MSELoss()
    similarity_loss = torch.nn.CosineSimilarity()

    for i in range(inputs.shape[0]):
        output_pred = model.forward(inputs[i].unsqueeze(0), target_layer=14)
        output_real = vgg(inputs[i].unsqueeze(0))
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]
        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
        model.zero_grad()
        total_loss.backward()

        temp[i] = inputs.grad[i]

    return temp


def gradients_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("Vanilla Backpropagation:")
    temp = None
    for i, sample in enumerate(test_dataloader):
        img = sample['image'].cuda()
        mask = sample['mask'].cuda()
        Y = sample['label'].cuda()
        X = Variable(img).cuda()
        grad = grad_calc(X, model, vgg, config)
        temp = np.zeros((grad.shape[0], grad.shape[2], grad.shape[3]))
        for i in range(grad.shape[0]):
            grad_temp = convert_to_grayscale(grad[i].cpu().numpy())
            grad_temp = grad_temp.squeeze(0)
            grad_temp = gaussian_filter(grad_temp, sigma=4)
            temp[i] = grad_temp
    return temp


class VanillaSaliency():
    def __init__(self, model, vgg, device, config):
        self.model = model
        self.vgg = vgg
        self.device = device
        self.config = config
        self.model.eval()

    def generate_saliency(self, data, make_single_channel=True):
        data_var_sal = Variable(data).to(self.device)
        self.model.zero_grad()
        if data_var_sal.grad is not None:
            data_var_sal.grad.data.zero_()
        data_var_sal.requires_grad_(True)

        lamda = self.config['lamda']
        criterion = nn.MSELoss()
        similarity_loss = torch.nn.CosineSimilarity()

        output_pred = self.model.forward(data_var_sal)
        output_real = self.vgg(data_var_sal)
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
        self.model.zero_grad()
        total_loss.backward()
        grad = data_var_sal.grad.data.detach().cpu()

        if make_single_channel:
            grad = np.asarray(grad.detach().cpu().squeeze(0))
            # grad = max_regarding_to_abs(np.max(grad, axis=0), np.min(grad, axis=0))
            # grad = np.expand_dims(grad, axis=0)
            grad = convert_to_grayscale(grad)
            # print(grad.shape)
        else:
            grad = np.asarray(grad)
        return grad


def generate_smooth_grad(data, param_n, param_sigma_multiplier, vbp, single_channel=True):
    smooth_grad = None

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(data) - torch.min(data)).item()
    VBP = vbp
    for x in range(param_n):
        noise = Variable(data.data.new(data.size()).normal_(mean, sigma ** 2))
        noisy_img = data + noise
        vanilla_grads = VBP.generate_saliency(noisy_img, single_channel)
        if not isinstance(vanilla_grads, np.ndarray):
            vanilla_grads = vanilla_grads.detach().cpu().numpy()
        if smooth_grad is None:
            smooth_grad = vanilla_grads
        else:
            smooth_grad = smooth_grad + vanilla_grads

    smooth_grad = smooth_grad / param_n
    return smooth_grad


class IntegratedGradients():
    def __init__(self, model, vgg, device):
        self.model = model
        self.vgg = vgg
        self.gradients = None
        self.device = device
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps + 1) / steps
        xbar_list = [input_image * step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.vgg, self.device)
        saliency = vanillaSaliency.generate_saliency(input_image, make_single_channel)
        if not isinstance(saliency, np.ndarray):
            saliency = saliency.detach().cpu().numpy()
        return saliency

    def generate_integrated_gradients(self, input_image, steps, make_single_channel=True):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = None
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, False)
            if integrated_grads is None:
                integrated_grads = deepcopy(single_integrated_grad)
            else:
                integrated_grads = (integrated_grads + single_integrated_grad)
        integrated_grads /= steps
        saliency = integrated_grads[0]
        img = input_image.detach().cpu().numpy().squeeze(0)
        saliency = np.asarray(saliency) * img
        if make_single_channel:
            saliency = max_regarding_to_abs(np.max(saliency, axis=0), np.min(saliency, axis=0))
        return saliency


def generate_integrad_saliency_maps(model, vgg, preprocessed_image, device, steps=100, make_single_channel=True):
    IG = IntegratedGradients(model, vgg, device)
    integrated_grads = IG.generate_integrated_gradients(preprocessed_image, steps, make_single_channel)
    if make_single_channel:
        integrated_grads = convert_to_grayscale(integrated_grads)
    return integrated_grads


class GuidedBackprop():
    def __init__(self, model, vgg, device):
        self.model = model
        self.vgg = vgg
        self.gradients = None
        self.forward_relu_outputs = []
        self.device = device
        self.hooks = []
        self.model.eval()
        self.update_relus()

    def update_relus(self):

        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                self.hooks.append(module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_image, config, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.vgg, self.device, config=config)
        sal = vanillaSaliency.generate_saliency(input_image, make_single_channel)
        if not isinstance(sal, np.ndarray):
            sal = sal.detach().cpu().numpy()
        for hook in self.hooks:
            hook.remove()
        return sal


def smooth_grad_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("Smooth Grad Method:")

    grad1 = None
    i = 0

    for i, sample in enumerate(test_dataloader):
        img = sample['image'].cuda()
        mask = sample['mask'].cuda()
        Y = sample['label'].cuda()
        X = Variable(img).cuda()
        grad1 = np.zeros((X.shape[0], 1, 224, 224), dtype=np.float32)
        for x in X:
            data = x.view(1, 3, 224, 224)

            vbp = VanillaSaliency(model, vgg, 'cuda:0', config)

            smooth_grad_saliency = abs(generate_smooth_grad(data, 50, 0.05, vbp))
            smooth_grad_saliency = (smooth_grad_saliency - min(smooth_grad_saliency.flatten())) / (
                    max(smooth_grad_saliency.flatten()) - min(smooth_grad_saliency.flatten()))
            saliency = smooth_grad_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[i] = saliency
            i += 1

    grad1 = grad1.reshape(-1, 224, 224)
    return grad1


def compute_localization_auc(grad, X, x_ground):
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    tpr = []
    fpr = []
    #print(x_ground.shape) # (51, 224, 224)
    x_ground_comp = torch.mean(x_ground, axis=1).cpu().numpy() # [51, 3, 224, 224])

    thresholds = [0.001 * i for i in range(1000)]

    ano_map_all = grad
    ano_map_all = min_max_norm(ano_map_all)  # 0~1 mapping [640,1,256,256]
    x_ground[x_ground > 0.5] = 1
    x_ground[x_ground <= 0.5] = 0
    #print(x_ground.shape)
    #print(ano_map_all.shape)
    #print(ano_map.shape) # [640, 224, 224]
    aupro_list.append(compute_pro(x_ground[:,0,:,:].cpu().numpy(), ano_map_all[:,:,:]))
    print("pro: ", (round(np.mean(aupro_list), 5)))

    dice_list = compute_dice(x_ground[:,0,:,:].cpu().numpy(), ano_map_all[:,:,:])
    print("dice: ", (round(np.mean(dice_list), 5)))
    

    for i in range(ano_map_all.shape[0]):
        ano_map = cvt2heatmap(np.uint8(ano_map_all[i]*255))
        # X: [640, 3, 224, 224] --> [640, 224, 224, 3]
        img = cv2.cvtColor(X.permute(0, 2, 3, 1).cpu().numpy()[i] * 255, cv2.COLOR_BGR2RGB)
        img = np.uint8(min_max_norm(img)*255)
        ano_map = show_cam_on_image(img, ano_map)
        ano_map = cv2.cvtColor(ano_map, cv2.COLOR_BGR2RGB)
        # ano_map = cvt2heatmap(ano_map*255)
        # img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        # img = np.uint8(min_max_norm(img)*255)
        # ano_map = show_cam_on_image(img, ano_map)
        
        #print(img.shape) #(224, 224, 3)
        #print(x_ground.shape) # (180, 224, 224, 3)
        #print(ano_map.shape) # (224, 224, 3)
        # wandb.log({
        #     "img": [wandb.Image(img), wandb.Image(x_ground[i]), wandb.Image(ano_map)]
        # })
    #print(x_ground_comp.shape)
    #print(grad.shape)
    for threshold in thresholds:
        grad_t = 1.0 * (grad >= threshold)
        grad_t = morphological_process(grad_t)
        tp_map = np.multiply(grad_t, x_ground_comp)
        tpr.append(np.sum(tp_map) / np.sum(x_ground_comp))

        inv_x_ground = 1 - x_ground_comp
        fp_map = np.multiply(grad_t, inv_x_ground)
        tn_map = np.multiply(1 - grad_t, 1 - x_ground_comp)
        fpr.append(np.sum(fp_map) / (np.sum(fp_map) + np.sum(tn_map)))

    return auc(fpr, tpr)

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