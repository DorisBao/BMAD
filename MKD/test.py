from argparse import ArgumentParser
from utils import get_config
from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms, TestDataset
from test_functions import detection_test, localization_test
from models.network import get_networks
import wandb, torch, time

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    vgg, model = get_networks(config, load_checkpoint=True)

    # Localization test
    # if config['localization_test']:
    #     test_dataloader, ground_truth = load_localization_data(config)
    #     roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, #ground_truth=ground_truth,config=config) 

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=config['dataset_name'],
    #     name = 'MKD',
    # )
    test_dataset = ValidDataset(data=config['dataset_name'], transform=get_valid_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8)
    #localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)

    # Detection tes
    roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    last_checkpoint = config['last_checkpoint']
    print("RocAUC after {} epoch:".format(last_checkpoint), roc_auc)
    
    test_dataset = TestDataset(data=config['dataset_name'], transform=get_valid_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8)
    t1 = time.time()
    #localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    t2 = time.time()
    print(t2-t1, len(test_dataloader), len(test_dataloader)/(t2-t1))

    # Detection test
    t1 = time.time()
    roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    t2 = time.time()
    print(t2-t1, len(test_dataloader), len(test_dataloader)/(t2-t1))
    last_checkpoint = config['last_checkpoint']
    print("RocAUC after {} epoch:".format(last_checkpoint), roc_auc)
    # visualize test


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
