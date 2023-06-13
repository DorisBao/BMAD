from test import *
from utils import *
from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test, localization_test
from loss_functions import *

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def train(config):
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']

    checkpoint_path = "./outputs/{}/checkpoints/".format(config['dataset_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    # Datasets
    train_dataset = TrainDataset(data=config['dataset_name'], transform=get_train_transforms())
    valid_dataset = ValidDataset(data=config['dataset_name'], transform=get_valid_transforms())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=11164, num_workers=8)

    if continue_train:
        vgg, model = get_networks(config, load_checkpoint=True)
    else:
        vgg, model = get_networks(config)

    # Criteria And Optimizers
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if continue_train:
        optimizer.load_state_dict(
            torch.load('{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))

    losses = []
    #roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=valid_dataloader, config=config)
    #print("pixel-auc: ", roc_auc)
    for epoch in range(num_epochs + 1):
        model.train()
        epoch_loss = 0
        for i, sample in enumerate(train_dataloader):
            img = sample['image'].cuda()

            output_pred = model.forward(img)
            output_real = vgg(img)

            total_loss = criterion(output_pred, output_real)

            # Add loss to the list
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())

            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        # if epoch % 10 == 0:
        #     print('yeah')
        #     roc_auc = detection_test(model, vgg, valid_dataloader, config)
        #     print("RocAUC at epoch {}:".format(epoch), roc_auc)
        #     print("1-RocAUC at epoch {}:".format(epoch), 1-roc_auc)
        #     #roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=valid_dataloader, config=config)
        #     #print("pixel-auc: ", roc_auc)

        if epoch % 200 == 0:
            torch.save(model.state_dict(),
                       '{}Cloner_{}_epoch_{}_2.pth'.format(checkpoint_path, normal_class, epoch))


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = parser.parse_args()
    config = get_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
