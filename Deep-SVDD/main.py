import click
import torch
import logging
import random
import numpy as np
from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms, TestDataset
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset
import os
import yaml


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['RESC', 'OCT2017', 'liver', 'bras2021', 'chest', 'camelyon']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU']))
@click.argument('xp_path')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
# @click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
#               help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
# @click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
# @click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
# @click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
#               help='Name of the optimizer to use for Deep SVDD network training.')
# @click.option('--lr', type=float, default=0.001,
#               help='Initial learning rate for Deep SVDD network training. Default=0.001')
# @click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
# @click.option('--lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
# @click.option('--weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
# @click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
#               help='Name of the optimizer to use for autoencoder pretraining.')
# @click.option('--ae_lr', type=float, default=0.001,
#               help='Initial learning rate for autoencoder pretraining. Default=0.001')
# @click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
# @click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
# @click.option('--ae_weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, device, pretrain, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    config_path = f"config/{dataset_name}_DeepSVDD.yaml"
    print(f"reading config {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    if os.path.exists(xp_path) !=True:
        os.makedirs(xp_path)
    # Get configuration
    
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % config['objective'])
    logger.info('Nu-paramerter: %.2f' % config['nu'])

    # Set seed
    if config['seed'] != -1:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        logger.info('Set seed to %d.' % config['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Datasets
    print(dataset_name)
    train_dataset = TrainDataset(data=dataset_name, transform=get_train_transforms())
    valid_dataset = ValidDataset(data=dataset_name, transform=get_valid_transforms())
    test_dataset = TestDataset(data=dataset_name, transform=get_valid_transforms())
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=11164, num_workers=8)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(config['objective'], config['nu'])
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % config['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % config['ae_lr'])
        logger.info('Pretraining epochs: %d' % config['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (config['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % config['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % config['ae_weight_decay'])

        #print(cfg.settings['ae_lr_milestone'], config['ae_lr_milestone'])
        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(train_dataset,
                           optimizer_name=config['ae_optimizer_name'],
                           lr=config['ae_lr'],
                           n_epochs=config['ae_n_epochs'],
                           lr_milestones=(config['ae_lr_milestone'],),
                           batch_size=config['ae_batch_size'],
                           weight_decay=config['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % config['optimizer_name'])
    logger.info('Training learning rate: %g' % config['lr'])
    logger.info('Training epochs: %d' % config['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (config['lr_milestone'],))
    logger.info('Training batch size: %d' % config['batch_size'])
    logger.info('Training weight decay: %g' % config['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(train_dataset,
                    optimizer_name=config['optimizer_name'],
                    lr=config['lr'],
                    n_epochs=config['n_epochs'],
                    lr_milestones=(config['lr_milestone'],),
                    batch_size=config['batch_size'],
                    weight_decay=config['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(valid_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    deep_SVDD.test(test_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # # Plot most anomalous and most normal (within-class) test samples
    # indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    # if dataset_name in ('mnist', 'cifar10', 'resc'):

    #     if dataset_name == 'mnist':
    #         X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
    #         X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

    #     if dataset_name == 'cifar10':
    #         X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
    #         X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

    #     if dataset_name == 'resc':
    #         X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
    #         X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

    #     plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
    #     plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)

    # Save results, model, and configuration
    #deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + f'/model_{dataset_name}.tar')
    #cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
