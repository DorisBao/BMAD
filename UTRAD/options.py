import argparse
import os
import torch

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', type=str, default="Exp0-r18", help='the name of the experiment')
        self.parser.add_argument('--epoch_start', type=int, default=0 , help='epoch to start training from')
        self.parser.add_argument('--epoch_num', type=int, default=150, help='number of epochs of training')
        self.parser.add_argument('--factor', type=int, default=1, help='not implemented yet')
        self.parser.add_argument('--seed', type=int, default=116, help='random seed')
        self.parser.add_argument('--num_row', type=int, default=4, help='number of image in a rows for display')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation type for transformer')
        self.parser.add_argument('--unalign_test', action='store_true', default=False, help='whether to valid with unaligned data: \
        in this mode, test images are random ratate +-10degree, and randomcrop from 256x256 to 224x224')
        self.parser.add_argument('--data_root', type=str, default='/home/jinan/Doris/dataset/', help='dir of the dataset')
        self.parser.add_argument('--dataset_name', type=str, default="RESC", help='category name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
        self.parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
        self.parser.add_argument('--n_cpu', type=int, default=12, help='number of cpu threads to use during batch generation')
        
        self.parser.add_argument('--image_result_dir', type=str, default='result_images', help=' where to save the result images')
        self.parser.add_argument('--model_result_dir', type=str, default='saved_models', help=' where to save the checkpoints')
        self.parser.add_argument('--validation_image_dir', type=str, default='validation_images', help=' where to save the validation image')

        self.parser.add_argument('--weight', type=str, default=None)
        


    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        # os.makedirs('%s-%s/%s' % (args.exp_name, args.dataset_name, args.image_result_dir), exist_ok=True)
        # os.makedirs('%s-%s/%s' % (args.exp_name, args.dataset_name, args.model_result_dir), exist_ok=True)

        self.args = args
        return self.args
