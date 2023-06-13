import datetime
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BMAD: Benchmarks for Medical Anomaly Detection')
    parser.add_argument('--data', default="RESC",
                        help='dataset information, select from ["RESC", "bras2021", "camelyon", "chest", "liver", "OCT2017"] ')
    parser.add_argument('--model', default="RD4AD",
                        help='dataset information, select from ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50","draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"] ')
    parser.add_argument('--mode', default="train",
                        help='train or test')
    parser.add_argument('--weight', default="results/padim/OCT2017_resnet50/run/weights/model.ckpt",
                        help='weight file')
    args = parser.parse_args()
    
    data = args.data
    model = args.model
    mode = args.mode

    if mode == 'train':
        if model in ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50","draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"]:
            os.system(f'python anomalib/tools/train.py --config config/{data}_{model}.yaml')
        elif mode == "Deep-SVDD":
            os.system(f"python Deep-SVDD/main.py {data} cifar10_LeNet Deep-SVDD/log/{data} /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 300 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3")
    elif mode == 'test':
        if model in ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50","draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"]:
            os.system(f'python anomalib/tools/test.py --model {model} --config config/{data}_{model}_test.yaml --weight_file {args.weight}')
