import datetime
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BMAD: Benchmarks for Medical Anomaly Detection')
    parser.add_argument('--data', default="RESC",
                        help='dataset information, select from ["RESC", "bras2021", "camelyon", "chest", "liver", "OCT2017"] ')
    parser.add_argument('--model', default="RD4AD",
                        help='dataset information, select from ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50", "draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"] ')
    parser.add_argument('--mode', default="train",
                        help='train or test')
    parser.add_argument('--weight', default=None,
                        help='weight file')
    args = parser.parse_args()
    
    data = args.data
    model = args.model
    mode = args.mode

    if mode == 'train':
        if model in ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50","draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"]:
            os.system(f'python anomalib/tools/train.py --config config/{data}_{model}.yaml')
        elif model == "Deep-SVDD":
            os.system(f"python Deep-SVDD/main.py {data} cifar10_LeNet Deep-SVDD/log/{data} /home/jinan/")
        elif model == "UTRAD":
            os.system(f"python UTRAD/main.py --dataset_name {data}")
        elif model == "MKD":
            os.system(f"python MKD/train.py --config config/{data}_{model}.yaml")
        elif model == "cutpaste":
            os.system(f"python pytorch-cutpaste/run_training.py --type {data}")
        else:
            print(f'ERROR, you input a wrong model {model}, please select from ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50", "draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"]')
    
    elif mode == 'test':
        if model in ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50","draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"]:
            os.system(f'python anomalib/tools/test.py --model {model} --config config/{data}_{model}_test.yaml --weight_file {args.weight}')
        elif model == "UTRAD":
            os.system(f"python UTRAD/valid.py --dataset_name {data} --weight {args.weight}")
        elif model == "MKD":
            os.system(f"python MKD/test.py --config config/{data}_{model}.yaml")
        elif model == "cutpaste":
            os.system(f"python pytorch-cutpaste/eval.py --type {data} --weight {args.weight}")
        else:
            print(f'ERROR, you input a wrong model {model}, please select from ["padim", "padim_resnet50", "stfpm", "stfpm_resnet50", "draem", "cfa", "cflow", "ganomaly", "RD4AD", "patchcore", "patchcore_resnet50"]')
