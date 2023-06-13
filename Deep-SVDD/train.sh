# python main.py RESC cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3

# !!!!!
# python main.py RESC cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3

python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.000001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-7 --normal_class 3;

nohup python main.py RESC cifar10_LeNet ../log/RESC /home/jinan/data/RESC_Pnet-Test --load_model /home/jinan/2023-Doris/hanshi/Retinal-OCT-AD/Deep-SVDD/log/RESC/model.tar --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > out.Val.log &

nohup python main.py OCT2017 cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > RESC_1.log & 

nohup python main.py RESC cifar10_LeNet log/RESC /home/jinan/ --objective one-class --load_model /home/jinan/2023-Doris/hanshi/Retinal-OCT-AD/Deep-SVDD/log/RESC/model.tar --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > RESC_2.log & 

nohup python main.py OCT2017 cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > OCT2017_0.log & 

nohup python main.py OCT2017 cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > OCT2017_1.log & 

nohup python main.py OCT2017 cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > OCT2017_2.log & 

python main.py OCT2017 cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 1 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3

nohup python main.py chest cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > chest_1.log & 

nohup python main.py liver cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > liver_1.log & 

CUDA_VISIBLE_DEVICES=1 nohup python main.py liver cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > liver_2.log & 

python main.py liver cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3

python main.py liver cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3

python main.py bras2021 cifar10_LeNet log/RESC /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3

nohup python main.py camelyon cifar10_LeNet log/camelyon /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 150 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 > liver_1.log & 

python main.py chest cifar10_LeNet log/chest /home/jinan/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-7 --pretrain True --ae_lr 0.00001 --ae_n_epochs 300 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3