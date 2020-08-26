import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import FundusDataset
from transform import train_transfrom, test_transfrom
from model import FundusModel
from loss import WeightFocalLoss

def build_argparse():
    parser = argparse.ArgumentParser()
    # Basic
    # parser.add_argument('--exp', help='The index of this experiment', type=int, default=0)
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--optim', type=str, default='Adam')
    
    # FC and Albumentation
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    # Additional Hyperparameter
    parser.add_argument('--lr', type=float,default=0.0001)
    parser.add_argument('--lr_name', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--output_class', type=int, default=10)
    parser.add_argument('--grad_acc', type=bool, default=False)

    # Loop control
    parser.add_argument('--epoch', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default=16)

    # Additional    
    parser.add_argument('--load_model_para', help='Enter the model.pth file name', type=str, default=None)
    parser.add_argument('--machine', help='The machine current is using: (local, dell172)', type=str, default='dell172')

    return parser

def check_argparse(args):
    assert args.model_name in [
                                'resnet18', 'resnet152', 
                                'densenet121', 'densenet161', 
                                'se_resnet50', 'se_resnet152',
                                'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                'efficientnet-b0', 
                                'efficientnet-b7' 
                                ], 'the model name is not included'

    assert args.optim in ['Adam', 'SGD']
    assert args.lr_name in ['ReduceLROnPlateau', 'StepLR']
    print('\n---- Training parameters ----')
    print(f'model name: {args.model_name}')
    print(f'image size: {args.image_size}')
    print(f'Grad acc  : {args.grad_acc}')
    print(f'Optimizer : {args.optim}')
    print(f'Activation: {args.activation}')
    print(f'Hidden dim: {args.hidden_dim}')
    print(f'Randomseed: {args.seed}')
    print(f'Initial lr: {args.lr}')
    print(f'lr name   : {args.lr_name}')
    print(f'Parafreeze: {args.freeze}')
    print(f'load .pth : {args.load_model_para}')
    print(f'Output cls: {args.output_class}')
    print(f'Epoch     : {args.epoch}')
    print(f'Batch size: {args.batch_size}')
    

def build_train_val_test_dataset(args):
    train_dataset = FundusDataset(mode='train', transform=True, 
                                image_size=args.image_size, seed=args.seed)
    val_dataset   = FundusDataset(mode='val', image_size=args.image_size, seed=args.seed)
    test_dataset  = FundusDataset(mode='test', image_size=args.image_size, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=os.cpu_count(),batch_size=args.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, pin_memory=True, num_workers=2*os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    test_dataloader   = DataLoader(test_dataset, pin_memory=True, num_workers=2*os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    weighting = train_dataset.classWeight()
    return train_dataloader, val_dataloader, test_dataloader, weighting

def freeze_pretrain(model, freeze=True):
    if freeze:
        for name, par in model.named_parameters():
            if name.startswith('cnn_model'):
                par.requires_grad = False
    else:
        for name, par in model.named_parameters():
            if name.startswith('cnn_model'):
                par.requires_grad = True

def build_scheduler(optimizer, name, freeze):
    if name == 'ReduceLROnPlateau':
        if freeze == True:
            scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=6)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=2)
        
    elif name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    return scheduler
  
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    # machine path
    if args.machine == 'local':
        machine_path = '/home/rico-li/Job/Ophthalmoscope'
    elif args.machine == 'dell172':
        machine_path = '/home/aiuser/Job/Ophthalmoscope'
    else:
        print('no support directory in your machine')
        raise IOError
    
    trial_num = os.listdir( os.path.join(machine_path, 'runs') )
    exp_num = [int(num.replace('trial_', '')) for num in trial_num]
    exp_num = max(exp_num) +1 
    print(f'\n== Trial {exp_num} begins ==\n')

    # data
    print('\n-------- Data Preparing --------\n')
    train_dataloader, val_dataloader, test_dataloader, weighting = build_train_val_test_dataset(args)
    print('\n-------- Data Preparing Done! --------\n')

    # model
    print('-------- Preparing Model --------')
    model = FundusModel(model_name = args.model_name, hidden_dim=args.hidden_dim, 
                        activation=args.activation, output_class=args.output_class)
    # freeze CNN pretrained model
    if args.freeze:
        freeze_pretrain(model, True)
    else:
        freeze_pretrain(model, False)

    # loading previous trained model parameters
    # usually for freeze-unfreeze method
    if args.load_model_para:
        model.load_state_dict(torch.load( os.path.join(machine_path,'model_save', args.load_model_para)))
    else:
        pass
    

    # pass to CUDA device
    model = model.to(device)
    
    # add in class weighting
    weighting = torch.tensor(weighting).to(device)
    criterion = nn.CrossEntropyLoss(weighting)

    if args.optim == 'Adam':
        # before acc 80 %
        optimizer = optim.Adam(model.parameters())
    elif args.optim == 'SGD':
        # after acc 80 %
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, nesterov=True, weight_decay=0.01)

    scheduler = build_scheduler(optimizer, args.lr_name, args.freeze)

    print('-------- Preparing Model Done! --------')

    # train
    print('\n-------- Starting Training --------\n')
    # tensorboard
    writer = SummaryWriter(f'runs/trial_{exp_num}')

    # comparsion of accuracy, in order to save the best weight
    accuracies = [0.]
    k = 0
    for epoch in range(args.epoch):
        start_time = time.time()

        train_running_loss = 0.0
        print(f'--- The {epoch+1}/{args.epoch} epoch ---')
        #  --------------------------- TRAINING LOOP ---------------------------
        print('\n--- Training Loop begins ---')
        print('[Epoch, Batch] : Loss')        
        optimizer.zero_grad()
        model.train()
        for i, data in enumerate(train_dataloader, start=0):
            input, target = data[0].to(device), data[1].to(device)
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            train_running_loss += loss.item()
            if args.grad_acc == True:
                if (i+1)%args.batch_size == 0:  # real batch size is args.batch_size**2
                    k += 1  
                    writer.add_scalar('Batch-Averaged loss', train_running_loss/(args.batch_size), k)
                    print( f"[{epoch+1}, {i+1}]: %.3f" % (train_running_loss/(args.batch_size)) )
                    optimizer.step()
                    optimizer.zero_grad()
                    train_running_loss = 0.0
            else:
                optimizer.step()
                optimizer.zero_grad()
                if (i+1)%50 == 0:
                    k += 1
                    writer.add_scalar('Batch-Averaged loss', train_running_loss, k)
                    print( f"[{epoch+1}, {i+1}]: %.3f" % train_running_loss)
                train_running_loss = 0.0
        
        lr = [group['lr'] for group in optimizer.param_groups]
        print('Epoch:', f'{epoch+1}/{args.epoch}',' LR:', lr[0])
        writer.add_scalar('Learning Rate', lr[0], epoch)

        print('--- Training Loop ends ---\n')
        print(f'--- Training spend time: %.1f sec ---' % (time.time() - start_time))
        
        #  --------------------------- VALIDATION LOOP ---------------------------
        with torch.no_grad():
            model.eval()
            val_run_loss = 0.0
            print('\n--- Validaion Loop begins ---')
            start_time = time.time()
            batch_count = 0
            total_count = 0
            correct_count = 0
            for data in tqdm(val_dataloader, desc='Validation'):
                input, target = data[0].to(device), data[1].to(device)
                output = model(input)
                _, predicted = torch.max(output, 1)
                loss = criterion(output, target)
                val_run_loss += loss.item()
                correct_count += (predicted == target).sum().item()

                batch_count += 1
                total_count += target.size(0)
            accuracy = (100 * correct_count/total_count)
            val_run_loss = val_run_loss/batch_count

            
            if max(accuracies) < accuracy:
                savepath = os.path.join(f'{machine_path}','model_save',f'{exp_num}_{args.model_name}_best.pth')
                torch.save(model.state_dict(), savepath)
                print('\n-------- Saveing the best weight --------')
            else:
                print('\n-------- Accuracy is not improving --------')
            accuracies.append(accuracy)
            
            if args.lr_name == 'ReduceLROnPlateau':
                scheduler.step(val_run_loss)
            elif args.lr_name == 'StepLR':
                scheduler.step()

            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f"Loss of {epoch+1} epoch is %.3f" % (val_run_loss))
            print(f"Accuracy is %.2f %% \n" % (accuracy))
                
            print('--- Validaion Loop ends ---\n')
            print(f'--- Validaion spend time: %.1f sec ---' % (time.time() - start_time))
    writer.close()
    print('\n-------- End Training --------\n')
    print(f'\n--- Best accuracy: {max(accuracies):.2f} % ---')
    print(f'\n== Trial {exp_num} finished ==\n')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('--- Execution time ---')
    exe_time = (time.time() - start_time)
    hr = int(exe_time // 3600)
    min = int(((exe_time / 3600) - hr) * 60)
    sec = ((((exe_time / 3600) - hr) * 60) - min)*60
    print(f'--- {hr}:{min}:{sec:.1f} (hr:min:sec)---')

# --- code snippet ---
# tensorboard --logdir runs/trial_X/
# time python yourprogram.py

# Freeze
# python train.py --exp X --epoch 10 --freeze True --output_class 36
# Unfreeze and load .pth
# python train.py --exp X --epoch 15  --load_model_para 65_se_resnext101_32x4d.pth --output_class 36

# scp
# scp train.py aiuser@210.240.240.172:/home/aiuser/Job/Ophthalmoscope

# server
# python train.py --epoch 15 --batch_size 64 --image_size 300 --model_name se_resnext101_32x4d --load_model_para 8_se_resnext101_32x4d_best.pth --optim 'Adam' --lr 0.000015
# python train.py --epoch 10 --batch_size 64 --image_size 300 --model_name efficientnet-b0 --optim 'Adam' --freeze True
# python train.py --epoch 10 --batch_size 64 --image_size 300 --model_name se_resnext101_32x4d --optim 'Adam' --freeze True --output_class 5