#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实训练脚本 - 使用分布式快照方法
训练h=5, 40, 256三种宽度，每个minibatch记录一次权重
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from dynamic_mode_decomposition import time_delay, data_matrices, dmd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 全局变量 - 将在训练时设置
h = 5

class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        alpha = 1.0
        self.fc1 = nn.Linear(28*28, hidden_size, bias=False)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight * alpha)
        self.fc2 = nn.Linear(hidden_size, 10, bias=False)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight * alpha)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, datapath):
    model.train()
    # 每个batch都记录，所以记录次数等于batch数量
    n_batches = len(train_loader)
    n_samples = len(train_loader.dataset)
    print(f'训练集信息: {n_samples} 个样本, {n_batches} 个batch (batch_size={args.batch_size})')
    loss_vec = np.zeros(n_batches)
    weights = np.zeros([n_batches, 10 * args.hidden_size])
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        if not datapath:
            loss.backward()
            optimizer.step()
        else:
            print(f'[WARNING] datapath={datapath}, 跳过训练步骤！')
        
        # 每个batch都记录
        loss_vec[batch_idx] = loss.item()
        weights[batch_idx, :] = model.fc2.weight.flatten().cpu().detach().numpy()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
        if args.dry_run:
            break
    
    return loss_vec, weights

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy

def main(hidden_size, batch_seed, init_seed, save_flag, load_flag):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training with Distribution Snapshots')
    parser.add_argument('--batch-size', type=int, default=60, metavar='N',
                        help='input batch size for training (default: 60)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--hidden-size', type=int, default=hidden_size, metavar='H',
                        help='hidden layer size')
    args = parser.parse_args()
    args.hidden_size = hidden_size
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    print(f'MNIST数据集加载: 训练集={len(dataset1)} 个样本, 测试集={len(dataset2)} 个样本')
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False)
    print(f'DataLoader创建: 训练集batch数={len(train_loader)}, 测试集batch数={len(test_loader)}')

    model = Net(hidden_size).to(device)
    
    torch.manual_seed(init_seed)
    if save_flag:
        folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results', '')
        os.makedirs(folder_name, exist_ok=True)
        experiment_name = 'SGD_MNIST_h=' + str(hidden_size)
        torch.save(model.state_dict(), os.path.join(folder_name, experiment_name + '_seed' + str(batch_seed) + '_initialization.pt'))
    
    if load_flag:
        folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results', '')
        experiment_name = 'SGD_MNIST_h=' + str(hidden_size)
        initialization = torch.load(os.path.join(folder_name, experiment_name + '_seed' + str(batch_seed) + '_initialization.pt'))
        initialization['fc1.weight'] += (0.001 * (2 * torch.rand(initialization['fc1.weight'].size()) - 1) * initialization['fc1.weight'])
        initialization['fc2.weight'] += (0.001 * (2 * torch.rand(initialization['fc2.weight'].size()) - 1) * initialization['fc2.weight'])
        model.load_state_dict(initialization)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    Loss = []
    Train_Loss = []
    Test_Accuracy = []
    
    torch.manual_seed(batch_seed)
    
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch+1}/{args.epochs} (h={hidden_size}) ===')
        loss_vec, W = train(args, model, device, train_loader, optimizer, epoch, '')
        Train_Loss.append(loss_vec)
        test_loss, test_acc = test(model, device, test_loader)
        Loss.append(test_loss)
        Test_Accuracy.append(test_acc)

    Loss = np.array(Loss)
    Test_Accuracy = np.array(Test_Accuracy)
    
    return args, Loss, Train_Loss, W, Test_Accuracy

if __name__ == '__main__':
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(current_dir, 'Results', '')
    os.makedirs(folder_name, exist_ok=True)
    
    # 实验参数
    h_values = [5, 40, 256, 1024]
    random_seeds = np.array([2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101])  # 25个不同种子
    n_inits = 1  # 每个种子1个初始化
    
    print("=" * 60)
    print("开始真实训练实验（分布式快照方法）")
    print(f"网络宽度: {h_values}")
    print(f"随机种子: {random_seeds}")
    print(f"每个种子初始化数: {n_inits}")
    print(f"训练epochs: 1")
    print("=" * 60)
    
    start_time = time.time()
    
    # GPU并行训练：使用线程池（因为PyTorch的DataLoader已经支持多进程）
    # 对于GPU训练，我们顺序执行不同种子，但每个训练内部可以使用GPU加速
    max_workers = 1  # 由于GPU资源限制，顺序执行；可以改为更大值如果有多个GPU
    
    def train_single_seed(h_val, batch_seed, init, experiment_name):
        """训练单个种子的函数"""
        print(f"\n--- 种子 {batch_seed}, 初始化 {init+1}/{n_inits} (h={h_val}) ---")
        
        try:
            if init == 0:
                Args, Loss, Train_Loss, W, Test_Accuracy = main(
                    h_val, batch_seed=batch_seed, init_seed=init, 
                    save_flag=True, load_flag=False)
            else:
                Args, Loss, Train_Loss, W, Test_Accuracy = main(
                    h_val, batch_seed=batch_seed, init_seed=init, 
                    save_flag=False, load_flag=True)
            
            # 保存结果
            weight_file = os.path.join(folder_name, 
                experiment_name + '_seed' + str(batch_seed) + 
                '_initialization' + str(init) + '_weights.npy')
            np.save(weight_file, W)
            
            train_loss_file = os.path.join(folder_name,
                experiment_name + '_seed' + str(batch_seed) + 
                '_initialization' + str(init) + '_Train_Loss.npy')
            np.save(train_loss_file, Train_Loss)
            
            test_loss_file = os.path.join(folder_name,
                experiment_name + '_seed' + str(batch_seed) + 
                '_initialization' + str(init) + '_Loss.npy')
            np.save(test_loss_file, Loss)
            
            test_acc_file = os.path.join(folder_name,
                experiment_name + '_seed' + str(batch_seed) + 
                '_initialization' + str(init) + '_Test_Accuracy.npy')
            np.save(test_acc_file, Test_Accuracy)
            
            print(f"  ✓ 种子 {batch_seed} (h={h_val}) 完成: 权重已保存 (shape: {W.shape}), 最终测试准确率: {Test_Accuracy[-1]:.2f}%")
            return True
        except Exception as e:
            print(f"  ✗ 种子 {batch_seed} (h={h_val}) 失败: {e}")
            return False
    
    for h_val in h_values:
        print(f"\n{'='*60}")
        print(f"训练 h={h_val} 的网络")
        print(f"{'='*60}")
        
        experiment_name = 'SGD_MNIST_h=' + str(h_val)
        
        # 顺序执行（GPU训练时推荐，避免GPU内存冲突）
        # 如果需要并行，可以取消下面的注释并使用ThreadPoolExecutor
        for seed_idx, batch_seed in enumerate(random_seeds):
            for init in range(n_inits):
                train_single_seed(h_val, batch_seed, init, experiment_name)
        
        # 可选：使用线程池并行（如果有多个GPU或CPU训练）
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     futures = []
        #     for seed_idx, batch_seed in enumerate(random_seeds):
        #         for init in range(n_inits):
        #             future = executor.submit(train_single_seed, h_val, batch_seed, init, experiment_name)
        #             futures.append(future)
        #     
        #     for future in as_completed(futures):
        #         future.result()  # 等待完成
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成！总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"{'='*60}")
    
    # 训练完成后自动运行可视化
    print("\n开始执行可视化分析...")
    import subprocess
    import sys
    
    # 运行分析脚本
    analysis_script = os.path.join(current_dir, 'analyze_distribution_snapshot.py')
    if os.path.exists(analysis_script):
        subprocess.run([sys.executable, analysis_script])
    else:
        print("分析脚本不存在，将创建并运行...")
        # 这里会在下一步创建分析脚本
