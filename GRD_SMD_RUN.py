
import pickle
import os
import torch
import pandas as pd
from tqdm import tqdm
from src.eval import evaluate
#from src.utils import *
from src.parser import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from transformers import get_linear_schedule_with_warmup
from src.my_plotting import plotter
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
#from torch.utils.tensorboard import SummaryWriter
import argparse
from src.pot import *
from src.dqb_plot import dplotter
from tadpak import pak
from sklearn.metrics import *
from src.eval import get_fp_tp_rate
import pprint
from xmlrpc.client import Boolean
import datetime 
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(dataset, part=None):
    loader = []
    # folder = 'DiffusionAE/processed/' + dataset
    folder = 'processed/' + dataset + '/machine-1-1'

    for file in ['train', 'test', 'validation', 'labels']:
        if part is None:
            loader.append(np.load(os.path.join(folder, f'{file}.npy')))
        else:
            loader.append(np.load(os.path.join(folder, f'{part}_{file}.npy')))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    validation_loader = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, validation_loader, loader[3]

def load_model( lr, window_size, dims, batch_size, noise_steps, denoise_steps):
    from models2 import ConditionalDiffusionTrainingNetwork
    from grd import GRD
    scheduler=None
    # model = None
    diffusion_training_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps).float()
    diffusion_prediction_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps, train=False).float()
    # dims 就是节点数（num_nodes）
    model = GRD(n_features=dims, window_size=int(window_size), out_dim=dims,  lr=float(lr),batch=batch_size,device=device).float()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(diffusion_training_net.parameters()), lr=float(lr))

    return model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler

def convert_to_windows(data, n_window):
    windows = list(torch.split(data, n_window))
    for i in range (n_window-windows[-1].shape[0]):
        windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0))) # 将最后一个窗口的最后一行(形状为(1,通道数))添加到最后一个窗口中,直到最后一个窗口的长度达到 window_size:100
    return torch.stack(windows)

def get_diffusion_sample(diffusion_prediction_net, conditioner, k):
    if k <= 1:
        return diffusion_prediction_net(conditioner)
    else:
        diff_samples = []
        for _ in range(k):
            diff_samples.append(diffusion_prediction_net(conditioner))
        return torch.mean(torch.stack(diff_samples), axis = 0)

CHECKPOINT_FOLDER = './anomaly-mts/a-mts/checkpoints'
def save_model(model, experiment, diffusion_training_net, optimizer, scheduler,  epoch, diff_loss, ae_loss):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}/'
    os.makedirs(folder, exist_ok=True)
    if model:
        file_path_model = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'ae_loss': ae_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, file_path_model) # 会覆盖掉之前保存的模型，留下最好的
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    torch.save({
        'epoch': epoch,
        'diffusion_loss': diff_loss,
        'model_state_dict': diffusion_training_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),}, file_path_diffusion)
    print(f'epoch：{e},saved model at ' + folder)

def load_from_checkpoint(training_mode, experiment, model, diffusion_training_net):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}' # 2024/2/27修改前：{experiment}
    file_path_model = f'{folder}/model.ckpt'
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    # load model
    if training_mode == 'both':
        checkpoint_model = torch.load(file_path_model,map_location=torch.device(device))
        model.load_state_dict(checkpoint_model['model_state_dict'])
    else:
        model = None
    # load diffusion
    checkpoint_diffusion = torch.load(file_path_diffusion,map_location=torch.device(device))
    diffusion_training_net.load_state_dict(checkpoint_diffusion['model_state_dict'])
    return model, diffusion_training_net




def backprop(epoch, model, diffusion_training_net, diffusion_prediction_net, data, diff_lambda, optimizer, scheduler, training_mode, anomaly_score, k, training = True):
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_x, data_x)
    bs = diffusion_training_net.batch_size if not model else model.batch
    dataloader = DataLoader(dataset, batch_size = bs)
    w_size = diffusion_training_net.window_size
    l1s, diff_losses, ae_losses = [], [], [] # 一轮训练内用来收集每一个bacthsize的数据
    samples = []
    if training:
        # 模型进入训练模式
        if training_mode == 'both':
            model.train()
        diffusion_training_net.train()

        for d, _ in dataloader:
            ##### Clean trend datset here
            """mins = torch.min(d[:, :, 0], dim=1)
            maxs = torch.max(d[:, :, 0], dim=1)
            original.append(d)
            diffs = maxs[0] - mins[0]
            d = d[diffs < 0.04]
            cleaned.append(d)
            all_mins.append(mins)
            all_maxs.append(maxs)"""
            #####
            window = d # 此时window.shape为(32,100,38)即(batc_size,时间步长,通道变量数)
            window = window.to(device)

            if args.model == 'MTAD-GAT':
                ae_reconstruction = model(window)
            if args.model == 'GRD':
                ae_reconstruction = model(window)
            else:
                ae_reconstruction = model(window, window)
            # B x (feats * win)
            ae_loss = l(ae_reconstruction, window)
            ae_reconstruction = ae_reconstruction.reshape(-1, w_size, feats)
            # un tensor cu un element
            diffusion_loss, _ = diffusion_training_net(ae_reconstruction)
            ae_losses.append(torch.mean(ae_loss).item())
            diff_losses.append(torch.mean(diffusion_loss).item())
            if e < 5:
                loss = torch.mean(ae_loss)
            else:
                loss = diff_lambda * diffusion_loss + torch.mean(ae_loss)

            l1s.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        tqdm.write(f'Epoch {epoch},\tAE = {np.mean(ae_losses)}')
        tqdm.write(f'Epoch {epoch},\tDiff = {np.mean(diff_losses)}')
        return np.mean(l1s), np.mean(ae_losses), np.mean(diff_losses)
    else:
        with torch.no_grad():
            if training_mode == 'both':
                model.eval()
            diffusion_prediction_net.load_state_dict(diffusion_training_net.state_dict())
            diffusion_prediction_net.eval()
            diffusion_training_net.eval()
            l1s = [] # scores
            sum_losses = []
            ae_losses = []
            diff_losses = []
            recons = []
            for d, _ in dataloader:
                window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, feats)
                if training_mode == 'both':
                    if args.model == 'MTAD-GAT':
                        ae_reconstruction = model(window)
                    if args.model == 'GRD':
                        ae_reconstruction = model(window)
                    ae_reconstruction_reshaped = ae_reconstruction.reshape(-1, w_size, feats)
                    recons.append(ae_reconstruction_reshaped)
                    ae_loss = l(ae_reconstruction, window)
                    ae_losses.append(torch.mean(ae_loss).item())
                    _, diff_sample = diffusion_prediction_net(ae_reconstruction_reshaped)
                    diff_sample = torch.squeeze(diff_sample, 1)
                    diffusion_loss = l(diff_sample, window_reshaped)
                    diffusion_loss = torch.mean(diffusion_loss).item()
                    sum_losses.append(torch.mean(ae_loss).item() + diffusion_loss)
                    diff_losses.append(diffusion_loss)
                    samples.append(diff_sample)
                    if anomaly_score == 'both': # 1
                        loss = l(diff_sample, ae_reconstruction_reshaped)
                    elif anomaly_score == 'diffusion': # 3
                        loss = l(diff_sample, window_reshaped)
                    elif anomaly_score == 'autoencoder': # 2
                        loss = l(ae_reconstruction, window)
                    elif anomaly_score == 'sum': # 4 = 2 + 3
                        loss = l(ae_reconstruction, window) + l(window, diff_sample)
                    elif anomaly_score == 'sum2': # 5 = 1 + 2
                        loss = l(diff_sample, ae_reconstruction) + l(ae_reconstruction, window)
                    elif anomaly_score == 'diffusion2': # 6 - 3 conditionat de gt
                        diff_sample = get_diffusion_sample(diffusion_prediction_net, window_reshaped, k)
                        loss = l(diff_sample, window_reshaped)

                l1s.append(loss)
        if training_mode == 'both':
            return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()
        return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
    parser.add_argument('--dataset',metavar='-d',type=str,required=False,default='SMD',help="dataset")
    parser.add_argument('--file',metavar='-f',type=str,required=False,default=None,help="dataset")
    parser.add_argument('--model', metavar='-m', type=str, required=False, default='GRD', help="model name") # MTAD-GAT
    parser.add_argument('--training', metavar='-t', type=str, required=False, default='both', help="model to train")
    parser.add_argument('--anomaly_score', metavar='-t', type=str, required=False, default='diffusion', help="anomaly score")
    parser.add_argument('--lr', metavar='-lr', type=str, required=False, default='1e-3', help="lerning rate")
    parser.add_argument('--window_size', metavar='-ws', type=str, required=False, default='100', help="window size")
    parser.add_argument('--p1', metavar='-p1', type=float, required=False, default='1', help="p1")
    parser.add_argument('--p2', metavar='-p2', type=float, required=False, default='1', help="p2")
    parser.add_argument('--k', metavar='-k', type=int, required=False, default='1', help="number of diff samples")
    parser.add_argument('--v', metavar='-v', type=bool, required=False, default=True, help="verbose")
    # parser.add_argument('--test_only', metavar='-t', type=bool, required=False, default=False, help="test_only")
    parser.add_argument('--batch_size', metavar='-t', type=int, required=False, default=32, help="batch_size")
    parser.add_argument('--num_epochs', metavar='-t', type=int, required=False, default=25, help="num_epochs") # 新添加
    parser.add_argument('--diff_lambda', metavar='-t', type=float, required=False, default=0.1, help="diff_lambda")
    parser.add_argument('--noise_steps', metavar='-t', type=int, required=False, default=100, help="noise_steps") # 训练diffusion时的加噪最大步骤
    parser.add_argument('--denoise_steps', metavar='-t', type=int, required=False, default=20, help="denoise_steps") # 做inference时的加噪然后去噪的固定步骤
    parser.add_argument('--group', metavar='-t', type=str, required=False, default='my_computer_GRD_version2', help="wandb group") # 用于wandb分组
    parser.add_argument('--test_only', metavar='-t', type=bool, required=False, default=True,help="train new model or not")
    parser.add_argument('--id', metavar='-t', type=int, required=False, default=0, help="experiment id for multiple runs")
    args = parser.parse_args()

    config = {
        "dataset": args.dataset,
        "file": args.file,
        "training_mode": args.training,
        "model": args.model,
        "learning_rate": float(args.lr),
        "window_size": int(args.window_size),
        "lambda": args.diff_lambda,
        "noise_steps": args.noise_steps,
        "batch_size": args.batch_size,
        "num_epochs":int(args.num_epochs)
    }
    anomaly_scores = [args.anomaly_score]

    experiment = 'GRD_SMD_1_1_GPU_v2' 
    wandb.init(project="try_GRD", config=config, group=args.group) 
    wandb.run.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") 

    dataset_name = args.dataset
    part = None if not args.file else args.file # None
    training_mode = 'both' if not args.training else args.training # 'both'
    anomaly_score = None if not args.anomaly_score else args.anomaly_score # 'diffusion'
    window_size = int(args.window_size) # 100
    synthetic_datasets = ['point_global', 'point_contextual', 'pattern_shapelet', 'pattern_seasonal', 'pattern_trend','all_types', 'pattern_trendv2']

    train_loader, test_loader, validation_loader, labels = load_dataset(dataset_name, part)
    model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
        load_model(args.lr, args.window_size,labels.shape[1], args.batch_size,args.noise_steps, args.denoise_steps)
    # 将模型放入device
    model = model.to(device)
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)
    # 获取整个的数据集
    trainD, testD, validationD = next(iter(train_loader)), next(iter(test_loader)), next(iter(validation_loader))
    trainO, testO, validationO = trainD, testD, validationD
    if args.v:
        print(f'\ntrainD.shape: {trainD.shape}') # torch.Size([20000, 5])
        print(f'testD.shape: {testD.shape}') # torch.Size([20000, 5])
        print(f'validationD.shape: {validationD.shape}') # torch.Size([10000, 5])
        print(f'labels.shape: {labels.shape}') # (20000, 5)
    # 特征数(即时序数据的变量数) point_global数据集的通道数为5
    feats=labels.shape[1]
    # 按照窗口大小进行切分(无重叠) # trainD=tensor(200,100,5)  testD=tensor(200,100,5), validationD=tensor(100,100,5)
    trainD, testD, validationD = convert_to_windows(trainD, window_size), convert_to_windows(testD,window_size), convert_to_windows(validationD, window_size) # 返回的shape:(窗口数量,窗口大小,通道变量数)

    num_epochs = int(args.num_epochs) # 训练轮数
    epoch = -1
    e = epoch + 1
    # start = time()

    max_roc_scores = [[0, 0, 0]] * 6
    max_f1_scores = [[0, 0, 0]] * 6
    roc_scores = []
    f1_scores = []
    f1_max = 0
    roc_max = 0
    validation_thresh = 0
    best_val_loss = float('inf') # 初始验证集上的损失
    # 开始训练
    if not args.test_only:
        wandb.watch(model,log='all',log_freq=100,log_graph=True) # 监控模型的梯度和参数
        start_time = datetime.datetime.now() # 记录训练开始时间
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            train_loss, ae_loss, diff_loss = backprop(e, model, diffusion_training_net, diffusion_prediction_net, trainD, args.diff_lambda, optimizer, scheduler, training_mode, anomaly_score, args.k)
            # 一轮训练结束后开始记录
            wandb.log({'sum_loss_train': train_loss, 'ae_loss_train': ae_loss, 'diff_loss_train': diff_loss, 'epoch': e}, step=e)
            # 在验证集上进行测试
            if ae_loss + diff_loss < 0.15:
                loss0, val_loss, ae_loss_val, diff_loss_val, samples, recons = backprop(e, model, diffusion_training_net,
                                                                                    diffusion_prediction_net,
                                                                                    validationD, args.diff_lambda,
                                                                                    optimizer, scheduler, training_mode='both',
                                                                                    anomaly_score='diffusion', k=args.k, training=False)
                wandb.log({'sum_loss_val': val_loss, 'ae_loss_val': ae_loss_val, 'diff_loss_val': diff_loss_val, 'epoch': e}, step=e)
                # 如果验证损失有改善，则保存模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model, experiment, diffusion_training_net, optimizer, None,  e, diff_loss, ae_loss)
        
        end_time = datetime.datetime.now() # 记录训练结束时间
        training_time = end_time - start_time # 计算训练时间
        print(f"训练结束了!此次训练总时间为：{training_time}")
        # 训练完后将训练结果上传到wandb云端进行保存
        artifact = wandb.Artifact(name='GRD_SMD',type='Try')
        artifact.add_dir('./anomaly-mts') # 添加文件夹
        wandb.log_artifact(artifact) # 上传对象
        print("训练数据已经成功上传到wandb云端!")


    if args.test_only:
        # 在测试集上进行测试
        # 加载模型
        model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
            load_model(args.lr, args.window_size, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps)
        # 加载之前训练好的模型参数
        model, diffusion_training_net = load_from_checkpoint(training_mode, experiment, model, diffusion_training_net)
        # 将模型放入设备
        model = model.to(device)
        diffusion_training_net = diffusion_training_net.to(device)
        diffusion_prediction_net = diffusion_prediction_net.to(device)
        test_start_time = datetime.datetime.now() # 开始时间
        # 将测试集放入模型，得到异常分数，测试总损失，重建损失，扩散损失，经扩散后的数据，重建后的数据。
        loss0, test_loss, ae_loss_test, diff_loss_test, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode='both', anomaly_score=args.anomaly_score, k=args.k, training=False)
        test_end_time = datetime.datetime.now() # 结束时间
        inference_time = test_end_time - test_start_time # 计算推测时间
        print(f"GRD模型推测总时间为:{inference_time}!")
        loss0 = loss0.reshape(-1, feats)
        lossFinal = np.mean(np.array(loss0), axis=1) # 得到异常分数
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0  # 实际标签 ，一维数据：（28479,）
        # 保存test上的检测结果
        save_test_result = False
        if save_test_result:
            np.save(r'dqb_test_result/try/lossFinal.npy', lossFinal)
            np.save(r'dqb_test_result/try/labelsFinal.npy', labelsFinal)
            np.save(r'dqb_test_result/try/samples.npy', samples)
            np.save(r'dqb_test_result/try/recons.npy', recons)



        POT = False
        Epsilon = False
        For2 = True
        plot_save = False # 是否画图并保存
        threshold_dict = {} # 创建一个字典用来存放各个方法所选择的阈值

        if POT or Epsilon:
            # POT算法以及Epsilon算法需要之前的训练数据作为依据
            lossT, _, _, _, _, _ = backprop(0, model, diffusion_training_net, diffusion_prediction_net, trainD, args.diff_lambda, optimizer, scheduler, training_mode='both', anomaly_score=args.anomaly_score, k=args.k, training=False)
            lossT = lossT.reshape(-1, feats)
            lossTFinal = np.mean(np.array(lossT), axis=1)  # 得到异常分
            lossTFinal = lossTFinal[0:len(labelsFinal)] # 使得异常分数的长度和labelsFinal相同
            lossFinal = lossFinal[0:len(labelsFinal)]
            if POT:
                # 使用POT方法选择阈值，并计算F1分数等指标(point-adjust)
                result, pred= pot_eval(lossTFinal, lossFinal, labelsFinal)
                pot_threshold = result['threshold']
                threshold_dict['pot_threshold'] = pot_threshold

                # 再用POT选择阈值后,计算PA%K
                f1s_pot = []
                fprs_pot = []
                tprs_pot = []
                # preds_pot = []
                # k：0、1、2、....、100
                for k in range(100 + 1):
                    adjusted_preds = pak.pak(lossFinal, labelsFinal, pot_threshold, k=k)
                    f1 = f1_score(labelsFinal, adjusted_preds)
                    f1s_pot.append(f1)

                    fpr, tpr = get_fp_tp_rate(adjusted_preds, labelsFinal)
                    fprs_pot.append(fpr)
                    tprs_pot.append(tpr)

                    #preds_pot.append(adjusted_preds)
                ks = [k / 100 for k in range(0, 100 + 1)]
                area_under_f1 = auc(ks, f1s_pot) # f1_auc,新的一个指标
                print('result: ',result)
                print(f"使用POT方法在测试集上所选择的阈值是：{result['threshold']},得到的f1分数是:{result['f1']},roc:{result['ROC/AUC']},得到的f1_auc是：{area_under_f1}")
                # 画图并保存
                if plot_save:
                    for dim in range(0, feats):
                        fig = dplotter(testD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), dim=dim,pred=pred)
                        print(f'第{dim}维的数据图已经画出！')
            if Epsilon:
                # 使用Epsilon方法选择阈值，并计算F1分数等指标(point-adjust)
                e_eval = epsilon_eval(lossTFinal, lossFinal, labelsFinal, reg_level=1)
                epsilon_threshold = e_eval['threshold']
                threshold_dict['epsilon_threshold'] = epsilon_threshold

                # 再用POT选择阈值后,计算PA%K
                f1s_epsilon = []
                fprs_epsilon = []
                tprs_epsilon = []
                # preds_pot = []
                # k：0、1、2、....、100
                for k in range(100 + 1):
                    adjusted_preds = pak.pak(lossFinal, labelsFinal, epsilon_threshold, k=k)
                    f1 = f1_score(labelsFinal, adjusted_preds)
                    f1s_epsilon.append(f1)

                    fpr, tpr = get_fp_tp_rate(adjusted_preds, labelsFinal)
                    fprs_epsilon.append(fpr)
                    tprs_epsilon.append(tpr)

                    # preds_epsilon.append(adjusted_preds)
                ks = [k / 100 for k in range(0, 100 + 1)]
                area_under_f1_epsilon = auc(ks, f1s_epsilon)  # f1_auc,新的一个指标
                print('result: ', e_eval)
                print(f"使用Epsilon方法在测试集上所选择的阈值是：{e_eval['threshold']},得到的f1分数是:{e_eval['f1']},得到的f1_auc是：{area_under_f1_epsilon}")


        if For2:
            # 双for循环寻找阈值和k值，并计算F1_AUC
            result, _, _ = evaluate(lossFinal, labelsFinal)
            result_roc = result["ROC/AUC"] # 不同的阈值和不同的k值所形成的AUC值
            result_f1 = result["f1"] # f1_k
            for2_threshold = result['threshold'] # f1_k对应的阈值
            threshold_dict['for2_threshold'] = for2_threshold
            np.save('preds_For2.npy',result["preds"]) # 保存预测结果
            print(f"使用双for循环在测试集上所选择的阈值是:{for2_threshold},得到的f1分数是:{result_f1},roc:{result_roc},点调整协议后的最佳f1:{result['f1_max']}")
            # 画图并保存
            if plot_save:
                for dim in range(0, feats):
                    fig = dplotter(testD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), dim=dim)
                    print(f'第{dim}维的数据图已经画出！')

        print(threshold_dict) # 打印出阈值字典
        # 提取字典中的值
        values = threshold_dict.values()
        # 计算均值
        threshold_average = sum(values) / len(values)
        print(f"平均阈值是：{threshold_average}")






    wandb.finish()



















