import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os
import torch
import numpy as np



def dplotter(test,lossFinal,labelsFinal,result,recons,samples,dim,pred=None):
    """"
    test:测试集原始数据,tensor格式 -例（28479，38）
    lossFinal：测试集上的异常得分 ，一维数据 例（28479，）
    labelsFinal：测试集标签，一维-例（28479，）
    result：
    recons：重建后的数据 ，二维
    samples：扩散重建后的数据，二维
    dim：画第几维的数据
    pred:得知阈值后的检测结果：例如{ndarray：(28479,)[False,False,...,True,True,...]}
    """
    timestamps = len(labelsFinal) # 时间步长
    print(f"测试集时间步长为：{timestamps}")
    # 取出要画出的维度
    test_dim = test[:, dim]  # 例子中：test_dim:{Tensor:(28500,)}
    recons_dim = recons[:,dim] # 例子中：recons_dim:{ndarray:(28500,)}
    samples_dim = samples[:,dim] # # 例子中：samples_dim:{ndarray:(28500,)}
    if pred is None:
        # 如果pred是None，则使用的是双for循环方法
        preds = result['preds']  # 例子中，preds:{ndarray:(28479,)}
        thresh = result['thresh_max']  # 获得阈值
    else:
        # 如果pred不是None，则使用的是POT法
        preds = pred
        thresh = result['threshold']  # 获得阈值

    text = 'SMD-1-1-dim1'
    TP = [1 if preds[i] and labelsFinal[i] else 0 for i in range(0, timestamps)]
    FP = [1 if preds[i] and not labelsFinal[i] else 0 for i in range(0, timestamps)]  # 误报
    FN = [1 if not preds[i] and labelsFinal[i] else 0 for i in range(0, timestamps)]  # 漏报

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    # 画出原始数据的第dim维数据曲线
    ax1.set_title(text, fontsize=7)
    ax1.plot(test_dim, linewidth=0.2)
    ax1.set_ylim(0, 1)  # 将ax1图的y轴范围设置为 (0,1)
    ax1.fill_between(np.arange(len(labelsFinal)),labelsFinal,color='red',alpha=0.3,linestyle='dashed',linewidth=0.1) # 标注异常
    # 画出第一次重建后的数据的第dim维
    ax2.plot(recons_dim,linewidth=0.2)
    ax2.set_ylim(0, 1)
    # 画出扩散重建后的数据的第dim维
    ax3.plot(samples_dim, linewidth=0.2)
    ax3.set_ylim(0, 1)
    # ax3.fill_between(np.arange(timestamps), TP, color='green', alpha=0.2, linestyle='dashed', linewidth=0.3, label='TP')
    # ax3.fill_between(np.arange(timestamps), FP, color='orange', alpha=0.3, linestyle='dashed', linewidth=0.3, label='FP')
    # ax3.fill_between(np.arange(timestamps), FN, color='blue', alpha=0.2, linestyle='dashed', linewidth=0.3, label='FN')
    # ax3.legend(loc=(-0.1, -0.2), borderaxespad=0, fontsize='xx-small')
    # 画出异常得分及阈值
    ax4.plot(lossFinal, linewidth=0.2)
    th = [thresh] * timestamps
    ax4.plot(th, '--', linewidth=0.2, alpha=0.5)
    ax4.set_xlabel('Timestamp')
    # 保存图像
    if pred is None:
        # 如果pred是None，则使用的是双for循环方法
        folder = './plots/3y29'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/test_dim_{dim}.pdf')  # 为了确保不失真，保存为pdf格式
    else:
        # 如果pred不是None，则使用的是POT法
        folder = './plots/POT'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/test_POT_dim_{dim}.pdf')  # 为了确保不失真，保存为pdf格式

    plt.close()
    return fig

















