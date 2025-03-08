import torch
import torch.nn as nn
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
'''递归预测函数 '''
def predict_future(model, scaler, initial_sequence, pred_steps=30,seq_length=80):
    """
    Args:
        model: 加载的模型
        scaler: 归一化器
        initial_sequence: 初始输入序列（未归一化，形状 [seq_length]）
        pred_steps: 预测步数
    Returns:
        predictions: 反归一化后的预测结果（形状 [pred_steps]）
    """
    # 归一化初始序列
    scaled_seq = scaler.transform(initial_sequence.reshape(-1, 1)).flatten()
    predictions = []
    for _ in range(pred_steps):
        # 转换为模型输入形状 (1, seq_length, 1)
        x = torch.tensor(scaled_seq, dtype=torch.float32).view(1, seq_length, 1)
        # 预测下一个点
        with torch.no_grad():
            pred = model(x).item()
        # 更新输入序列：移除第一个值，添加预测值
        scaled_seq = np.append(scaled_seq[1:], pred)
        # 保存预测结果（归一化的）
        predictions.append(pred)
    # 反归一化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions
def test():
    '''配置参数（需与训练时一致）'''
    seq_length = 80          # 时间步长（与训练时一致）
    pred_steps = 100          # 预测未来30天的价格
    current_dir = os.path.dirname(__file__)
    last_dir = os.path.dirname(current_dir)
    codelist = []
    
    '''定义模型结构（需与训练时完全一致）'''
    class StockLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            self.fc = nn.Linear(128, 1)
        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 128)
            c0 = torch.zeros(2, x.size(0), 128)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])
    
    '''读取数据代码列表'''
    with open(os.path.join(current_dir, 'data', 'datalist.csv'), newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            codelist.append(row[0])
    for stockcode in codelist:
        csv_dir = os.path.join(current_dir, 'data','rawdata', f'{stockcode}.csv')
        with open(csv_dir, 'r') as file:
            csv_reader = csv.reader(file)
            raw_data = np.array([float(row[1]) for row in csv_reader])  # 原始数据
        '''加载模型和归一化器'''
        model_path = os.path.join(current_dir,'output', 'model',f'{stockcode}.pth')  # 模型保存路径
        scaler_path = os.path.join(current_dir,'output', 'pkl',f'{stockcode}.pkl')         # 归一化器路径
        model = StockLSTM()
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            continue
        model.eval()  # 设置为评估模式
        scaler = joblib.load(scaler_path)  # 加载归一化器
        ''' 准备最新数据（示例）'''
        # 假设从CSV读取最新的 seq_length 天数据（未归一化）
        current_dir = os.path.dirname(__file__)
        last_dir = os.path.dirname(current_dir)
        with open(csv_dir, 'r') as f:
            reader = csv.reader(f)
            raw_data = [float(row[1]) for row in reader]

        # 提取最后 seq_length 天的数据
        initial_sequence = np.array(raw_data[-seq_length:])  # 形状: (seq_length,)
        '''执行预测'''
        predictions = predict_future(model, scaler, initial_sequence, pred_steps=pred_steps,seq_length = seq_length)
        print("未来30天预测价格:", predictions)
        '''可视化结果'''
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(figsize=(12, 6))
        plt.plot(raw_data, label='history data')
        plt.plot(np.arange(len(raw_data), len(raw_data) + pred_steps), predictions, label='predict output')
        plt.legend()
        plt.title("stockvalue predict")
        plt.savefig(os.path.join(current_dir,  'predict', f'{stockcode}.png'))
    print('测试结束')
    
test()