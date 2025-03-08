import torch
import torch.nn as nn
import os
import csv
import joblib 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from datetime import datetime
current_time = datetime.now()
    
formatted_time = current_time.strftime("%Y-%m-%d")
def count_csv_rows(file_path):

    """
    计算CSV文件的行数（不包括表头）
    
    :param file_path: CSV文件路径
    :return: 行数（不包括表头）
    """
    try:
        df = pd.read_csv(file_path)
        return len(df)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
def counttime(stockcode):# 计算上一次运行和几天的时间差值
    current_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(current_dir, 'output', 'lastruntime',f'{stockcode}flag.csv'), header=None)

    # 获取最后一行的日期并转换为 datetime 类型
    last_date_str = df.iloc[-1, 0]  # 获取最后一行的日期（假设日期在第一列）
    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')  # 转换为 datetime 类型

    # 获取今天的日期
    today = datetime.today()

    # 计算日期差值
    return (today - last_date).days

def mainx(seq_length):
    # 设置计算设备q
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # 超参数
    test_redio = 0.05
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATIO = 0.001
    EPOCHS = 50
    runsumtime = 0 # 计算总共要计算次数
    runtime = 0 # 计算已经运行次数
    # 数据加载和预处理（修改部分）
    current_dir = os.path.dirname(__file__)
    last_dir = os.path.dirname(current_dir)
    codelist = []
    outputlevellist = []
    csvfilename = os.path.join(current_dir, 'output', 'output.csv')
    costture = []
    # 读取股票代码列表
    with open(os.path.join(current_dir,  'data',  'datalist.csv'), newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            codelist.append(row[0])
    runsumtime = EPOCHS*len(codelist)
    for stockcode in codelist:
        try:# 根据上次执行时间计算出EPOCH
            EPOCHS = counttime(stockcode)
            # signal_emitter.text_print.emit(info_box,f'{stockcode}上次执行在{counttime(stockcode)}天前，将执行{EPOCHS}轮')
        except:
            EPOCHS = 10
            # signal_emitter.text_print.emit(info_box,f'{stockcode}没有执行记录，将执行{EPOCHS}轮')
        csv_dir = os.path.join(current_dir, 'data','rawdata',  f'{stockcode}.csv')
        try:
            if count_csv_rows(csv_dir) < 100:
                # signal_emitter.text_print.emit(abinfo_box,f'{stockcode}可获取的股票天数太少，不予考虑')
                continue
        except:
            # signal_emitter.text_print.emit(abinfo_box,f'{stockcode}在天数比较阶段出现问题')
            continue
           
        try:
            with open(csv_dir, 'r') as file:
                csv_reader = csv.reader(file)
                raw_data = np.array([float(row[1]) for row in csv_reader])  # 原始数据
        except:
            print(f'文件{csv_dir}打开失败')
            # signal_emitter.text_print.emit(abinfo_box,f'文件{csv_dir}打开失败')
            continue
        # 按时间顺序拆分训练集和测试集（未归一化）
        train_size = int(len(raw_data) * (1 - test_redio))
        train_raw = raw_data[:train_size]
        test_raw = raw_data[train_size:]

        # 归一化（仅用训练数据拟合）
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 防止某个文件没有数据
        try:
            train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
        except:
            print(f'{stockcode}数据异常')
            # signal_emitter.text_print.emit(abinfo_box,f'{stockcode}数据异常')
            continue

        joblib.dump(scaler, os.path.join(current_dir, 'output', 'pkl',f'{stockcode}.pkl'))  # 保存归一化器

        # 处理测试数据（使用训练集的归一化参数）
        test_scaled = scaler.transform(test_raw.reshape(-1, 1)).flatten()

        # 生成序列
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length - 1):
                xs.append(data[i:i+seq_length])
                ys.append(data[i+seq_length])
            return np.array(xs), np.array(ys)

        train_X, train_y = create_sequences(train_scaled, seq_length)
        test_X, test_y = create_sequences(test_scaled, seq_length)

        # 转换为张量并调整形状
        train_X = torch.Tensor(train_X).unsqueeze(-1).to(device)
        train_y = torch.Tensor(train_y).unsqueeze(-1).to(device)
        test_X = torch.Tensor(test_X).unsqueeze(-1).to(device)
        test_y = torch.Tensor(test_y).unsqueeze(-1).to(device)

        # 模型定义（保持不变）
        class StockLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=DROPOUT)
                self.fc = nn.Linear(HIDDEN_SIZE, 1)
            def forward(self, x):
                h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
                c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device) 
                out, _ = self.lstm(x, (h0, c0))
                return self.fc(out[:, -1, :])

        model = StockLSTM().to(device)  # 模型移动到设备
        if os.path.exists(os.path.join(current_dir, 'output', 'model', f'{stockcode}.pth')):# 读取训练过的模型
        # 加载保存的 checkpoint
            # signal_emitter.text_print.emit(info_box,f'{stockcode}读取曾训练模型')
            checkpoint = torch.load(os.path.join(current_dir, 'output', 'model', f'{stockcode}.pth'), weights_only=True)
            model.load_state_dict(checkpoint)
            model = model.to(device)
        else:
            model = StockLSTM().to(device)  # 模型移动到设备
            # signal_emitter.text_print.emit(info_box,f'{stockcode}创建新模型')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATIO)
        # print("train_X 形状:", train_X.shape)  # 期望输出类似 (N, 60, 1)
        # print("test_X 形状:", test_X.shape)    # 若输出类似 (M, 60)，则问题在此
        # 训练循环（添加损失记录）
        train_losses = []
        test_losses = []
        tryflag = 0
        tryflag2 = 0
        for epoch in range(EPOCHS):
            model.train()
            try:
                outputs = model(train_X)
            except:
                print(f'{stockcode}的格式有问题')
                # signal_emitter.text_print.emit(abinfo_box,f'{stockcode}的格式有问题')
                tryflag = 1
                break
            loss = criterion(outputs, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        test_preds = model(test_X).cpu()  # 移回CPU
                        tryflag2 = 0
                    except Exception as e:
                        print(f'{stockcode}的数据可能有问题，数据移动回失败：{e}')
                        # signal_emitter.text_print.emit(abinfo_box,f'{stockcode}的数据可能有问题，数据移动回失败')
                        tryflag2 == 1
                        break
                    test_loss = criterion(test_preds, test_y.cpu())
                test_losses.append(test_loss.item())
                print(f'{stockcode}Epoch [{epoch+1}/{EPOCHS}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, rate{runtime/runsumtime:.4f}')
                # signal_emitter.text_print.emit(info_box,f'{stockcode}Epoch [{epoch+1}/{EPOCHS}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        # 保存损失曲线
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.legend()
        plt.savefig(os.path.join(current_dir, 'output', 'loss', f'loss_{stockcode}.png'))
        plt.close()

        if tryflag == 0 and tryflag2 == 0:
            # 预测与反归一化
            model.eval()
            with torch.no_grad():
                try:
                    train_pred = model(train_X).cpu().numpy()  # 移回CPU
                    test_pred = model(test_X).cpu().numpy()     # 移回CPU
                except:
                    print(stockcode,'在预测里发生错误')
                    # signal_emitter.text_print.emit(abinfo_box,f'{stockcode}在预测里发生错误')
                    continue

            def inverse_transform(data):
                return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

            train_true = inverse_transform(train_y.cpu().numpy())
            train_pred = inverse_transform(train_pred)
            test_true = inverse_transform(test_y.cpu().numpy()) 
            test_pred = inverse_transform(test_pred)

            # 绘图
            plt.figure(figsize=(12,6))
            plt.plot(train_true, label='Train True')
            plt.plot(train_pred, label='Train Predict')
            plt.plot(np.arange(len(train_true), len(train_true)+len(test_true)), test_true, label='Test True')
            plt.plot(np.arange(len(train_true), len(train_true)+len(test_true)), test_pred, label='Test Predict')
            plt.legend()
            plt.savefig(os.path.join(current_dir, 'output', 'predict', f'{stockcode}.png'))
            # plt.show()
            plt.close()

            # 保存模型
            torch.save(model.state_dict(), os.path.join(current_dir, 'output', 'model', f'{stockcode}.pth'))

            # 计算评估指标（MAE）
            outputlevelloss = np.mean(np.abs(test_true - test_pred))
            outputlevellist.append(outputlevelloss)
            costture.append(stockcode)
            # print(f'模型效果评估（MAE，越小越好）: {outputlevelloss}')
            flagfilename = os.path.join(current_dir, 'output','lastruntime', f'{stockcode}flag.csv')# 记录模型上次被运行时什么时候
            runtime = runtime+1
            # pprogress_ms.pprogress_signal.emit(progress_bar,(runtime/len(codelist))*100)
            try:
                with open(flagfilename, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                # 如果文件最后一行没有换行符，则手动添加换行符
                if lines and not lines[-1].endswith('\n'):
                    lines[-1] += '\n'  # 添加换行
            except:
                with open(flagfilename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([formatted_time])  # formatted_time 是你要写入的时间
            # 追加新数据
            with open(flagfilename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([formatted_time])  # formatted_time 是你要写入的时间
    # 保存评估结果到CSV
    with open(csvfilename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for code, loss in zip(costture, outputlevellist):
            writer.writerow([code, loss])
     # 保存评估结果到CSV
     # 获取当前时间

    print('运行结束')
    # signal_emitter.text_print.emit(info_box,'运行结束')

mainx(80)