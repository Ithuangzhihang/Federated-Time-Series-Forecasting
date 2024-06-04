# """
# Training pipeline.
# """

# import sys

# from pathlib import Path

# parent = Path(__file__).resolve().parents[2]
# if parent not in sys.path:
#     sys.path.insert(0, str(parent))

# import copy
# from logging import INFO
# from typing import Tuple, List, Union

# import torch
# from matplotlib import pyplot as plt

# from ml.utils.logger import log
# from ml.utils.helpers import get_optim, get_criterion, EarlyStopping, accumulate_metric


# def train(model: torch.nn.Module, # 要训练的PyTorch模型
#           train_loader, #训练数据加载器
#           test_loader, #测试数据加载器
#           epochs: int = 10, #训练的轮数
#           optimizer: str = "Adam", #优化器类型（如 "adam"）
#           lr: float = 1e-3, #学习率
#           reg1: float = 0., #L1正则化系数
#           reg2: float = 0., #L2正则化系数
#           max_grad_norm: float = 0., #最大梯度范数，用于梯度裁剪
#           criterion: str = "mse", #损失函数类型（如 "mse"）
#           early_stopping: bool = True, #是否启用提前停止
#           patience: int = 50, #提前停止的耐心（轮数）
#           plot_history: bool = False, #是否绘制训练历史
#           device="cuda", #训练设备（如 "cuda"）
#           fedprox_mu: float = 0., #FedProx正则化系数
#           log_per: int = 1, #日志记录频率
#           use_carbontracker: bool = False): #是否启用碳跟踪器
#     """Trains a neural network defined as torch module."""
#     #初始化最佳模型、最佳损失和最佳 epoch
#     best_model, best_loss, best_epoch = None, -1, -1
#     #初始化训练和测试的损失、RMSE 历史记录
#     train_loss_history, train_rmse_history = [], []
#     test_loss_history, test_rmse_history = [], []
#     #如果启用提前停止，则初始化 EarlyStopping 监控器
#     if early_stopping:
#         es_trace = True if log_per == 1 else False
#         monitor = EarlyStopping(patience, trace=es_trace)
#     cb_tracker = None
#     #如果启用碳跟踪器，则尝试导入 CarbonTracker 并初始化
#     if use_carbontracker:
#         try:
#             from carbontracker.tracker import CarbonTracker
#             cb_tracker = CarbonTracker(epochs=epochs, components="all", verbose=1)
#         except ImportError:
#             pass
#     #根据给定的名称和学习率获取优化器
#     optimizer = get_optim(model, optimizer, lr)
#     #根据给定的名称获取损失函数
#     criterion = get_criterion(criterion)
#     global_weight_collector = copy.deepcopy(list(model.parameters())) #深拷贝模型的参数，用于 FedProx 正则化
#     #模型设置和训练模式
#     for epoch in range(epochs):
#         if use_carbontracker and cb_tracker is not None: #如果启用了碳跟踪器，记录epoch开始
#             cb_tracker.epoch_start()
#         model.to(device) #将模型转移到指定设备（如GPU）
#         model.train() #将模型设置为训练模式
#         epoch_loss = [] #初始化epoch损失列表
#         for x, exogenous, y_hist, y in train_loader:
#             x, y = x.to(device), y.to(device) #将输入数据和目标数据转移到指定设备
#             y_hist = y_hist.to(device)
#             if exogenous is not None and len(exogenous) > 0:
#                 exogenous = exogenous.to(device) #如果存在外生变量，也将其转移到设备，否则设为 None
#             else:
#                 exogenous = None
#             optimizer.zero_grad() #梯度清零
#             y_pred = model(x, exogenous, device, y_hist) #前向传播计算预测值 y_pred
#             loss = criterion(y_pred, y) #计算损失 loss
#             #如果 fedprox_mu 大于 0，计算 FedProx 正则化项，并添加到损失中
#             if fedprox_mu > 0.:
#                 fedprox_reg = 0.
#                 for param_index, param in enumerate(model.parameters()):
#                     fedprox_reg += ((fedprox_mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
#                 loss += fedprox_reg
#             #如果 reg1 大于 0，计算 L1 正则化项，并添加到损失中
#             if reg1 > 0.:
#                 params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
#                 loss += reg1 * torch.norm(params, 1)
#             #如果 reg2 大于 0，计算 L2 正则化项，并添加到损失中
#             if reg2 > 0.:
#                 params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
#                 loss += reg2 * torch.norm(params, 2)
#             loss.backward() #反向传播计算梯度
#             if max_grad_norm > 0.: #如果 max_grad_norm 大于 0，对梯度进行裁剪
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             optimizer.step() #更新模型参数
#             epoch_loss.append(loss.item()) #将当前批次的损失添加到 epoch_loss 列表中
#         train_loss = sum(epoch_loss) / len(epoch_loss) #计算当前epoch的平均训练损失
#         #使用 test 函数计算训练和测试集上的损失和各种评价指标
#         _, train_mse, train_rmse, train_mae, train_r2, train_nrmse = test(model, train_loader,
#                                                                                       criterion, device)
#         test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = test(model, test_loader,
#                                                                                         criterion, device)
#         #每隔 log_per 轮记录一次日志，包含训练和测试的损失及评价指标
#         if (epoch + 1) % log_per == 0:
#             log(INFO, f"Epoch {epoch + 1} [Train]: loss {train_loss}, mse: {train_mse}, "
#                       f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
#             log(INFO, f"Epoch {epoch + 1} [Test]: loss {test_loss}, mse: {test_mse}, "
#                       f"rmse: {test_rmse}, mae {test_mae}, r2: {test_r2}, nrmse: {test_nrmse}")
#         #将当前epoch的训练和测试损失、RMSE添加到历史记录列表中
#         train_loss_history.append(train_mse)
#         train_rmse_history.append(train_rmse)
#         test_loss_history.append(test_mse)
#         test_rmse_history.append(test_rmse)
#         #提前停止和保存最佳模型
#         if early_stopping:
#             monitor(test_loss, model) #使用 monitor 更新监控器，判断是否需要提前停止
#             best_loss = abs(monitor.best_score)
#             best_model = monitor.best_model #保存当前最好的模型和损失
#             #根据不同情况更新最佳epoch
#             if epoch + 1 > patience:
#                 best_epoch = epoch + 1
#             elif epoch + 1 == epochs:
#                 best_epoch = epoch + 1 - monitor.counter
#             else:
#                 best_epoch = epoch + 1 - patience
#             #如果满足提前停止条件，记录日志并中断训练循环
#             if monitor.early_stop:
#                 log(INFO, "Early Stopping")
#                 break
#         else:
#             if best_loss == -1 or test_loss < best_loss: #检查当前epoch的测试损失是否为最佳，如果是，更新最佳模型和损失
#                 best_loss = test_loss
#                 best_model = copy.deepcopy(model)
#                 best_epoch = epoch + 1
#         if use_carbontracker and cb_tracker is not None:#如果启用了碳跟踪器，记录epoch结束
#             cb_tracker.epoch_end()

#     if plot_history:#如果 plot_history 为真，绘制训练和测试损失、RMSE历史曲线
#         plt.plot(train_loss_history, label="Train MSE")
#         plt.plot(test_loss_history, label="Test MSE")
#         plt.legend()
#         plt.show()
#         plt.close()

#         plt.plot(train_rmse_history, label="Train RMSE")
#         plt.plot(test_rmse_history, label="Test RMSE")
#         plt.legend()
#         plt.show()
#         plt.close()
#     #根据是否启用提前停止和训练的轮数，记录最佳损失和最佳epoch
#     if early_stopping and epochs > patience:
#         log(INFO, f"Best Loss: {best_loss}, Best epoch: {best_epoch}")
#     else:
#         log(INFO, f"Best Loss: {best_loss}")
#     return best_model #返回最佳模型

# #test 函数负责测试模型，计算和返回损失及各种评价指标
# def test(model, data, criterion, device="cuda") -> Union[
#     Tuple[float, float, float, float], List[torch.tensor], torch.tensor]:
#     """Tests a trained model."""
#     model.to(device) #将模型转移到指定设备
#     model.eval() #将模型设置为评估模式
#     y_true, y_pred = [], [] #初始化 y_true 和 y_pred 列表，用于存储真实值和预测值
#     loss = 0. #初始化 loss 为 0
#     with torch.no_grad(): #使用 torch.no_grad() 关闭梯度计算
#         for x, exogenous, y_hist, y in data:
#             x, y = x.to(device), y.to(device) #将输入数据和目标数据转移到指定设备
#             y_hist = y_hist.to(device)
#             #如果存在外生变量，也将其转移到设备，否则设为 None
#             if exogenous is not None and len(exogenous) > 0:
#                 exogenous = exogenous.to(device)
#             else:
#                 exogenous = None
#             #前向传播计算预测值 out
#             out = model(x, exogenous, device, y_hist)
#             #如果指定了损失函数，计算并累加损失
#             if criterion is not None:
#                 loss += criterion(out, y).item()
#             #将当前批次的真实值和预测值分别添加到 y_true 和 y_pred 列表中
#             y_true.extend(y)
#             y_pred.extend(out)

#     loss /= len(data.dataset) #计算平均损失
#     #将 y_true 和 y_pred 转换为 torch.tensor
#     y_true = torch.stack(y_true)
#     y_pred = torch.stack(y_pred)
#     #使用 accumulate_metric 计算各种评价指标：MSE, RMSE, MAE, R2, NRMSE
#     mse, rmse, mae, r2, nrmse = accumulate_metric(y_true.cpu(), y_pred.cpu())
#     #如果没有指定损失函数，返回评价指标和预测值，否则返回损失和评价指标
#     if criterion is None:
#         return mse, rmse, mae, r2, nrmse, y_pred

#     return loss, mse, rmse, mae, r2, nrmse
"""
Training pipeline.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import copy
from logging import INFO
from typing import Tuple, List, Union

import torch
from matplotlib import pyplot as plt

from ml.utils.logger import log
from ml.utils.helpers import get_optim, get_criterion, EarlyStopping, accumulate_metric


def train(model: torch.nn.Module, train_loader, test_loader, epochs: int = 10, optimizer: str = "adam", lr: float = 1e-3, reg1: float = 0., reg2: float = 0., max_grad_norm: float = 0., criterion: str = "mse", early_stopping: bool = True, patience: int = 50, plot_history: bool = False, device="cuda", fedprox_mu: float = 0., log_per: int = 1, use_carbontracker: bool = False):
    best_model, best_loss, best_epoch = None, -1, -1
    train_loss_history, train_rmse_history = [], []
    test_loss_history, test_rmse_history = [], []
    if early_stopping:
        es_trace = True if log_per == 1 else False
        monitor = EarlyStopping(patience, trace=es_trace)
    cb_tracker = None
    if use_carbontracker:
        try:
            from carbontracker.tracker import CarbonTracker
            cb_tracker = CarbonTracker(epochs=epochs, components="all", verbose=1)
        except ImportError:
            pass
    optimizer = get_optim(model, optimizer, lr)
    criterion = get_criterion(criterion)
    global_weight_collector = copy.deepcopy(list(model.parameters()))
    for epoch in range(epochs):
        if use_carbontracker and cb_tracker is not None:
            cb_tracker.epoch_start()
        model.to(device)
        model.train()
        epoch_loss = []
        for x, exogenous, y_hist, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            optimizer.zero_grad()
            y_pred = model(x, exogenous, device, y_hist)
            loss = criterion(y_pred, y)
            if fedprox_mu > 0.:
                fedprox_reg = 0.
                for param_index, param in enumerate(model.parameters()):
                    fedprox_reg += ((fedprox_mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fedprox_reg
            if reg1 > 0.:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                loss += reg1 * torch.norm(params, 1)
            if reg2 > 0.:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                loss += reg2 * torch.norm(params, 2)
            loss.backward()
            if max_grad_norm > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss.append(loss.item())
        train_loss = sum(epoch_loss) / len(epoch_loss)
        _, train_mse, train_rmse, train_mae, train_r2, train_nrmse = test(model, train_loader, criterion, device)
        test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = test(model, test_loader, criterion, device)
        if (epoch + 1) % log_per == 0:
            log(INFO, f"Epoch {epoch + 1} [Train]: loss {train_loss}, mse: {train_mse}, rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
            log(INFO, f"Epoch {epoch + 1} [Test]: loss {test_loss}, mse: {test_mse}, rmse: {test_rmse}, mae {test_mae}, r2: {test_r2}, nrmse: {test_nrmse}")
        train_loss_history.append(train_mse)
        train_rmse_history.append(train_rmse)
        test_loss_history.append(test_mse)
        test_rmse_history.append(test_rmse)

        if early_stopping:
            monitor(test_loss, model)
            best_loss = abs(monitor.best_score)
            best_model = monitor.best_model
            if epoch + 1 > patience:
                best_epoch = epoch + 1
            elif epoch + 1 == epochs:
                best_epoch = epoch + 1 - monitor.counter
            else:
                best_epoch = epoch + 1 - patience
            if monitor.early_stop:
                log(INFO, "Early Stopping")
                break
        else:
            if best_loss == -1 or test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch + 1
        if use_carbontracker and cb_tracker is not None:
            cb_tracker.epoch_end()

    if plot_history:
        plt.plot(train_loss_history, label="Train MSE")
        plt.plot(test_loss_history, label="Test MSE")
        plt.legend()
        plt.show()
        plt.close()

        plt.plot(train_rmse_history, label="Train RMSE")
        plt.plot(test_rmse_history, label="Test RMSE")
        plt.legend()
        plt.show()
        plt.close()
    if early_stopping and epochs > patience:
        log(INFO, f"Best Loss: {best_loss}, Best epoch: {best_epoch}")
    else:
        log(INFO, f"Best Loss: {best_loss}")
    return best_model

def test(model, data, criterion, device="cuda") -> Union[Tuple[float, float, float, float], List[torch.tensor], torch.tensor]:
    """Tests a trained model."""
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    loss = 0.
    with torch.no_grad():
        for x, exogenous, y_hist, y in data:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            out = model(x, exogenous, device, y_hist)
            if criterion is not None:
                loss += criterion(out, y).item()
            y_true.extend(y)
            y_pred.extend(out)

    loss /= len(data.dataset)

    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    mse, rmse, mae, r2, nrmse = accumulate_metric(y_true.cpu(), y_pred.cpu())
    if criterion is None:
        return mse, rmse, mae, r2, nrmse, y_pred

    return loss, mse, rmse, mae, r2, nrmse
