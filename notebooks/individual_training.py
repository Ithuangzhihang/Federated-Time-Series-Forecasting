# %% [markdown]
# ### In this notebook we perform individual training.
# In individual learning each base station has access only to it's private dataset.

# %%
import sys
import os

from pathlib import Path
# 解释：将当前目录的父目录加入到sys.path中，这样就可以在当前目录下导入父目录的模块
parent = Path(os.path.abspath("")).resolve().parents[0]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

# %%
import random

import numpy as np
import torch

from argparse import Namespace

# %%
# 解释
from ml.utils.data_utils import read_data, generate_time_lags, time_to_feature, handle_nans, to_Xy, \
    to_torch_dataset, to_timeseries_rep, assign_statistics, \
    to_train_val, scale_features, get_data_by_area, remove_identifiers, get_exogenous_data_by_area, handle_outliers

# %%
from ml.utils.train_utils import train, test

# %%
from ml.models.mlp import MLP
from ml.models.rnn import RNN
from ml.models.lstm import LSTM
from ml.models.gru import GRU
from ml.models.cnn import CNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder

# %%
args = Namespace(
    #数据相关参数
    data_path='../dataset/full_dataset.csv', # dataset  训练集路径
    data_path_test=['../dataset/ElBorn_test.csv'], # test dataset  测试集路径
    test_size=0.2, # validation size  验证集大小 数据划分为训练集和验证集时使用的比例,据的20%将用于验证，其余80%用于训练
    targets=['rnti_count', 'rb_down', 'rb_up', 'down', 'up'], # the target columns 需要预测的列，指定模型的目标输出。 
    #时间序列相关参数
    num_lags=10, # the number of past observations to feed as input  在时间序列预测中，过去的观测值对未来的预测有重要影响。这个参数指定了使用多少个过去的时间步作为输入。

    #数据处理相关参数
    filter_bs=None, # whether to use a single bs for training. It will be changed dynamically 是否只使用单个基站进行训练，在这里设置为None，表示不使用这个功能。
    identifier='District', # the column name that identifies a bs 标识基站的列名。

    nan_constant=0, # the constant to transform nan values 替换NaN值的常数。 
    x_scaler='minmax', # x_scaler 数据标准化的方法，使用最小-最大缩放（Min-Max Scaler）。
    y_scaler='minmax', # y_scaler 数据标准化的方法，使用最小-最大缩放（Min-Max Scaler）。
    outlier_detection=None, # whether to perform flooring and capping 是否进行异常值检测。在这里设置为None，表示不使用这个功能。

    #模型训练相关参数
    criterion='mse', # optimization criterion, mse or l1 损失函数，MSE（均方误差）用于衡量模型预测值与真实值之间的差异。
    epochs=150, # the number of maximum epochs 训练的最大迭代次数（轮数）。
    lr=0.001, # learning rate 学习率，控制模型权重更新的步长。
    optimizer='adam', # the optimizer, it can be sgd or adam 优化器，Adam是一种自适应学习率优化算法，广泛用于深度学习。
    batch_size=128, # the batch size to use 批量大小，指每次迭代中使用的样本数量。
    early_stopping=True, # whether to use early stopping 早停机制，防止过拟合。当验证集性能不再提升时提前停止训练。
    patience=50, # patience value for the early stopping parameter (if specified) 提前停止的耐心值，当验证集性能在连续50轮训练中没有提升时停止训练。
    max_grad_norm=0.0, # whether to clip grad norm 最大梯度范数，用于梯度裁剪。如果为0.0，表示不进行梯度裁剪。
    reg1=0.0, # l1 regularization L1正则化参数，防止过拟合。在这里设置为0.0，表示未使用L1正则化。
    reg2=0.0, # l2 regularization L2正则化参数，防止过拟合。在这里设置为0.0，表示未使用L2正则化。
    
    plot_history=True, # plot loss history 是否绘制训练过程中的损失和评价指标曲线图。
    #硬件和复现性相关参数
    cuda=True, # whether to use gpu 是否使用GPU进行计算。如果为True，则使用CUDA加速。
    
    seed=0, # reproducibility 随机种子，用于确保结果的可重复性。
    #其他参数
    assign_stats=None, # whether to use statistics as exogenous data, ["mean", "median", "std", "variance", "kurtosis", "skew"] 是否使用统计数据作为外生数据，选项包括["mean", "median", "std", "variance", "kurtosis", "skew"]。在这里设置为None，表示未使用这个功能。
    use_time_features=False # whether to use datetime features 是否使用日期时间特征。如果为True，表示在特征中加入时间信息（如时间戳）。
)

# %% [markdown]
# > You can define the base station to perform train on the filter_bs parameter and use it in block 12 or you can define the base station to block 12 explicitly 

# %%
print(f"Script arguments: {args}\n")

# %%
print(f"Script arguments: {args}\n")

# %%
import torch.version


device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
print("torch_version",torch.__version__)
print("args.cuda",args.cuda)
print("torch.cuda.is_available()",torch.cuda.is_available())
print(f"Using {device}")

# %%
# Outlier detection specification 异常值检测
if args.outlier_detection is not None:
    outlier_columns = ['rb_down', 'rb_up', 'down', 'up'] #需要进行异常值检测的列名
    outlier_kwargs = {"ElBorn": (10, 90), "LesCorts": (10, 90), "PobleSec": (5, 95)} #每个键是一个区域的名称，每个值是一个元组，用于指定该区域的异常值检测参数。"ElBorn": (10, 90)：表示对于ElBorn区域，异常值检测的阈值是10和90。
    args.outlier_columns = outlier_columns 
    args.outlier_kwargs = outlier_kwargs

# %%
def seed_all():
    # ensure reproducibility 确保结果的可重复性
    random.seed(args.seed) #设置Python标准库中的随机数生成器的种子为args.seed。这会影响使用random模块生成的所有随机数。
    np.random.seed(args.seed) #设置NumPy库的随机数生成器的种子为args.seed。这会影响使用numpy.random模块生成的所有随机数。
    torch.manual_seed(args.seed) #设置PyTorch库的随机数生成器的种子为args.seed。这会影响CPU上的所有PyTorch操作生成的随机数。
    torch.cuda.manual_seed_all(args.seed) #设置所有CUDA设备（即GPU）的随机数生成器的种子为args.seed。这会影响在GPU上执行的所有PyTorch操作生成的随机数。
    torch.backends.cudnn.deterministic = True #设置CuDNN后端为确定性模式，这意味着CuDNN将使用确定性的算法，从而确保相同的输入始终产生相同的输出。
    torch.backends.cudnn.benchmark = False #禁用CuDNN的benchmark模式。启用benchmark模式可能会导致不同的计算选择不同的算法，从而产生不同的结果，因此禁用它可以确保结果的一致性。

# %%
seed_all()

# %% [markdown]
# ### The preprocessing pipeline performed here for the base station specified in filter_bs argument
# Preprocessing inlcudes:
# 1. NaNs Handling NANs处理
# 2. Outliers Handling 异常值处理
# 3. Scaling Data 数据缩放
# 4. Generating time lags 生成时间滞后
# 5. Generating and importing exogenous data as features (time, statistics) (if applied) 生成和导入外生数据

# %%
def make_preprocessing(filter_bs=None): 
    #本函数接受一个可选参数‘filter_bs’,用于指定要过滤的基站。如果没有提供该参数，则处理所有基站的数据。
    """Preprocess a given .csv"""
    # read data 使用read_data函数读取指定路径（args.data_path）的CSV数据，并根据filter_bs过滤数据。
    df = read_data(args.data_path, filter_data=filter_bs)
    # handle nans 使用handle_nans函数处理数据中的缺失值。缺失值将被替换为args.nan_constant指定的常数。
    df = handle_nans(train_data=df, constant=args.nan_constant,
                     identifier=args.identifier)
    # split to train/validation 将数据集划分为训练集和验证集。具体的划分比例由预先定义的参数决定。
    train_data, val_data = to_train_val(df)
    
    # handle outliers (if specified) 如果启用了异常值检测（args.outlier_detection不为None），则使用handle_outliers函数处理训练数据中的异常值。
    if args.outlier_detection is not None:
        train_data = handle_outliers(df=train_data, columns=args.outlier_columns,
                                     identifier=args.identifier, kwargs=args.outlier_kwargs)
    
    # get X and y 将训练集和验证集中的特征（X）和目标变量（y）分离出来。
    X_train, X_val, y_train, y_val = to_Xy(train_data=train_data, val_data=val_data,
                                          targets=args.targets)
    # 对特征数据和目标变量分别进行缩放（标准化或归一化），以确保它们在相似的尺度上。缩放方法由args.x_scaler和args.y_scaler指定。
    # scale X
    X_train, X_val, x_scaler = scale_features(train_data=X_train, val_data=X_val,
                                             scaler=args.x_scaler, identifier=args.identifier)
    # scale y
    y_train, y_val, y_scaler = scale_features(train_data=y_train, val_data=y_val,
                                             scaler=args.y_scaler, identifier=args.identifier)
    
    # generate time lags 生成时间滞后特征，即在时间序列数据中，使用过去若干时间步的数据作为当前时间步的输入特征。
    X_train = generate_time_lags(X_train, args.num_lags)
    X_val = generate_time_lags(X_val, args.num_lags)
    y_train = generate_time_lags(y_train, args.num_lags, is_y=True)
    y_val = generate_time_lags(y_val, args.num_lags, is_y=True)
    
    #生成并导入时间特征（如时间戳、日期等）和统计特征（如均值、方差等），作为外生数据特征。
    # get datetime features as exogenous data
    date_time_df_train = time_to_feature(
        X_train, args.use_time_features, identifier=args.identifier
    )
    date_time_df_val = time_to_feature(
        X_val, args.use_time_features, identifier=args.identifier
    )
    
    # get statistics as exogenous data
    stats_df_train = assign_statistics(X_train, args.assign_stats, args.num_lags,
                                       targets=args.targets, identifier=args.identifier)
    stats_df_val = assign_statistics(X_val, args.assign_stats, args.num_lags, 
                                       targets=args.targets, identifier=args.identifier)
    
    #合并生成的外生特征（时间特征和统计特征），并去除重复的列。如果没有外生特征，则设为None。
    #exogenous_data_train 和 exogenous_data_val如果不为 None，这些是训练集和验证集的外生特征数据。
    # concat the exogenous features (if any) to a single dataframe
    if date_time_df_train is not None or stats_df_train is not None:
        exogenous_data_train = pd.concat([date_time_df_train, stats_df_train], axis=1)
        # remove duplicate columns (if any)
        exogenous_data_train = exogenous_data_train.loc[:, ~exogenous_data_train.columns.duplicated()].copy()
        assert len(exogenous_data_train) == len(X_train) == len(y_train)
    else:
        exogenous_data_train = None
    if date_time_df_val is not None or stats_df_val is not None:
        exogenous_data_val = pd.concat([date_time_df_val, stats_df_val], axis=1)
        exogenous_data_val = exogenous_data_val.loc[:, ~exogenous_data_val.columns.duplicated()].copy()
        assert len(exogenous_data_val) == len(X_val) == len(y_val)
    else:
        exogenous_data_val = None
        
    return X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler

# %%
# here exogenous_data_train and val are None.
X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler = make_preprocessing(
    filter_bs="LesCorts"
)

# %%
X_train.head()
#time：时间戳，表示数据记录的具体时间
#down：下行流量，单位可能是字节
#up：上行流量，单位可能是字节
#rnti_count：RNTI（Radio Network Temporary Identifier）计数，表示在指定时间内活跃的RNTI数量。
#mcs_down：下行调制和编码方案（MCS，Modulation and Coding Scheme）的平均值，表示下行链路的调制和编码效率。
#mcs_down_var：下行MCS的方差，表示下行链路调制和编码效率的变化程度
#mcs_up：上行调制和编码方案（MCS，Modulation and Coding Scheme）的平均值，表示上行链路的调制和编码效率。
#mcs_up_var：上行MCS的方差，表示上行链路调制和编码效率的变化程度
#rb_down：下行资源块（RB，Resource Block）的平均值，表示在指定时间内下行链路使用的资源块的平均数量。
#rb_down_var：下行资源块的方差，表示下行链路资源块使用数量的变化程度
#rb_up：上行资源块（RB，Resource Block）的平均值，表示在指定时间内上行链路使用的资源块的平均数量
#rb_up_var：上行资源块的方差，表示上行链路资源块使用数量的变化程度

# %%
#解释
X_val.head()


# %%
y_train.head()

# %%
x_scaler, y_scaler

# %% [markdown]
# ### Postprocessing Stage
# 
# In this stage we transform data in a way that can be fed into ML algorithms.

# %%
def make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler):
    #X_train, X_val: 训练集和验证集的特征数据。
    #y_train, y_val: 训练集和验证集的目标数据。
    #exogenous_data_train, exogenous_data_val: 训练集和验证集的外生数据。
    #x_scaler, y_scaler: 特征数据和目标数据的缩放器。
    """Make data ready to be fed into ml algorithms"""
    # if there are more than one specified areas, get the data per area 
    #检查训练数据集中是否有多个不同的区域（由 args.identifier 标识）。如果有多个区域，则调用 get_data_by_area 函数将数据按区域分割。
    if X_train[args.identifier].nunique() != 1:
        area_X_train, area_X_val, area_y_train, area_y_val = get_data_by_area(X_train, X_val,
                                                                              y_train, y_val, 
                                                                              identifier=args.identifier)
    else:
        area_X_train, area_X_val, area_y_train, area_y_val = None, None, None, None

    # Get the exogenous data per area. 获取每个区域的外生数据 如果存在外生数据，则将其按区域分割。
    if exogenous_data_train is not None:
        exogenous_data_train, exogenous_data_val = get_exogenous_data_by_area(exogenous_data_train,
                                                                              exogenous_data_val)
    # transform to np 将区域划分的数据转换为 NumPy 数组，并移除标识符列（如区域名称）。
    if area_X_train is not None:
        for area in area_X_train:
            tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val = remove_identifiers(
                area_X_train[area], area_y_train[area], area_X_val[area], area_y_val[area])
            tmp_X_train, tmp_y_train = tmp_X_train.to_numpy(), tmp_y_train.to_numpy()
            tmp_X_val, tmp_y_val = tmp_X_val.to_numpy(), tmp_y_val.to_numpy()
            area_X_train[area] = tmp_X_train
            area_X_val[area] = tmp_X_val
            area_y_train[area] = tmp_y_train
            area_y_val[area] = tmp_y_val
    
    if exogenous_data_train is not None:
        for area in exogenous_data_train:
            exogenous_data_train[area] = exogenous_data_train[area].to_numpy()
            exogenous_data_val[area] = exogenous_data_val[area].to_numpy()
    
    # remove identifiers from features, targets 对整体数据集，移除标识符列，并确保训练集和验证集的特征列数相同。
    X_train, y_train, X_val, y_val = remove_identifiers(X_train, y_train, X_val, y_val)
    assert len(X_train.columns) == len(X_val.columns)
    #计算特征数量，这里每个特征都有多个滞后期，所以除以滞后期的数量
    num_features = len(X_train.columns) // args.num_lags
    
    # to timeseries representation 将特征数据转换为时间序列表示，即构建时间滞后的特征集
    X_train = to_timeseries_rep(X_train.to_numpy(), num_lags=args.num_lags,
                                            num_features=num_features)
    X_val = to_timeseries_rep(X_val.to_numpy(), num_lags=args.num_lags,
                                          num_features=num_features)
    
    if area_X_train is not None:
        area_X_train = to_timeseries_rep(area_X_train, num_lags=args.num_lags,
                                                     num_features=num_features)
        area_X_val = to_timeseries_rep(area_X_val, num_lags=args.num_lags,
                                                   num_features=num_features)
    
    # transform targets to numpy 将目标数据转换为 NumPy 数组
    y_train, y_val = y_train.to_numpy(), y_val.to_numpy()
    
    # centralized (all) learning specific 在集中学习的情况下，将所有区域的外生数据合并成一个数据集
    if not args.filter_bs and exogenous_data_train is not None:
        exogenous_data_train_combined, exogenous_data_val_combined = [], []
        for area in exogenous_data_train:
            exogenous_data_train_combined.extend(exogenous_data_train[area])
            exogenous_data_val_combined.extend(exogenous_data_val[area])
        exogenous_data_train_combined = np.stack(exogenous_data_train_combined)
        exogenous_data_val_combined = np.stack(exogenous_data_val_combined)
        exogenous_data_train["all"] = exogenous_data_train_combined
        exogenous_data_val["all"] = exogenous_data_val_combined
    return X_train, X_val, y_train, y_val, area_X_train, area_X_val, area_y_train, area_y_val, exogenous_data_train, exogenous_data_val

# %%
X_train, X_val, y_train, y_val, area_X_train, area_X_val, area_y_train, area_y_val, exogenous_data_train, exogenous_data_val = make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler)

# %%
X_train[:2]

# %% [markdown]
# 

# %%
y_train[:2]

# %%
len(X_train), len(X_val)

# %% [markdown]
# ### Define the input dimensions for the model architecture

# %%
def get_input_dims(X_train, exogenous_data_train): 
    #计算模型输入的维度，X_train: 训练集的主输入数据（通常是一个多维数组）；exogenous_data_train: 训练集的外生数据（可以是 None 或包含外生特征的字典）。
    if args.model_name == "mlp":
        input_dim = X_train.shape[1] * X_train.shape[2] #如果模型是多层感知机（MLP），则将输入的所有特征展平成一个一维向量，因此输入维度是 X_train 的第二维度和第三维度的乘积（即 X_train.shape[1] * X_train.shape[2]）
    else:
        input_dim = X_train.shape[2] #对于其他模型（如卷积神经网络或循环神经网络），输入维度保持为 X_train 的第三维度（即 X_train.shape[2]）
    #计算外生数据的维度
    if exogenous_data_train is not None: #如果 exogenous_data_train 不为 None
        if len(exogenous_data_train) == 1: #如果外生数据中只有一个区域
            cid = next(iter(exogenous_data_train.keys()))
            exogenous_dim = exogenous_data_train[cid].shape[1] #从字典中获取该区域的维度。
        else:
            exogenous_dim = exogenous_data_train["all"].shape[1] #如果外生数据中有多个区域，则使用键为 "all" 的区域的维度
    else:
        exogenous_dim = 0 #如果 exogenous_data_train 为 None，则外生数据的维度为 0
    
    return input_dim, exogenous_dim #input_dim: 主输入数据的维度；exogenous_dim: 外生数据的维度

# %% [markdown]
# ### Initialize the model for training

# %%
def get_model(model: str,
              input_dim: int,
              out_dim: int,
              lags: int = 10,
              exogenous_dim: int = 0,
              seed=0):
    if model == "mlp":
        model = MLP(input_dim=input_dim, layer_units=[256, 128, 64], num_outputs=out_dim)
    elif model == "rnn":
        model = RNN(input_dim=input_dim, rnn_hidden_size=128, num_rnn_layers=1, rnn_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "lstm":
        model = LSTM(input_dim=input_dim, lstm_hidden_size=128, num_lstm_layers=1, lstm_dropout=0.0,
                     layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "gru":
        model = GRU(input_dim=input_dim, gru_hidden_size=128, num_gru_layers=1, gru_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "cnn":
        model = CNN(num_features=input_dim, lags=lags, exogenous_dim=exogenous_dim, out_dim=out_dim)
    elif model == "da_encoder_decoder":
        model = DualAttentionAutoEncoder(input_dim=input_dim, architecture="lstm", matrix_rep=True)
    else:
        raise NotImplementedError("Specified model is not implemented. Plese define your own model or choose one from ['mlp', 'rnn', 'lstm', 'gru', 'cnn', 'da_encoder_decoder']")
    return model

# %%
# define the model
args.model_name = "mlp"

input_dim, exogenous_dim = get_input_dims(X_train, exogenous_data_train)

print(input_dim, exogenous_dim)

model = get_model(model=args.model_name,
                  input_dim=input_dim,
                  out_dim=y_train.shape[1],
                  lags=args.num_lags,
                  exogenous_dim=exogenous_dim,
                  seed=args.seed)

# %%
model

# %% [markdown]
# ### The fit function used to train the model specified above

# %%
def fit(model, X_train, y_train, X_val, y_val, #model: 需要训练的模型；X_train, y_train: 训练数据和标签；X_val, y_val: 验证数据和标签
        exogenous_data_train=None, exogenous_data_val=None, #exogenous_data_train, exogenous_data_val: 训练和验证数据的外生特征
        idxs=[8, 3, 1, 10, 9], # the indices of our targets in X ；idxs: 目标变量在输入数据中的索引
        log_per=1): #log_per: 记录训练日志的频率
    
    # get exogenous data (if any)
    if exogenous_data_train is not None and len(exogenous_data_train) > 1:
        exogenous_data_train = exogenous_data_train["all"]
        exogenous_data_val = exogenous_data_val["all"]
    elif exogenous_data_train is not None and len(exogenous_data_train) == 1:
        cid = next(iter(exogenous_data_train.keys()))
        exogenous_data_train = exogenous_data_train[cid]
        exogenous_data_val = exogenous_data_val[cid]
    else:
        exogenous_data_train = None
        exogenous_data_val = None
    num_features = len(X_train[0][0]) #计算特征数量，X_train 是一个 3D 数组，形状为 (num_samples, num_lags, num_features)
    
    # to torch loader
    train_loader = to_torch_dataset(X_train, y_train,
                                    num_lags=args.num_lags,
                                    num_features=num_features,
                                    exogenous_data=exogenous_data_train,
                                    indices=idxs,
                                    batch_size=args.batch_size, 
                                    shuffle=False)
    val_loader = to_torch_dataset(X_val, y_val, 
                                  num_lags=args.num_lags,
                                  num_features=num_features,
                                  exogenous_data=exogenous_data_val,
                                  indices=idxs,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    
    # train the model
    model = train(model, 
                  train_loader, val_loader,
                  epochs=args.epochs,
                  optimizer=args.optimizer, lr=args.lr,
                  criterion=args.criterion,
                  early_stopping=args.early_stopping,
                  patience=args.patience,
                  plot_history=args.plot_history, 
                  device=device, log_per=log_per)
    
    
    return model

# %%
trained_model = fit(model, X_train, y_train, X_val, y_val)


