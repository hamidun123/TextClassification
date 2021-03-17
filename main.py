import torch
from torch.utils import data as Data
import data
import train
import models
from utils import PredictUtils
from utils import TransformONNX
import config
from torch.utils.tensorboard import SummaryWriter
from utils.zhuyin_test import zhuyin_predict

args = config.LSTMConfig()

if args.record:
    writer = SummaryWriter()
else:
    writer = None

train_data, train_line, train_label = data.get_pad_data("DataSet/train_zhuyin.json", args)
test_data, test_line, test_label = data.get_pad_data("DataSet/test_zhuyin.json", args)

train_data = torch.tensor(train_data, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

batch_size = args.batch_size
train_loader = Data.DataLoader(data.MyDataSet(train_data, train_label), batch_size, True)
test_loader = Data.DataLoader(data.MyDataSet(test_data, test_label), batch_size, True)


textcnn_model = getattr(models, "LSTM_ATT")(args)


# train.train(train_loader, test_loader, textcnn_model, args, writer)

# TransformONNX()
zhuyin_predict(["今天太热了空调风力开到最大", "空调温度调高一点", "今天天气不好打开空调除湿"], textcnn_model, args)


