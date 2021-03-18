import torch
from torch.utils import data as Data
import data
import train
import models
from utils.PredictUtils import predict
from utils import TransformONNX
import config
from torch.utils.tensorboard import SummaryWriter
from utils.zhuyin_test import zhuyin_predict

args = config.LSTMConfig()

if args.record:
    writer = SummaryWriter()
else:
    writer = None

train_data, train_length, train_label = data.get_pad_data("DataSet/train_zhuyin.json", args, no_noise=False)
train_data_no_noise, train_length_no_noise, train_label_no_noise = data.get_pad_data("DataSet/train_zhuyin.json", args, no_noise=True)
test_data, test_length, test_label = data.get_pad_data("DataSet/test_zhuyin.json", args, no_noise=True)

train_data = torch.tensor(train_data + train_data_no_noise, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

batch_size = args.batch_size
train_loader = Data.DataLoader(data.MyDataSet(train_data, train_label + train_label_no_noise, train_length + train_length_no_noise), batch_size, True)
test_loader = Data.DataLoader(data.MyDataSet(test_data, test_label, test_length), batch_size, True)


model = getattr(models, "LSTM_ATT")(args)


# train.train(train_loader, test_loader, model, args, writer)

zhuyin_predict(["空调风速大一点", "空调自动设定", "今天天气不好打开空调除湿"], model, args)

zhuyinlist = ["ㄎㄨㄥ ㄊㄧㄠˊ ㄈㄥ ㄙㄨˋ ㄧㄚ ㄎㄨˋ"]
predict(zhuyinlist, model, args)


