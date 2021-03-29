import torch
from torch.utils import data as Data
import data
import train
import models
from utils.PredictUtils import predict
import config
from torch.utils.tensorboard import SummaryWriter
from utils.zhuyin_test import zhuyin_predict

args = config.LSTMConfig()

if args.record:
    writer = SummaryWriter()
else:
    writer = None

train_data, train_length, train_domain, train_command, train_value = data.get_pad_data("DataSet/joint_data"
                                                                                       "/train_zhuyin.json", args,
                                                                                       no_noise=True)
test_data, test_length, test_domain, test_command, test_value = data.get_pad_data("DataSet/joint_data/test_zhuyin.json", args, no_noise=True)

train_data = torch.tensor(train_data, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

batch_size = args.batch_size
train_loader = Data.DataLoader(data.MyDataSet(train_data, train_length, train_domain, train_command, train_value), batch_size, True)
test_loader = Data.DataLoader(data.MyDataSet(test_data, test_length, test_domain, test_command, test_value), batch_size, True)


model = getattr(models, "LSTM_ATT")(args)


train.train(train_loader, test_loader, model, args, writer)

zhuyin_predict(["空调风速还可以再高一点吗", "让空调风速再强一点", "空调调高温度"], model, args)

zhuyinlist = ["ㄖㄤˋ ㄎㄨㄥ ㄊㄧㄠˊ ㄈㄥ ㄙㄨˋ ㄥˋ ㄉㄨㄥˋ"]
predict(zhuyinlist, model, args)


