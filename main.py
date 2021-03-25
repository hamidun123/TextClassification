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

# train_data, train_length, train_label = data.get_pad_data("DataSet/train_zhuyin.json", args, no_noise=False)
train_data_no_noise, train_length_no_noise, train_label_no_noise = data.get_pad_data("DataSet/light_train_zhuyin.json", args, no_noise=True)
test_data, test_length, test_label = data.get_pad_data("DataSet/light_test_zhuyin.json", args, no_noise=True)

train_data = torch.tensor(train_data_no_noise, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

batch_size = args.batch_size
train_loader = Data.DataLoader(data.MyDataSet(train_data, train_label_no_noise, train_length_no_noise), batch_size, True)
test_loader = Data.DataLoader(data.MyDataSet(test_data, test_label, test_length), batch_size, True)


model = getattr(models, "LSTM_ATT")(args)


# train.train(train_loader, test_loader, model, args, writer)

zhuyin_predict(["灯打开了吗", "灯切换到彩光", "让灯再暖一点"], model, args)

zhuyinlist = ["ㄖㄤˋ ㄞㄥ ㄗㄞˋ ㄋㄨㄢˇ ㄧˋ ㄉㄢˇ"]
predict(zhuyinlist, model, args)


