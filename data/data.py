from torch.utils import data as Data
import json
from utils import get_label_2_id
import numpy as np


def get_pad_data(data_file, args, no_noise=True):
    """
    获取pad后的数据
    :param data_file: 原始文件
    :return: pad_data, data_line, label
    """
    text, label = turn_data_2_num(data_file, args, no_noise)
    data_line = [len(i) for i in text]
    # pad补零
    for i in text:
        for j in range(args.max_line - len(i)):
            i.append(0)
    assert len(text) == len(data_line)
    return text, data_line, label


def turn_data_2_num(data_file, args, no_noise=True):
    """
    将数据转为数字
    :param data_file: 原始数据
    :return: 句子编码，标签编码
    """
    label_2_id = get_label_2_id("DataSet/Command_words.json")
    with open(args.id_file, "r", encoding="UTF-8") as f:
        word_2_num = json.load(f)
    word_2_num_padding = {i[0]: i[1] + 1 for i in word_2_num.items()}
    word_2_num_padding["PAD"] = 0
    data_text = []
    data_label = []

    with open(data_file, "r", encoding="UTF-8") as f:
        data = json.load(f)

    for i in data:
        if args.use_syllable:
            i_text = [word_2_num_padding.get(word, word_2_num_padding["UNK"]) for word in i["text"]]
            if no_noise is False:
                if args.noise_rate != 0:
                    length = len(i_text)
                    id_list = np.random.randint(0, length, int(round(length * args.noise_rate)))
                    for id in id_list:
                        noise = np.random.randint(0, 41)
                        i_text[id] = noise
        else:
            i_text = [word_2_num_padding.get(word, word_2_num_padding["UNK"]) for word in i["text"].split()]
        data_text.append(i_text)
        data_label.append(label_2_id[i["Command_word"] + i["value"]])

    assert len(data_label) == len(data_text)
    return data_text, data_label


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        super(MyDataSet, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.label[item]
