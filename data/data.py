from torch.utils import data as Data
import json
import numpy as np


def get_pad_data(data_file, args, no_noise=True):
    """
    获取pad后的数据
    :param data_file: 原始文件
    :return: pad_data, data_line, label
    """
    text, domain, command, value = turn_data_2_num(data_file, args, no_noise)
    data_line = [len(i) for i in text]
    # pad补零
    for i in text:
        for j in range(args.max_line - len(i)):
            i.append(0)
    assert len(text) == len(data_line)
    return text, data_line, domain, command, value


def turn_data_2_num(data_file, args, no_noise=True):
    """
    将数据转为数字
    :param data_file: 原始数据
    :return: 句子编码，标签编码
    """
    with open(args.label_file, "r", encoding="UTF-8") as f:
        label_2_id = json.load(f)
    with open(args.id_file, "r", encoding="UTF-8") as f:
        word_2_num = json.load(f)
    word_2_num_padding = {i[0]: i[1] + 1 for i in word_2_num.items()}
    word_2_num_padding["PAD"] = 0
    data_text = []
    data_domain = []
    data_command = []
    data_value = []

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
        data_domain.append(label_2_id[0][i["domain"]])
        data_command.append(label_2_id[1][i["Command_word"]])
        data_value.append(label_2_id[2][i["value"]])

    return data_text, data_domain, data_command, data_value


class MyDataSet(Data.Dataset):
    def __init__(self, data, length, domain, command, value):
        super(MyDataSet, self).__init__()
        self.data = data
        self.length = length
        self.domain = domain
        self.command = command
        self.value = value

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.length[item], [self.domain[item], self.command[item], self.value[item]]
