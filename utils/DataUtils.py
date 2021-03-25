import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import sys
from pypinyin import lazy_pinyin, Style


def get_label_2_id(label_file):
    """
    获取label_2_id
    :param label_file: 标签文件
    :return: label_2_id
    """
    with open(label_file, "r", encoding="UTF-8") as f:
        command_data = json.load(f)
    command_value_list = []
    for i in range(len(command_data)):
        command_value_list.append([command_data[i]["Command_word"]])
        for j in command_data[i]["value"]:
            command_value_list[i].append(j)
    label_list = []
    for i in command_value_list[0:-1]:
        for j in i[1::]:
            label = i[0] + j
            label_list.append(label)
    label_list.append("NoneNone")

    id_list = [i for i in range(len(label_list))]
    label_2_id = dict(zip(label_list, id_list))
    return label_2_id


def plot_data_distribute(data):
    """
    画出数据分布
    :param data: 数据路径 要求json格式
    :return:None
    """
    with open(data, "r", encoding="UTF-8") as f:
        data = json.load(f)
    label_2_id = get_label_2_id("../DataSet/Command_words.json")
    data_label = np.zeros(len(label_2_id))
    print(len(data))
    for i in data:
        i_label = i["Command_word"] + i["value"]
        data_label[label_2_id[i_label]] += 1
    plt.bar(range(len(data_label)), data_label)
    plt.show()


def divide_data(origin_data_file, train_data_file, test_data_file):
    """
    数据集划分程序
    :param test_data_file: 测试集
    :param train_data_file: 训练集
    :param origin_data_file: 原始数据集
    :return: None
    """
    with open(origin_data_file, "r", encoding="UTF-8") as f:
        data = json.load(f)

    # 获取label，并构造label_2_id
    data_label = []
    for i in data:
        i_label = i["Command_word"] + i["value"]
        data_label.append(i_label)
    data_label_counter = Counter(data_label)
    label_list = [i for i in data_label_counter]
    id_list = [i for i in range(len(label_list))]
    label_2_id = dict(zip(label_list, id_list))

    # 将数据按label重排
    new_data = [[] for i in label_list]
    for i in data:
        i_label = i["Command_word"] + i["value"]
        new_data[label_2_id[i_label]].append(i)

    # 构建训练集和测试集
    train_data = []
    test_data = []
    for i in new_data:
        index = np.arange(len(i))
        train_index = np.random.choice(index, round(len(i) * 0.7), replace=False)
        test_index = [i for i in index if i not in train_index]
        train_data_ = []
        test_data_ = []
        for j in train_index:
            train_data_.append(i[j])
        for k in test_index:
            test_data_.append(i[k])
        train_data += train_data_
        test_data += test_data_

    # 保存数据集
    with open(train_data_file, "w", encoding="UTF-8") as f:
        json.dump(train_data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))
    with open(test_data_file, "w", encoding="UTF-8") as f:
        json.dump(test_data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))


def convert_to_zhuyin(file, target_file):
    with open(file, "r", encoding="UTF-8") as f:
        word_data = json.load(f)
    for i in range(len(word_data)):
        line = word_data[i]["text"]
        line = lazy_pinyin(line, style=Style.BOPOMOFO)
        word_data[i]["text"] = " ".join(line)
    with open(target_file, "w", encoding='UTF-8') as f:
        json.dump(word_data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))
    print('convert finish')


def zhuyin_2_num(file, target_file):
    with open(file, "r", encoding="UTF-8") as f:
        zhuyin_data = f.read().split()
    zhuyin = {}
    j = 0
    for i in zhuyin_data:
        zhuyin[i] = j
        j = j + 1
    with open(target_file, "w") as f:
        json.dump(zhuyin, f)
        print("save finish")


def add_domain(file):
    with open(file, "r", encoding="UTF-8") as f:
        data = json.load(f)
    for i in data:
        i["domain"] = "灯"
    with open(file, "w", encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    # divide_data("../DataSet/air_conditioner_data.json", "../DataSet/train0.json", "../DataSet/test0.json")
    # convert_to_zhuyin("../DataSet/test0.json", "../DataSet/test_zhuyin0.json")
    # convert_to_zhuyin("../DataSet/train0.json", "../DataSet/train_zhuyin0.json")
    # label_2_id = get_label_2_id("../DataSet/Command_words.json")
    # plot_data_distribute("../DataSet/air_conditioner_data.json")
    # print(label_2_id)
    pass
