import torch
import json
from pypinyin import lazy_pinyin, Style
from .PredictUtils import predict

def zhuyin_predict(predict_sentence, model, args):
    """
    从注音开始预测类别，输入是中文，中间会转成注音
    :param predict_sentence: 待预测句子
    :param model: 分类模型
    :return: 类别
    """
    zhuyin_lists = []
    for line in predict_sentence:
        line = lazy_pinyin(line, style=Style.BOPOMOFO)
        zhuyin = " ".join(line)
        zhuyin_lists.append(zhuyin)
    predict(predict_sentence, zhuyin_lists, model, args)


if __name__ == "__main__":
    pass
