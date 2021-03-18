from utils import DataUtils
import json
import torch

def predict(sentences, model, args):
    label_2_id = DataUtils.get_label_2_id("DataSet/Command_words.json")
    id_2_label = {i[1]: i[0] for i in label_2_id.items()}

    with open(args.id_file, "r", encoding="UTF-8") as f:
        word_2_num = json.load(f)
    word_2_num_padding = {i[0]: i[1] + 1 for i in word_2_num.items()}
    word_2_num_padding["PAD"] = 0
    sentences_num = []
    for line in sentences:
        sentence_num = []
        if args.use_syllable:
            for word in line:
                sentence_num.append(word_2_num_padding.get(word, word_2_num_padding["UNK"]))
        else:
            for word in line.split():
                sentence_num.append(word_2_num_padding.get(word, word_2_num_padding["UNK"]))
        sentences_num.append(sentence_num)

    # pad补零
    length = []
    for line in sentences_num:
        length.append(len(line))
        for j in range(args.max_line - len(line)):
            line.append(0)

    sentences_num = torch.tensor([sentences_num], dtype=torch.long).squeeze(0)
    model.load("checkpoints/{}_best.pth".format(args.model_name))
    model.eval()
    if args.cuda:
        device = torch.device("cuda:0")
        model.cuda(device)
        if device != "cpu":
            sentences_num = sentences_num.cuda()

    if args.RNN_flag:
        logit = model(sentences_num, length)
    else:
        logit = model(sentences_num)

    results = torch.max(logit, 1)[1]
    labels = []
    for result in results:
        labels.append(id_2_label[result.item()])
    for i in range(len(sentences)):
        print("注音：{}, predict:{}".format(sentences[i], labels[i]))
