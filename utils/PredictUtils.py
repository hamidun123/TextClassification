from utils import DataUtils
import json
import torch

def predict(predict_sentence, sentences, model, args):
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
    for line in sentences_num:
        for j in range(args.max_line - len(line)):
            line.insert(0, 0)

    sentences_num = torch.tensor([sentences_num], dtype=torch.long).squeeze(0)
    model.load("checkpoints/{}_best.pth".format(args.model_name))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda(device)
    if device != "cpu":
        sentences_num = sentences_num.cuda()
    logit = model(sentences_num)

    results = torch.max(logit, 1)[1]
    labels = []
    for result in results:
        labels.append(id_2_label[result.item()])
    for i in range(len(predict_sentence)):
        print("sentence:{}, 注音：{}, predict:{}".format(predict_sentence[i], sentences[i], labels[i]))
