class BasicConfig(object):
    cuda = True

    label_number = 32
    use_syllable = False
    if use_syllable:
        word_number = 44
        max_line = 65
        id_file = "DataSet/zhuyin_2_id.json"
        name = "syllable"
    else:
        word_number = 1230
        id_file = "DataSet/zhuyin_word_2_id.json"
        max_line = 15
        name = "word"
    save_dir = "checkpoints/"


class TextCNNConfig(BasicConfig):
    model_name = "TextCNN" + "_" + BasicConfig.name

    # model parameter
    word_vector = 300
    input_channel = 1
    kernel_num = 250
    if BasicConfig.use_syllable:
        kernel_sizes = [4, 6, 9, 12]
    else:
        kernel_sizes = [3, 4, 5]
    dropout = 0.8
    attention = True
    reduction_ratio = 16

    # train parameter
    save_best = True
    batch_size = 20
    lr = 0.001
    weight_decay = 0.01
    epochs = 200
    early_stop = 100
    record = False


class DPCNNConfig(BasicConfig):
    model_name = "DPCNN" + "_" + BasicConfig.name

    # model parameter
    word_vector = 300

    # train parameter
    save_best = True
    batch_size = 40
    lr = 1e-3
    epochs = 150
    early_stop = 100
    weight_decay = 0.01
    record = False  # 是否记录训练日志


class TransformerConfig(BasicConfig):
    model_name = "Transformer" + "_" + BasicConfig.name

    embedding_pretrained = None
    # model parameter
    word_vector = 512
    pad_size = 65
    dropout = 0.8
    dim_model = 512
    num_head = 8
    hidden = 2048
    num_encoder = 6

    # train parameter
    save_best = True
    batch_size = 20
    lr = 5e-4
    epochs = 150
    early_stop = 100
    weight_decay = 0.01
    record = False  # 是否记录训练日志


class LSTMConfig(BasicConfig):
    model_name = "LSTM_ATT" + "_" + BasicConfig.name

    embedding_pretrained = None
    # model parameter
    word_vector = 256
    dropout = 0.4
    hidden_size = 128
    num_layers = 2
    hidden_size2 = 64

    # train parameter
    save_best = True
    batch_size = 20
    lr = 0.001
    epochs = 150
    early_stop = 100
    weight_decay = 0.001
    record = False  # 是否记录训练日志
