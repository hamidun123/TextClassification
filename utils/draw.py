import matplotlib.pyplot as plt

def plot_attention(alphas):
    for i in range(len(alphas)):
        alphas[i] = alphas[i].cpu().data
    x = [i for i in range(80)]
    sentence_num = alphas[0].shape[0]
    for i in range(sentence_num):
        l1, = plt.plot(x, alphas[0][i, :, 0])
        l2, = plt.plot(x, alphas[1][i, :, 0])
        l3, = plt.plot(x, alphas[2][i, :, 0])
        plt.legend(handles=[l1, l2, l3], labels=['domain', 'command', 'value'], loc='best')
        plt.show()
