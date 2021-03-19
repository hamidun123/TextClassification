import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from models import FGM, PGD


def train(train_iter, dev_iter, model, args, writer):
    if args.cuda:
        device = torch.device("cuda:0")
        model.cuda(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    best_acc = 0
    last_step = 0

    print('training...')
    fgm = FGM.FGM(model)
    pgd = PGD.PGD(model)
    K = 3
    for epoch in range(1, args.epochs + 1):
        correct = 0
        total = 0
        model.train()
        for batch in train_iter:

            feature, target, length = batch[0], batch[1], batch[2]  # (W,N) (N)

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            if args.RNN_flag:
                logit = model(feature, length)
            else:
                logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()

            # FGM对抗训练
            # fgm.attack()
            # logit_adv = model(feature, length)
            # loss_adv = F.cross_entropy(logit_adv, target)
            # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复embedding参数

            # PGD对抗训练
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                logit_adv = model(feature, length)
                loss_adv = F.cross_entropy(logit_adv, target)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数


            optimizer.step()

            result = torch.max(logit, 1)[1].view(target.size())
            corrects = (result.data == target.data).sum()
            correct = correct + corrects
            total = total + len(feature)

        accuracy = correct * 100.0 / total

        ExpLR.step()

        if writer is not None:
            writer.add_scalar('Loss', loss.item(), epoch)
        print('\repoch[{}] - loss: {:.6f} acc: {:.4f}%({}/{})'.format(epoch,
                                                                      loss.data.item(),
                                                                      accuracy,
                                                                      correct,
                                                                      total), end='')
        dev_acc = test(dev_iter, model, args)
        if writer is not None:
            writer.add_scalar('ACC', dev_acc, epoch)
        if dev_acc >= best_acc:
            best_acc = dev_acc
            last_step = epoch
            if args.save_best:
                model.save("checkpoints/{}_best.pth".format(args.model_name))
        else:
            if epoch - last_step >= args.early_stop:
                print('\nearly stop by {} epoch.'.format(args.early_stop))
                break
    print("\nbest acc :{}%".format(best_acc))


def test(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target, length = batch[0], batch[1], batch[2]

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        if args.RNN_flag:
            logit = model(feature, length)
        else:
            logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f} acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, size), end='')

    return accuracy
