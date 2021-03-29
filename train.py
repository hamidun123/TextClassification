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
    # fgm = FGM.FGM(model)
    pgd = PGD.PGD(model)
    K = 3
    for epoch in range(1, args.epochs + 1):
        correct_batch = [0, 0, 0]  # [domain, command, value]
        all_corrects = 0
        total = 0
        model.train()
        for batch in train_iter:

            feature, length, label = batch[0], batch[1], batch[2]  # (W,N) (N)

            if args.cuda:
                feature, label[0], label[1], label[2] = feature.cuda(), label[0].cuda(), label[1].cuda(), label[2].cuda()

            optimizer.zero_grad()
            if args.RNN_flag:
                logit = model(feature, length)
            else:
                logit = model(feature)
            loss_list = []
            for i in range(len(logit)):
                loss_list.append(F.cross_entropy(logit[i], label[i]))

            loss = loss_list[0] + loss_list[1] + loss_list[2]
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

                loss_list_adv = []
                for i in range(len(logit)):
                    loss_list_adv.append(F.cross_entropy(logit_adv[i], label[i]))

                loss_adv = loss_list_adv[0] + loss_list_adv[1] + loss_list_adv[2]
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数

            optimizer.step()

            # 分类统计
            result = []
            correct_matrix = torch.zeros(3, feature.shape[0])
            for i in range(len(logit)):
                result.append(torch.max(logit[i], 1)[1])
                for j in range(result[i].shape[0]):
                    if result[i][j].data == label[i][j].data:
                        correct_matrix[i][j] = 1

            all_correct = (correct_matrix.sum(dim=0).data == 3).sum()
            all_corrects = all_correct + all_corrects

            for i in range(len(correct_batch)):
                correct_batch[i] = correct_batch[i] + correct_matrix[i].sum(dim=0).data

            total = total + len(feature)
        accuracy = []
        for i in correct_batch:
            accuracy.append(i * 100.0 / total)
        accuracy.append(all_corrects * 100.0 / total)
        ExpLR.step()
        if writer is not None:
            writer.add_scalar('Loss', loss.data.item(), epoch)
        print('\repoch[{}] loss: {:.6f} d_acc: {:.4f}% c_acc: {:.4f}% v_acc: {:.4f}% acc: {:.4f}% ({}/{})'.format(epoch,
                                                                      loss.data.item(),
                                                                      accuracy[0], accuracy[1], accuracy[2], accuracy[3],
                                                                      all_corrects, total
                                                                      ), end='')
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
    correct_batch = [0, 0, 0]  # [domain, command, value]
    all_corrects = 0
    total = 0
    avg_loss = 0.0
    for batch in data_iter:
        feature, length, label = batch[0], batch[1], batch[2]  # (W,N) (N)

        if args.cuda:
            feature, label[0], label[1], label[2] = feature.cuda(), label[0].cuda(), label[1].cuda(), label[2].cuda()

        if args.RNN_flag:
            logit = model(feature, length)
        else:
            logit = model(feature)

        loss_list = []
        for i in range(len(logit)):
            loss_list.append(F.cross_entropy(logit[i], label[i]))
        loss = loss_list[0] + loss_list[1] + loss_list[2]
        avg_loss += loss.item()

        result = []
        correct_matrix = torch.zeros(3, feature.shape[0])
        for i in range(len(logit)):
            result.append(torch.max(logit[i], 1)[1])
            for j in range(result[i].shape[0]):
                if result[i][j].data == label[i][j].data:
                    correct_matrix[i][j] = 1

        all_correct = (correct_matrix.sum(dim=0).data == 3).sum()
        all_corrects = all_correct + all_corrects

        for i in range(len(correct_batch)):
            correct_batch[i] = correct_batch[i] + correct_matrix[i].sum(dim=0).data

        total = total + len(feature)

    accuracy = []
    for i in correct_batch:
        accuracy.append(i * 100.0 / total)
    accuracy.append(all_corrects * 100.0 / total)
    size = len(data_iter.dataset)
    avg_loss /= size

    print(
        ' test loss: {:.6f} d_acc: {:.4f}% c_acc: {:.4f}% v_acc: {:.4f}% acc: {:.4f}% ({}/{})'.format(
            avg_loss,
            accuracy[0], accuracy[1], accuracy[2], accuracy[3], all_corrects, total
            ), end='')

    return accuracy[3]
