import os

import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
import numpy as np


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, cuda, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, dice_loss, cls_weights, num_classes,
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss  = 0
    total_f_score   = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs  = imgs.cuda(local_rank)
                pngs  = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            pred_np = np.array(outputs.data.cpu()[0])[0]
            img_save_path = r'./train_img'
            np.save(os.path.join(img_save_path, f"{epoch}.npy"), pred_np)
            loss = CE_Loss(outputs, pngs, weights, num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            # outputs1 = outputs[:, :, 0:100, 0:100].clone()
            # pngs1 = pngs[:, 0:100, 0:100].clone()
            # outputs2 = outputs[:, :, 156:, :100].clone()
            # pngs2 = pngs[:, 156:, :100].clone()
            # outputs3 = outputs[:, :, :100, 156:].clone()
            # pngs3 = pngs[:, :100, 156:].clone()
            # outputs4 = outputs[:, :, 156:, 156:].clone()
            # pngs4 = pngs[:, 156:, 156:].clone()
            # loss1 = CE_Loss(outputs1, pngs1, weights, num_classes)
            # loss2 = CE_Loss(outputs2, pngs2, weights, num_classes)
            # loss3 = CE_Loss(outputs3, pngs3, weights, num_classes)
            # loss4 = CE_Loss(outputs4, pngs4, weights, num_classes)
            # loss = loss + (loss1 + loss2 + loss3 + loss4) * 2

            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)


    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
        model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            outputs = model_train(imgs)
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            # outputs1 = outputs[:, :, 0:100, 0:100].clone()
            # pngs1 = pngs[:, 0:100, 0:100].clone()
            # outputs2 = outputs[:, :, 156:, :100].clone()
            # pngs2 = pngs[:, 156:, :100].clone()
            # outputs3 = outputs[:, :, :100, 156:].clone()
            # pngs3 = pngs[:, :100, 156:].clone()
            # outputs4 = outputs[:, :, 156:, 156:].clone()
            # pngs4 = pngs[:, 156:, 156:].clone()
            # loss1 = CE_Loss(outputs1, pngs1, weights, num_classes)
            # loss2 = CE_Loss(outputs2, pngs2, weights, num_classes)
            # loss3 = CE_Loss(outputs3, pngs3, weights, num_classes)
            # loss4 = CE_Loss(outputs4, pngs4, weights, num_classes)
            # loss = loss + (loss1 + loss2 + loss3 + loss4) * 2

            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))