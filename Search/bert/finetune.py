import os
import re
import time
import logging
import json
from random import random
from functools import reduce, partial

import numpy as np
import logging
import argparse
from pathlib import Path
import paddle as P
from bert import get_param
from propeller import log
import propeller.paddle as propeller#辅助模型训练的高级框架，包含NLP常用的前、后处理流

from log import get_logger
logger = get_logger()

from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from bert.utils import create_if_not_exists, get_warmup_and_linear_decay

"""
describe:利用data中的数据对预训练的分类模型进行finetune, 确保你的环境安装paddle框架
author :Leah
"""
args = get_param.get_param()
# print(str(args.save_dir)+"\\ckpt.bin")
tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)

feature_column = propeller.data.FeatureColumns([
    propeller.data.TextColumn('seg_a', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize),
    propeller.data.TextColumn('seg_b', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize),
    propeller.data.LabelColumn('label', vocab_dict={b"0": np.int64(0), b"1": np.int64(1)})])#####注意这里的key,value形式,label的类型为int64,注意转换


def map_fn(seg_a, seg_b, label):
    seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
    sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
    return sentence, segments, label

###注意数据的目录形式
train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                               .map(map_fn) \
                               .padded_batch(args.bsz, (0, 0, 0))

dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                               .map(map_fn) \
                               .padded_batch(args.bsz, (0, 0, 0))

model = ErnieModelForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=2, name='')

g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
param_name_to_exclue_from_weight_decay = re.compile(r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')
if args.use_lr_decay:####是否使用学习率衰减
    lr_scheduler = P.optimizer.lr.LambdaDecay(args.lr,get_warmup_and_linear_decay(args.max_steps, int(args.warmup_proportion * args.max_steps)))
    opt = P.optimizer.AdamW(lr_scheduler, parameters=model.parameters(), weight_decay=args.wd, apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n), grad_clip=g_clip)
else:
    lr_scheduler = None
    opt = P.optimizer.AdamW(args.lr, parameters=model.parameters(), weight_decay=args.wd, apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n), grad_clip=g_clip)

scaler = P.amp.GradScaler(enable=args.use_amp)####

step, inter_step = 0, 0
acc_step = 1

with P.amp.auto_cast(enable=args.use_amp):###使用自动混合精度来完成下面的训练（加速）
    for epoch in range(args.epoch):
        for ids, sids, label in P.io.DataLoader(train_ds, batch_size=None):
            inter_step += 1
            loss, _ = model(ids, sids, labels=label)####ErnieModelForPretraining forwards
        #     src_ids (`Variable` of shape `[batch_size, seq_len]`): Indices of input sequence tokens in the vocabulary.
        #     sent_ids (optional, `Variable` of shape `[batch_size, seq_len]`):
            loss /= acc_step
            loss = scaler.scale(loss)###损失放大
            loss.backward()###反向传播
            if inter_step % acc_step != 0:
                continue
            step += 1
            scaler.minimize(opt, loss)
            model.clear_gradients()###不知道，
            lr_scheduler and lr_scheduler.step()

            if step % 10 == 0:
                _lr = lr_scheduler.get_lr() if args.use_lr_decay else args.lr
                if args.use_amp:
                    _l = (loss / scaler._scale).numpy()
                    msg = '[step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                        step, _l, _lr, scaler._scale.numpy())
                else:
                    _l = loss.numpy()
                    msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l,
                                                                 _lr)
                logger.debug(msg)
                # log_writer.add_scalar('loss', _l, step=step)
                # log_writer.add_scalar('lr', _lr, step=step)
            if step % 100 == 0:
                acc = []
                with P.no_grad():
                    model.eval()
                    for ids, sids, label in P.io.DataLoader(dev_ds, batch_size=None):
                        loss, logits = model(ids, sids, labels=label)
                        #print('\n'.join(map(str, logits.numpy().tolist())))
                        a = (logits.argmax(-1) == label)
                        acc.append(a.numpy())
                    model.train()# 将模型设置为训练状态
                acc = np.concatenate(acc).mean()
                # log_writer.add_scalar('eval/acc', acc, step=step)
                logger.debug('acc %.5f' % acc)
                if args.save_dir is not None:
                    P.save(model.state_dict(), "./ckpt.bin")####路径设置为相对路径，不然会报错(0xC0000005)
if args.save_dir is not None:
    P.save(model.state_dict(), "./ckpt.bin")



# if __name__ == "__main__":
    # for i in dev_ds:
    #     for j in i:
    #         print(j.shape)



