import paddle as P
import propeller.paddle as propeller#辅助模型训练的高级框架，包含NLP常用的前、后处理流
import numpy as np
from log import get_logger
logger = get_logger()
from bert import get_param
import os
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
##load_model
model_path = "./ckpt.bin"
args = get_param.get_param()
def model_load(model_path = model_path):
    params_state = P.load(model_path)
    model = ErnieModelForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=2, name='')
    model.set_dict(params_state)
    return model


###predict
def single_predict(seg_a, seg_b):
    model = model_load()
    model.eval()
    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
    sen, seg = tokenizer.encode(seg_a, pair=seg_b)
    ids = P.to_tensor(np.expand_dims(sen, 0))
    sids = P.to_tensor(np.expand_dims(seg, 0))
    _, logit = model(ids, sids)
    label = np.argmax(logit.numpy())
    return logit.numpy()[0][label], label



def batch_predict(test_dir):#目录形式
    model = model_load()
    model.eval()
    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn('seg_a', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize),
        propeller.data.TextColumn('seg_b', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize)
        ])#####注意这里的key,value形式,label的类型为int64,注意转换


    def map_fn(seg_a, seg_b):
        seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
        return sentence, segments

    test_ds = feature_column.build_dataset('test', data_dir=test_dir, shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0))

    for ids, sids in P.io.DataLoader(test_ds, batch_size=None):
        loss, logits = model(ids, sids)
        # print(logits)
        a = np.argmax(logits.numpy(), axis=1)
        yield a

if __name__ == "__main__":
    # seg_a = "我是中国人"
    # seg_b = "中国人我是"
    # log,label = single_predict(seg_a, seg_b)
    # print(log)
    # print(label)

    res = batch_predict('./data/test')
    for batch_no, i in enumerate(res):
        print(batch_no, "\n", i)