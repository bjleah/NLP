import numpy as np
import paddle as P
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel,ErnieModelForSequenceClassification
import os

model_path = os.getcwd()+"\\model-ernie-gram-zh.1"
model = ErnieModelForSequenceClassification.from_pretrained(model_path, num_labels=3)    # Try to get pretrained model from server, make sure you have network connection
model.eval()
tokenizer = ErnieTokenizer.from_pretrained(model_path)
# ids, _ = tokenizer.encode('中国')
# ids = P.to_tensor(np.expand_dims(ids, 0))  # insert extra `batch` dimension
# pooled, encoded = model(ids)                 # eager execution
# print(ids)
# print(encoded)
# print(pooled)

if __name__ == "__main__":
    a = "我是中国人"
    b = "中国人我是"
    sen, seg = tokenizer.encode(a,b)
    print(sen)
    print(seg)
    ids = P.to_tensor(np.expand_dims(sen,0))
    ssid = P.to_tensor(np.expand_dims(seg,0))
    loss,logit = model(ids,ssid)
    print(loss)
    print(logit.numpy())







