from __future__ import division##导入python 高版本出发 return float
from collections import Counter, defaultdict
import os
from random import shuffle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from log import get_logger
from seg.corpus_process import corpus_p
from seg.Glove_train_predict import embedding_for
logger = get_logger()

"""
describe:
todo:define crf_bilstm network,training process and preiction process
todo2:define cut function
todo3:define evaluate function

author:Leah(liuye)
"""

embedding_size = 300
tag2id = {"b":0, "m":1, "e":2, "s":3}#####一定要从0开始
id2tag = {0: "b", 1: "m", 2: "e", 3: "s"}

train_path = os.getcwd() + "\\data\\rmrb_train.txt"
test_path = os.getcwd() + "\\data\\rmrb_test.txt"

class CRF_BILSTM(object):
    def __init__(self, nums_epoch=50, batch_size=10, time_steps=30, dim=300, nums_tags=4, drop_rate=0.01):
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.nums_tags = nums_tags
        self.drop_rate = drop_rate
        self.nums_epoch = nums_epoch

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(shape=[self.batch_size, self.time_steps, embedding_size], dtype=tf.float32)
            self.y = tf.placeholder(shape=[self.batch_size, self.time_steps], dtype=tf.int32)
            self.sequence_lengths_t = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32)
            #
            #
            # #define bilstm layers
            Lstm_fw_layers1 = layers.LSTM(self.dim, return_sequences=True, go_backwards=False, dropout=self.drop_rate)
            Lstm_bw_layers1 = layers.LSTM(self.dim, return_sequences=True, go_backwards=True, dropout=self.drop_rate)

            Lstm_fw_layers2 = layers.LSTM(self.dim, return_sequences=True, go_backwards=False, dropout=self.drop_rate)
            Lstm_bw_layers2 = layers.LSTM(self.dim, return_sequences=True, go_backwards=True, dropout=self.drop_rate)

            # merge_mode 前向和后向 RNN 的输出的结合模式
            bilstm1 = layers.Bidirectional(merge_mode="sum", layer=Lstm_fw_layers1, backward_layer=Lstm_bw_layers1)
            bilstm2 = layers.Bidirectional(merge_mode="sum", layer=Lstm_fw_layers2, backward_layer=Lstm_bw_layers2)

            h1 = bilstm1(self.x)
            h2 = bilstm2(h1)


            #define crf layer
            #发射矩阵


            weights = tf.get_variable("weights", [self.dim, self.nums_tags])
            matricized_x_t = tf.reshape(h2, [-1, self.dim])
            matricized_unary_scores = tf.matmul(matricized_x_t, weights)
            unary_scores = tf.reshape(matricized_unary_scores,[self.batch_size, self.time_steps, self.nums_tags])

            ##计算预测序列和真实序列相似度，和转移矩阵参数,shape =（self.nums_tags,self.nums_tags)）
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, self.y, self.sequence_lengths_t)
            #计算维特比得分和序列
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params, self.sequence_lengths_t)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
            self.saver = tf.compat.v1.train.Saver(max_to_keep=3)###需在图定义下初始化saver变量


    def batchy(self, path = None, sen = None):
        if path:
            logger.info("reading data from file......")
            corpus, tags = corpus_p().read_rmrb_data(path)
            interval = self.batch_size * self.time_steps
            for i in range(0, len(corpus), interval):
                if i+interval > len(corpus):
                    tmp_corpus = np.vstack((np.array([embedding_for(c) for c in corpus[i:len(corpus)]]), 2 * np.random.rand(i+interval-len(corpus), embedding_size) - 1))
                    tmp_tags = np.array([tag2id[t] for t in tags[i:len(corpus)]] + [3]*(i+interval-len(corpus)))
                    seq_len = np.array([1]*(len(corpus)-i) + [0]*(i+interval-len(corpus)))
                else:
                    tmp_corpus = np.array([embedding_for(c) for c in corpus[i:i+interval]])
                    tmp_tags =np.array([tag2id[t] for t in tags[i:i+interval]])
                    seq_len = np.array([1]*interval)
                tmp_corpus = tmp_corpus.reshape((self.batch_size, self.time_steps, embedding_size))
                tmp_tags = tmp_tags.reshape((self.batch_size, self.time_steps))
                seq_len = np.sum(seq_len.reshape((self.batch_size, self.time_steps)), axis=1)
                yield (tmp_corpus,tmp_tags,seq_len)
        else:
            logger.info("predict sentence tags......")
            interval = self.batch_size * self.time_steps
            corpus = sen
            for i in range(0, len(corpus), interval):
                if i + interval > len(corpus):
                    tmp_corpus = np.vstack((np.array([embedding_for(c) for c in corpus[i:len(corpus)]]),
                                            2 * np.random.rand(i + interval - len(corpus), embedding_size) - 1))
                    seq_len = np.array([1] * (len(corpus) - i) + [0] * (i + interval - len(corpus)))
                else:
                    tmp_corpus = np.array([embedding_for(c) for c in corpus[i:i + interval]])
                    seq_len = np.array([1] * interval)
                tmp_corpus = tmp_corpus.reshape((self.batch_size, self.time_steps, embedding_size))
                seq_len = np.sum(seq_len.reshape((self.batch_size, self.time_steps)), axis=1)
                yield (tmp_corpus, seq_len)


    def train(self):
        self.build_graph()
        logger.info("graph build success......")
        path = os.getcwd() + "\\model\\bilstm_crf"
        data_path = os.getcwd() + "\\data\\rmrb_train.txt"
        batches = self.batchy(data_path)

        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            # Train for a fixed number of iterations.
            for i in range(self.nums_epoch):
                for batch_index, batch in enumerate(batches):
                    x, y, seq_len = batch
                    total_labels = np.sum(seq_len)
                    mask = (np.expand_dims(np.arange(self.time_steps), axis=0) < np.expand_dims(seq_len, axis=1))

                    tf_viterbi_sequence, _ = session.run([self.viterbi_sequence, self.train_op], feed_dict={self.x: x, self.y: y, self.sequence_lengths_t: seq_len})
                    correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                    accuracy = 100.0 * correct_labels / float(total_labels)
                    if batch_index%1000 == 0:
                        print("Accuracy: %.2f%%" % accuracy)

                self.saver.save(session, path, global_step=i+1)###表明第几步的时候存储,model命名 model_name-step



    def model_test(self):
        self.build_graph()
        path = os.getcwd() + "\\model\\bilstm_crf-49"

        data_path = os.getcwd() + "\\data\\rmrb_test.txt"
        batches = self.batchy(data_path)

        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            self.saver.restore(session, path)
            real = []
            predict = []
            for batch_index, batch in enumerate(batches):
                x, y, seq_len = batch
                total_labels = np.sum(seq_len)
                mask = (np.expand_dims(np.arange(self.time_steps), axis=0) < np.expand_dims(seq_len, axis=1))
                viterbi_s = session.run(self.viterbi_sequence,feed_dict={self.x: x, self.sequence_lengths_t: seq_len})

                correct_labels = np.sum((y == viterbi_s) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)

                real.extend(y.reshape(self.batch_size*self.time_steps)[:int(total_labels)])
                predict.extend(viterbi_s.reshape(self.batch_size*self.time_steps)[:int(total_labels)])

        P, R, F1 = self.indicator(self.cut(tags = predict),self.cut(tags = real))
        logger.info("the test data Precision is %f, Recall is %f, F1 value is %f"%(P, R, F1))

        return real,predict


    def inter(self, A, B):  # 求交集,A,B,list
        res = list(set(A).intersection(set(B)))
        res.sort()
        return res


    def union(self, A, B):  # 求并集
        res = list(set(A).union(set(B)))
        res.sort()
        return res


    def diff(self, A, B):  # 求差集，A有，B没有
        res = list(set(A).difference(set(B)))
        res.sort()
        return res


    def indicator(self, pred_ind_list, true_ind_list):
        # P精确率---正确切分的词/切分出的词
        # R召回率---正确切分的词/标注出的所有切词
        # F1值---2*p*r/p+r
        if len(pred_ind_list) == 0 or len(true_ind_list) == 0:
            return 0, 0, 0
        P = len(self.inter(pred_ind_list, true_ind_list)) / len(pred_ind_list)
        R = len(self.inter(pred_ind_list, true_ind_list)) / len(true_ind_list)
        F1 = 2 * P * R / (P + R)
        return P, R, F1


    def evaluate(self):
        pass


    def cut(self,sen = None, tags = None):
        res = []
        words = []
        if sen is None:
            sen = "￥"*len(tags)

        tmp = []
        for no, tag in enumerate(tags):
            if no == len(tags)-1:
                if len(tmp) == 1:
                    res.append((tmp[0], no))
                    words.append(sen[tmp[0]:no+1])
                elif len(tmp) == 0:
                    res.append((no, no))
                    words.append(sen[no])
                break
            if tag == tag2id["b"]:
                if tags[no+1] == tag2id["s"] or tags[no+1] == tag2id["b"]:
                    res.append((no, no))
                    words.append(sen[no])
                else:
                    tmp.append(no)

            elif tag == tag2id["m"]:
                if tags[no+1] == tag2id["b"] or tags[no+1] == tag2id["s"]:
                    tmp.append(no)
                    if len(tmp) == 1:
                        tmp.append(no)
                        res.append(tuple(tmp))
                        words.append(sen[tmp[0]:tmp[1]+1])
                        tmp = []
                    elif len(tmp) == 2:
                        res.append(tuple(tmp))
                        words.append(sen[tmp[0]:tmp[1] + 1])
                        tmp = []
                else:
                    if len(tmp) == 0:
                        tmp.append(no)
                    else:
                        continue

            elif tag == tag2id["e"]:
                tmp.append(no)
                if len(tmp) == 1:
                    res.append((no,no))
                    words.append(sen[no])
                    tmp = []
                elif len(tmp) == 2:
                    res.append(tuple(tmp))
                    words.append(sen[tmp[0]:tmp[1]+1])
                    tmp = []
            elif tag == tag2id["s"]:
                res.append((no,no))
                words.append(sen[no])
        if sen[-1] == "￥":
            return res
        return words, res




    # @staticmethod
    def predict(self, sentence):
        self.build_graph()
        path = os.getcwd() + "\\model\\bilstm_crf-49"
        batches = self.batchy(sen = sentence)
        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            self.saver.restore(session, path)
            predict = []
            for batch_index, batch in enumerate(batches):
                x,seq_len = batch
                total_labels = np.sum(seq_len)
                viterbi_s = session.run(self.viterbi_sequence,feed_dict={self.x: x, self.sequence_lengths_t: seq_len})
                predict.extend(viterbi_s.reshape(self.batch_size*self.time_steps)[:int(total_labels)])

        return self.cut(sentence, predict)





if __name__ == "__main__":
    obj = CRF_BILSTM()
    a,b = obj.model_test()
    sentence = "五星红旗 迎风飘扬，胜利歌声多么响亮，歌唱我们亲爱的祖国，一起走向繁荣富强"
    words, res= obj.predict(sentence)
    print(words,res)
    # obj.train()


