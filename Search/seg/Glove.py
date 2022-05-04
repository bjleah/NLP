from __future__ import division
from collections import Counter, defaultdict
import os
from random import shuffle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


"""
embedding_size: the target dimensionality of the trained word representations. Typically between 50 and 300.
context_size: how many tokens on either side of a given word to include in each context window. Can be either a tuple of two ints, indicating how many token on the left and right to include, or a single int, which will be interpreted to mean symmetric context.
max_vocab_size (Optional): the maximum size of the model's vocabulary. The model's vocabulary will be the most frequently occurring words in the corpus up to this amount. The default is 100,000.
min_occurrences (Optional): the minimum number of times a word must have appeared in the corpus to be included in the model's vocabulary. Default is 1.
scaling_factor (Optional): the alpha term in Eqn. 9 of Pennington et al.'s paper. Default is 3/4, which is the paper's recommendation
cooccurrence_cap (Optional): the x_max term in Eqn. 9 of Pennington et al.'s paper. Default is 100, which is the paper's recommendation
batch_size (Optional): the number of cooccurrences per minibatch of in training. Default is 512, which seems to work well on my machine. If training is very slow, consider playing with this.
learning_rate (Optional): the Adagrad learning rate used in training. Default is 0.05, which is the paper's recommendation

"""

"""
利用人民日报的语料进行训练
"""


class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class GloVeModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 scaling_factor=3/4, cooccurrence_cap=100, batch_size=512, learning_rate=0.05):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__embeddings = None###list[[word1_embedding],[word2_embedding]]

    def fit_to_corpus(self, corpus):
        self.__fit_to_corpus(corpus, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__build_graph()

    def __fit_to_corpus(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrences]
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
        self.__cooccurrence_matrix = {
            (self.__word_to_id[words[0]], self.__word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id}###将文本转为id并进行计算

    def __build_graph(self):
        self.__graph = tf.Graph()##指定图
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")#feed 进去的是batch文本
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")#feed 进去的是batch文本
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),#初始化word embedding 均匀分布
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="context_embeddings")

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)#利用lookup从param中找到输入word的embedding，返回的向量shape为【batch-size，embedding——size】
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            #权重函数，上升到1停止f(Xij)
            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))



            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)#将embedding维度数据相加，shape = （batch_size,）

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.__total_loss)#将变量保存在直方图中
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        batches = self.__prepare_batches()
        total_steps = 0
        with tf.Session(graph=self.__graph) as session:###在指定图里sess运行
            if should_write_summaries:
                print("Writing TensorBoard summaries to {}".format(log_dir))
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                    session.run([self.__optimizer], feed_dict=feed_dict)
                    if should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                        summary_str = session.run(self.__summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_steps)
                    total_steps += 1
                if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                    current_embeddings = self.__combined_embeddings.eval()
                    output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                    self.generate_tsne(output_path, embeddings=current_embeddings)
            self.__embeddings = self.__combined_embeddings.eval()#####tensorflow中eval意为run一个tensor，run可同时多个
            if should_write_summaries:
                summary_writer.close()

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))#[([],[],[]),()]
    # property调用方法同属性一样，不需要加括号
    @property
    def vocab_size(self):
        return len(self.__words)


    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    @property
    def id_for_word(self):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)


if __name__ == "__main__":
    model = GloVeModel(embedding_size=30, context_size=5,batch_size=30)
    corpus = ["八廓街延展着历史的脉络，三坊七巷的青石板晕染着岁月的光泽，针针细密的苗绣上镌刻着悠远的时光……历史伟力与时代张力交握，盛放着中华文化的磅礴力量。中华优秀传统文化是中华民族的“根”和“魂”，善于继承才能更好创新，不忘历史才能开辟未来。","弦歌不辍，薪火相传。每一个文化故事都是古与今的对话，闪烁着穿越时空的璀璨光芒，是山川河流间回荡的长歌壮曲。新征程上，我们要更加坚定文化自信，凝聚发展共识，开创文化繁荣发展新局面，谱写文化强国新篇章。"]
    model.fit_to_corpus(corpus)
    model.train(num_epochs=10)
    res = model.embedding_for("的")
    print(res)