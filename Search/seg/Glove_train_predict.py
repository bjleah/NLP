from seg.corpus_process import corpus_p
import os
from seg import Glove
import pickle
import numpy as np
embedding_size = 300

def train():
    corpus_path = os.getcwd() + "\\data\\rmrb_file"
    corpus = corpus_p.read_rmrb_file(corpus_path)
    model = Glove.GloVeModel(embedding_size=300, context_size=10)
    model.fit_to_corpus(corpus)
    model.train(num_epochs=100)
    outpath1 = os.getcwd() + "\\Glove_embedding.m"
    outpath2 = os.getcwd() + "\\Glove_word2id.m"
    pickle.dump(model.embeddings, open(outpath1, "wb"))
    pickle.dump(model.id_for_word, open(outpath2, "wb"))
    return model

def embedding_for(word_str_or_id):
    outpath1 = os.getcwd() + "\\Glove_embedding.m"
    outpath2 = os.getcwd() + "\\Glove_word2id.m"

    embeddings = pickle.load(open(outpath1, "rb"))
    word_to_id = pickle.load(open(outpath2, "rb"))
    if isinstance(word_str_or_id, str):
        id = word_to_id.get(word_str_or_id,-1)
        if id == -1:
            return 2 * np.random.rand(embedding_size) - 1
        return embeddings[id]
    elif isinstance(word_str_or_id, int):
        return embeddings[word_str_or_id]


if __name__ == "__main__":
    model = train()
    word = "中国人"
    for char in word:
        print(model.embedding_for(char))

    # word_str_or_id = "楫"
    # vector = embedding_for(word_str_or_id)
    # print(vector)



