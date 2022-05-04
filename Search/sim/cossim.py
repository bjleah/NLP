from collections import Counter

class cossim:
    def __init__(self, docs, query):
        self.q_c = Counter(query)
        self.docs = docs
        self.query = query

    def cos_sim(self, dct1, dct2):
        dot = 0.0
        sum1 = 0.0
        sum2 = 0.0
        for key in dct1:
            tf1 = dct1[key]
            tf2 = 0.0
            if key in dct2:
                tf2 = dct2[key]
                dot += tf1 * tf2
            sum1 += tf1 * tf1
            sum2 += tf2 * tf2
        for key in dct2:
            if key not in dct1:
                tf2 = dct2[key]
                sum2 += tf2 * tf2

        if sum1 == 0.0 or sum2 == 0.0:
            return 0.0
        sim = dot / ((sum1 ** 0.5) * (sum2 ** 0.5))
        return sim

    def cos_score(self):
        id2score = {}
        for doc in self.docs:
            id2score[doc] = self.cos_sim(doc.count, self.q_c)

        return dict(sorted(id2score.items(), key=lambda x : x[1], reverse=True))

    def top_doc_id(self, k = 10):
        id = []
        for k,v in self.cos_score().items():
            id.append(k.id)
        return id[:min(k,len(id))]









