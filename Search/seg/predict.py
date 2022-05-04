from seg.crf_bilstm import CRF_BILSTM

class segment:
    @classmethod
    def seg(cls,sentence):###加参数需指明没被实例化的类本身
        word_list, site = CRF_BILSTM().predict(sentence)
        res = list(zip(word_list, site))
        print(res)
        return res



if __name__ == "__main__":
    sen = "晚风依旧很温柔,一个人慢慢走，在街道的岔路口，眺望银河与星斗"
    res = segment.seg(sen)

