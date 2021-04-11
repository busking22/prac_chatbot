from konlpy.tag import Mecab
import pickle


class Preprocess:
    def __init__(self, word2idx_dic="", userdic=None) -> None:
        if word2idx_dic != "":
            f = open(word2idx_dic, "rb")
            self.word_index = pickle.load(f)
            f.close()
        else:
            self.word_index = None

        if userdic is None:
            self.mecab = Mecab()
        else:
            self.mecab = Mecab(dicpath=userdic)
        self.exclusion_tags = [
            "JKS",
            "JKC",
            "JKG",
            "JKO",
            "JKB",
            "JKV",
            "JKQ",
            "JX",
            "JC",
            "SF",
            "SP",
            "SS",
            "SE",
            "SO",
            "EP",
            "EF",
            "EC",
            "ETN",
            "ETM",
            "XSN",
            "XSV",
            "XSA",
        ]

    def pos(self, sent):
        return self.mecab.pos(sent)

    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if not f(p[1]):
                word_list.append(p if not without_tag else p[0])
        return word_list

    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                w2i.append(self.word_index["OOV"])
        return w2i


Preprocess()