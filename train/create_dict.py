import sys

sys.path.append(".")

from utils.preprocess import Preprocess
from keras import preprocessing
import pickle


def read_corpus_data(filename):
    with open(filename, "r") as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:]
    return data


corpus_data = read_corpus_data("./train/corpus.txt")

p = Preprocess()
dic = []
for c in corpus_data:
    pos = p.pos(c[1])
    for k in pos:
        dic.append(k[0])

tokenizer = preprocessing.text.Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(dic)
word_index = tokenizer.word_index

f = open("./train/chatbot_bin.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()
