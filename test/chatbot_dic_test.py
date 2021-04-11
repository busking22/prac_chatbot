import sys

sys.path.append(".")
import pickle
from utils.preprocess import Preprocess

f = open("./train/chatbot_bin.bin", "rb")
word_index = pickle.load(f)
f.close()

sent = "갑자기 짜장면 먹고 싶네 ㅋㅋ"

p = Preprocess("./train/chatbot_bin.bin")
pos = p.pos(sent)
keywords = p.get_keywords(pos, without_tag=True)

print(p.word_index)
print(p.get_wordidx_sequence(keywords))
for word in keywords:
    try:
        print(word, word_index[word])
    except KeyError:
        print(word, word_index["OOV"])
