import sys

sys.path.append(".")
from utils.preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2idx_dic="./train/chatbot_bin.bin")
intent = IntentModel(model_name="./models/intent/intent_model.h5", preprocess=p)

query = input()
predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print("intent label :", predict_label)
