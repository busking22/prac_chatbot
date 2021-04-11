import sys
import os

sys.path.append(".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.models import Model, load_model
from keras import preprocessing


class IntentModel:
    def __init__(self, model_name, preprocess) -> None:
        self.labels = {0: "인사", 1: "욕설", 2: "주문", 3: "예약", 4: "기타"}

        self.model = load_model(model_name)

        self.p = preprocess

    def predict_class(self, query):
        pos = self.p.pos(query)

        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        from config.GlobalParams import MAX_SEQ_LEN

        padded_seq = preprocessing.sequence.pad_sequences(sequences, MAX_SEQ_LEN, padding="post")

        predict = self.model.predict(padded_seq)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]
