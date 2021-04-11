import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(".")

import pandas as pd
import tensorflow as tf
from keras import preprocessing
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

train_file_path = "./models/intent/total_train_data.csv"
data = pd.read_csv(train_file_path)
queries = data["query"].tolist()
intents = data["intent"].tolist()

from utils.preprocess import Preprocess

p = Preprocess(word2idx_dic="./train/chatbot_bin.bin")

sequences = []
for sent in queries:
    pos = p.pos(sent)
    keywords = p.get_keywords(pos, without_tag=True)
    seq = p.get_wordidx_sequence(keywords)
    sequences.append(seq)

from config.GlobalParams import MAX_SEQ_LEN
from sklearn.model_selection import train_test_split

padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post")

data_size = len(padded_seqs)
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents)).shuffle(data_size)

train_size = int(data_size * 0.7)
val_size = int(data_size * 0.2)
test_size = int(data_size * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

DROPOUT_PROB = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1

input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=DROPOUT_PROB)(embedding_layer)

conv1 = Conv1D(filters=128, kernel_size=3, padding="valid", activation=tf.nn.relu)(dropout_emb)
conv2 = Conv1D(filters=128, kernel_size=4, padding="valid", activation=tf.nn.relu)(dropout_emb)
conv3 = Conv1D(filters=128, kernel_size=5, padding="valid", activation=tf.nn.relu)(dropout_emb)

pool1 = GlobalMaxPool1D()(conv1)
pool2 = GlobalMaxPool1D()(conv2)
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=DROPOUT_PROB)(hidden)
logits = Dense(5)(dropout_hidden)
predictions = Dense(5, activation=tf.nn.softmax)(logits)

model = Model(inputs=input_layer, outputs=predictions)
print(model.summary())
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

loss, accuracy = model.evaluate(test_ds, verbose=1)
print("acc :", accuracy)
print("loss :", loss)

model.save("./intent/intent_model.h5")
