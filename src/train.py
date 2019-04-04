import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import pickle

from preprocess import normalize, create_word_index, text_to_sequence
from embedding_handler import create_embedding_matrix
from model import get_model

def train():

    MAX_LENGTH = 150

    print("READING FILES\n")
    train = pd.read_csv('data/train.csv')

    # train = train.head(200)

    print("NORMALIZING TEXT\n")
    seqs = [normalize(text) for text in train['comment_text']]

    print("CREATING WORD INDEX\n")
    word_index = create_word_index(seqs)

    print("STORING WORD INDEX\n")
    with open("data/word_index.pkl","wb") as fp:
        pickle.dump(word_index,fp)

    print("CREATING WORD SEQS\n")
    train_words = [text_to_sequence(seq, word_index) for seq in seqs]
    train_words = pad_sequences(train_words, maxlen=MAX_LENGTH)

    print("CREATING EMBEDDING MATRIX\n")
    embeding_matrix, MAX_WORDS = create_embedding_matrix(word_index)

    input_shape = (MAX_LENGTH, )

    print("CREATING MODEL\n")
    model = get_model(input_shape=input_shape, MAX_WORDS=MAX_WORDS, embeding_matrix=embeding_matrix)

    print(model.summary())

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from keras.callbacks import Callback

    batch_size = 128
    epochs = 4
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    X_train, X_val, y_train, y_val = train_test_split(train_words, y_train, train_size=0.95, random_state=42)

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)


    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=1)

    model.save('models/new_model.h5')

if __name__ == "__main__":
    train()