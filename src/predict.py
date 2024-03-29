from keras.models import load_model
from preprocess import normalize, create_word_index, text_to_sequence
from keras.preprocessing.sequence import pad_sequences
import pickle


def predict(text_list):
    
    MAX_LENGTH = 150
    model = load_model('../models/new_model.h5')

    with open("../data/word_index.pkl","rb") as fp:
        word_index = pickle.load(fp)

    seqs = [normalize(text) for text in text_list]
    test_words = [text_to_sequence(seq, word_index) for seq in seqs]
    test_words = pad_sequences(test_words, maxlen=MAX_LENGTH)
    y_pred = model.predict(test_words, batch_size=1024)

    return y_pred

if __name__ == "__main__":
    
    text_list = []

    text_list.append('This is a demo threat')
    text_list.append('hope everything is going great')
    
    predictions = predict(text_list)

    print(predictions)