import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
from keras.models import load_model
from src.preprocess import normalize, create_word_index, text_to_sequence
from keras.preprocessing.sequence import pad_sequences
import pickle
import json

app = Flask(__name__)

MAX_LENGTH = 150
model = load_model('models/new_model.h5')
model._make_predict_function()
print('Model Loaded')

with open("data/word_index.pkl","rb") as fp:
    word_index = pickle.load(fp)

@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text_list = []
    text = ''
    
    if request.method == 'GET':
        text = request.args.get('text') 
    elif request.method == 'POST':
        data = json.loads(request.get_data().decode('utf-8'))
        text = data['text']

    print(text)    
    text_list.append(text)
    predictions = get_predictions(text_list)
    
    return jsonify(predictions)


def get_predictions(text_list):

    seqs = [normalize(text) for text in text_list]
    test_words = [text_to_sequence(seq, word_index) for seq in seqs]
    test_words = pad_sequences(test_words, maxlen=MAX_LENGTH)

    predictions = model.predict(test_words, batch_size=1024)

    results = []
    
    for prediction in predictions:
        result_dict = dict()
        result_dict['Toxic'] = str(round(prediction[0], 2))
        result_dict['Severely Toxic'] =str(round(prediction[1], 2))
        result_dict['Obscene'] =str(round(prediction[2], 2))
        result_dict['Threat'] =str(round(prediction[3], 2))
        result_dict['Insult'] =str(round(prediction[4], 2))
        result_dict['Identity Hate'] =str(round(prediction[5], 2))

        results.append(json.dumps(result_dict))

    return(results)


if __name__ == '__main__':
    app.run(port=5000, debug=True)