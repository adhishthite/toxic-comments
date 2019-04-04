import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def get_part_of_speech(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def normalize(text):
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    stop_words = set(stopwords.words('english'))
    translate_map = str.maketrans(filters, " " * len(filters))
    
    text = text.lower()
    text = text.translate(translate_map)
    
    tokens = nltk.word_tokenize(text)
    
    tags = nltk.pos_tag(tokens)
    
    normalized_text = [WordNetLemmatizer().lemmatize(tag[0], pos=get_part_of_speech(tag[1])) for tag in tags if tag[0] not in stop_words if len(tag[0]) > 2]

    return normalized_text


def text_to_sequence(text, word_index):
    sequence = []
    
    for word in text:
        if not word_index.get(word): continue 
        sequence.append(word_index[word])
    
    return sequence


def create_word_index(text_list):
    word_counts = dict()
    for norm_text in text_list:
        for word in norm_text:
            if word not in word_counts:
                word_counts[word] = 0
            
            word_counts[word] += 1
    
    w_counts = list(word_counts.items())
    w_counts.sort(key=lambda x: x[1], reverse=True)
    
    sorted_voc = [wc[0] for wc in w_counts]
    
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
    
    return word_index