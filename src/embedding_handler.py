import numpy as np

def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float64)


def create_embedding_dict():
    
    print("CREATING EMBEDDING DICT\n")
    EMBEDDING_FILE_PATH = '../data/glove.840B.300d.txt'
    embedding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(EMBEDDING_FILE_PATH))

    return embedding_dict


def create_embedding_matrix(word_index):
    MAX_FEATURES = 50000
    EMBED_SIZE = 300
    
    embeding_dict = create_embedding_dict()

    MAX_WORDS = max(MAX_FEATURES, len(word_index))
    embeding_matrix = np.zeros((MAX_WORDS+1, EMBED_SIZE))

    for word,i in word_index.items():
        if word not in embeding_dict: 
            continue
        if i > MAX_WORDS:
            break
        
        embeding_matrix[i] = embeding_dict[word]

    return embeding_matrix, MAX_WORDS
