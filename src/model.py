from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Conv1D
from keras.callbacks import Callback

def get_model(input_shape, MAX_WORDS, embeding_matrix, MAX_LENGTH=150, EMBED_SIZE=300):
    
    # inp = Input(shape=(MAX_LENGTH,))
    
    input = Input(shape = input_shape, name='InputLayer')

    x = Embedding(MAX_WORDS+1, EMBED_SIZE, weights=[embeding_matrix], trainable=False)(input)
    x = SpatialDropout1D(rate = 0.5)(x)
    x = Bidirectional(GRU(MAX_LENGTH, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    conc = concatenate([max_pool, avg_pool])
    output = Dense(6, activation='sigmoid')(conc)
    
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model