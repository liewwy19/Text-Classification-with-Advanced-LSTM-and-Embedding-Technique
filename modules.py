import re
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding

def text_cleaning(text):
    
    # URL (bit.ly/asdjkhk)
    text = re.sub('bit.ly/\d\w{1,10}','',text)
    # @realDonaldTrump
    text = re.sub('@[^\s]+','',text)
    # WASHINGTON (Reuter)
    text = re.sub('^.*?\)\s*-','',text)
    # [1901 EST]
    text = re.sub('\[.*?EST\]','',text)
    #[^a-zA-Z]
    text = re.sub('[^a-zA-Z]',' ',text).lower()

    return text


def lstm_model_creation(num_words,nb_classes,embedding_layer=64,dropout=0.3,num_neurons=64):
    
    model = Sequential()
    model.add(Embedding(num_words,embedding_layer))
    model.add(LSTM(embedding_layer,return_sequences=True)) #,input_shape=(X_train.shape[1:])
    model.add(Dropout(dropout)) # reduce the chance for overfitting
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation='softmax'))
    model.summary()

    plot_model(model, show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model