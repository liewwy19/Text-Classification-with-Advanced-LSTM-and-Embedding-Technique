'''
    STEP-AI01: LIEW Wai Yip (liewwy19@gmail.com)
    2022.12.29
'''
#%%
# 1. Import library
import pandas as pd
import numpy as np
import re, os, datetime, json, pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
#%%
# 2. Data loading
CSV_PATH = os.path.join(os.getcwd(),'dataset','True.csv')
df = pd.read_csv(CSV_PATH)
#%%
# 3. Data inspection
print(df.head(2))
print(df.info())
print(df['text'][3])
print(df['subject'].value_counts())

# check for duplicates
print('Duplicates: ',df.duplicated().sum())
#%%
# 4. Data cleaning

# check for HTML-like tag
df['text'].str.contains('<.*?>').sum()

# remove: [x],(x),{x},<x>,@x,1st,2nd,3rd,xth
# remove news header
for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('^.*?\)\s*-|[\(\[\{].*?[\)\]\}]|<.*?>|@[^\s]+|\d+(st|nd|rd|th)|[^a-zA-Z]',' ',data).lower()     

df.drop_duplicates(inplace=True)


#%%
# 5. Features selection
text = df['text']
subject = df['subject']
#%%
# 6. Data preprocessing
    # - Tokenizer
    # - One-Hot Encoding
    # - Padding sequences
    # - Train-test-split

#Tokenizer
num_words = 5000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token) #instantiate the object

#to train the tokenizer
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

#%%
# to transform the text using tokenizer
text = tokenizer.texts_to_sequences(text)

# Padding
padded_text = pad_sequences(text, maxlen=200, padding='post',truncating='post')

#One hot encoder

ohe = OneHotEncoder(sparse=False)
subject = ohe.fit_transform(subject[::,None])

#%%
#expand dimension before feeding to train_test_split
padded_text = np.expand_dims(padded_text, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(padded_text,subject,test_size=0.2,random_state=123)

# 7. Model development
#%%
embedding_layer = 64

model = Sequential()
model.add(Embedding(num_words,embedding_layer))
model.add(LSTM(embedding_layer,return_sequences=True)) #,input_shape=(X_train.shape[1:])
model.add(Dropout(0.3)) # reduce the chance for overfitting
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.summary()
plot_model(model, show_shapes=True)
#%%
#Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 

#%%
#Create TensorBoard callback
log_path = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_path)
es = EarlyStopping(monitor='val_loss',patience=8, verbose=1, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# model training
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=64, epochs=50, callbacks=[tb,es,mc])
#%%
# 8. Model analysis

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','validation'])
plt.show()

y_predicted = model.predict(X_test)

y_predicted = np.argmax(y_predicted, axis=1)
y_actual = np.argmax(y_test, axis=1)

print(classification_report(y_actual,y_predicted))
cm = confusion_matrix(y_actual, y_predicted)
print(cm)

disp = ConfusionMatrixDisplay(cm)
dummy = disp.plot()
#%%
# 9. Model saving

# to save trained model
saved_path = os.path.join(os.getcwd(),'saved_model','model.h5')
model.save(saved_path) # save train model

# to save one hot encoder model
pickle_path = os.path.join(os.getcwd(),'saved_model','ohe.pkl')
with open(pickle_path,'wb') as f:
    pickle.dump(ohe,f)

# to save tokenizer
tokenizer_path = os.path.join(os.getcwd(),'saved_model','tokenizer.json')
token_json = tokenizer.to_json() # convert to json format 1st
with open(tokenizer_path,'w') as f:
    json.dump(token_json,f)

#%%
# 10. Model deployment

# please refer to deploy.py
