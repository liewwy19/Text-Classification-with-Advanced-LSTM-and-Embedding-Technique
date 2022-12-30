#%% 
# Import Library
import re, json, pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#%% 
# Data loading
new_text = '''
Here is an example of a good movie review featuring Mr. Warren as the main character in a fantasy genre film:

"I was pleasantly surprised by the new fantasy film featuring Mr. Warren. The plot was engaging and the special effects were top-notch. Mr. Warren's performance as the lead character was strong and he really brought the role to life. The supporting cast was also excellent and their chemistry added an extra layer of depth to the film. Overall, I highly recommend this movie to fans of the fantasy genre. It's definitely worth the watch.
'''
new_text = [new_text]

#%% 
# Data cleaning

#need to remove punctuations and HTML tags and to convert into lowercase
for index, data in enumerate(new_text):
    new_text[index] = re.sub('<.*?>','',data)
    new_text[index] = re.sub('[^a-zA-Z]',' ',new_text[index]).lower()

#%% 
# Data preprocessing

#Load tokenizer
with open('saved_model/tokenizer.json','r') as f:
    loaded_tokenizer = json.load(f)

loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
new_text = loaded_tokenizer.texts_to_sequences(new_text)

# paddign sequences
new_text = pad_sequences(new_text, maxlen=200, padding='post',truncating='post')

#%% Deployment

# to load the saved model
loaded_model = load_model('saved_model/model.h5')
loaded_model.summary()

# to load ohe model
with open('saved_model/ohe.pkl','rb') as f:
    loaded_ohe = pickle.load(f)

output = loaded_model.predict(new_text)

print(output)
print(loaded_ohe.get_feature_names_out())
print(loaded_ohe.inverse_transform(output))
# %%
