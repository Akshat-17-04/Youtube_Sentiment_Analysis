import pickle
import pandas as pd
import re
from CoomentsScrapper import output_file
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#LOADING THE MODEL AS PER YOUR CHOICE
with open('C:\\code\\project\\trained_model_SVM.sav', 'rb') as file:
    loaded_model = pickle.load(file)

#LOADING THE VECTORIZER(AS PER YOUR CHOICE OF MODEL)
with open('C:\\code\\project\\vectorizer_SVM.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

#DATAFRAME CREATION BY USING OUTPUTFILE FROM COMMENTSCRAPPER CODE
column_names=['comments']
new_df=pd.read_csv(output_file,names=column_names,encoding='ISO-8859-1')
new_df['comments'] = new_df['comments'].fillna('')
#PREPROCESSING DATA
port_stem = PorterStemmer()
def preprocess_text(content):
    stemmed_content=content
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = content.lower()
    stemmed_content = content.split()
    stemmed_content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
    stemmed_content=''.join(stemmed_content)
    return stemmed_content

#APPLYING PREPROCESSING TO DATASET
new_df['processed_comments'] = new_df['comments'].apply(preprocess_text)

#CONVERTING DATSET TO BINARY FORM
X_new = vectorizer.transform(new_df['processed_comments'])

#PREDICTING THE SENTIMENTS
predictions = loaded_model.predict(X_new)

#ADDING THE PREDICTIONS TO THE DATAFRAME
new_df['sentiment'] = predictions

#PRINTING THE SENTIMENTS
print(new_df['sentiment'])


#TEST INPUT COMMENTS VIDEO ID
#4gulVzzh82g