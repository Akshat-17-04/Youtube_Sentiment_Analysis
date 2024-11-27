#IMPORTING OUR DATA FILE FROM COOMENTSSCRAPPER CODE BY IMPORTING IT
#from CoomentsScrapper import output_file

#IMPORTING OTHER NECCESSARY MODULES
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
stopwords.words("english")

#READING THE COMMENTS
youTube_data=pd.read_csv((r'C:\code\project\output_file'),encoding='ISO-8859-1')

#ADDING HEADING TO DATAFILE
column_names=['comments']
youTube_data=pd.read_csv(r'C:\code\project\output_file',names=column_names,encoding='ISO-8859-1')
youTube_data['comments'] = youTube_data['comments'].fillna('')

#DATA CLEANING PROCESS/PREPROCESSING
port_stemmer=PorterStemmer()
def stemming(content):
    stemmed_content=content
    stemmed_content=stemmed_content.lower()
    stemmed_content=re.sub('[^a-z]',' ',content)
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

#ADDING FILTERED COMMENTS TO DATASET
youTube_data['filtered_comments']=youTube_data['comments'].apply(stemming)

#ASSINGNING POLARITY SCORES TO THE COMMENTS
sentiments=SentimentIntensityAnalyzer()
youTube_data['postivie']=[sentiments.polarity_scores(i)['pos'] for i in youTube_data['filtered_comments']]
youTube_data['negative']=[sentiments.polarity_scores(i)['neg'] for i in youTube_data['filtered_comments']]
youTube_data['neutral']=[sentiments.polarity_scores(i)['neu'] for i in youTube_data['filtered_comments']]
youTube_data['compound']=[sentiments.polarity_scores(i)['compound'] for i in youTube_data['filtered_comments']]
score=youTube_data['compound'].values
sentiment=[]
for i in score:
    if i>=0.05:
        sentiment.append(1)
    elif i<= -0.05:
        sentiment.append(0)
    else:
        sentiment.append(1)
youTube_data['sentiment']=sentiment

#youTube_data.replace({'sentiment':{'Positive':output_file,'Negative':0,'Neutral':output_file}},inplace=True)


#ASSIGNING THE COMMENTS AND SENTIMENTS VALUES TO ARRAYS FOR FUTURE REFFERENCE
X=youTube_data['filtered_comments'].values
Y=youTube_data['sentiment'].values

#TRAIN_TEST_SPLIT(ASSIGNING TRAINING AND TEST DATA FOR OUR MODEL)
X_train,X_tesSt,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
X = youTube_data['filtered_comments'].values
Y = youTube_data['sentiment'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
vectoriser = TfidfVectorizer()
X_train = vectoriser.fit_transform(X_train)
X_test = vectoriser.transform(X_test)

# Save the vectorizer
with open('C:\\code\\project\\vectorizer_KNN.pkl', 'wb') as file:
    pickle.dump(vectoriser, file)
model=KNeighborsClassifier(n_neighbors=8)
model.fit(X_train,Y_train)


#CALCULATING THE ACCURACY OF OUR MODEL FOR BOTH TEST AND TRAIN DATA
X_train_Predction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_Predction,Y_train)

print("Accuracy score on train data is",training_data_accuracy)

X_test_Predction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_Predction,Y_test)

print("Accuracy score on test data is",test_data_accuracy)

#MODEL SAVING
filename='trained_model_KNN.sav'
pickle.dump(model,open(filename,'wb'))

