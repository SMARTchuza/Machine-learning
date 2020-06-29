import pymongo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
dataset=pd.read_csv('hotel-reviews.csv')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
def cleanData(sentence):
    processedList = ""
    
    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)
    sentence = re.sub(r'[^A-Za-z0-9\s.]',r'',str(sentence).lower())
    sentence = re.sub(r'\n',r' ',sentence)
    
    # remove stop words
    sentence = " ".join([word for word in sentence.split() if word not in stopWords])
    
    return sentence
corpus=[]
for i in range(38932):
    x=cleanData(dataset['Description'][i])
    corpus.append(x)
from sklearn.model_selection import train_test_split
y=dataset['Is_Response']

X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
tvec=TfidfVectorizer()
clf2=LogisticRegression(solver="lbfgs")
from sklearn.pipeline import Pipeline
model = Pipeline([('vectorizer',tvec),('classifier',clf2)])
model.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_pred,y_test)

client = pymongo.MongoClient("mongodb://tanmay:tanmay@cluster0-shard-00-00-byvwx.mongodb.net:27017,cluster0-shard-00-01-byvwx.mongodb.net:27017,cluster0-shard-00-02-byvwx.mongodb.net:27017/hotel_management?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true&w=majority")
db = client.get_database('hotel_management')
review=db.reviews
print(review.count_documents({}))
examples=list(review.find())
final_check=[]
for obj in examples:
    if obj['status']=='inqueue':
        Ans_want=model.predict([obj['text']])
        print(Ans_want[0])
        review.update_one({'text':obj['text']},{"$set":Ans_want[0]})



# ans=model.predict(example)