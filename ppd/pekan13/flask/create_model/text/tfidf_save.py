
"""
Created on 2020-06-16
@author: Dr. Ganjar Alfian
@email: ganjar@dongguk.edu
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import joblib

def mlp(X, y):
    model=MLPClassifier(max_iter=500)
    model = model.fit(X, y)
    filename = 'mlp_tfidf.model'
    joblib.dump(model, filename)
    
if __name__ == '__main__':
    data='Tweets_100.csv'
    df = pd.read_csv(data)
    df_text = df['text'].astype(str)
    df_class = df['airline_sentiment']
    lines_clean = list()
    lines = df_text.values.tolist()
    porter=PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    for line in lines:
        tokens = word_tokenize(line)
        tokens = [w.lower() for w in tokens]
        #check alphabhet
        tokens = [word for word in tokens if word.isalpha()]
        #remove stopwords
        tokens = [word for word in tokens if not word in stopwords.words()]
                #stemming
        #tokens = [porter.stem(word) for word in tokens]
        #stemming
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
            #print(tokens)
        lines_clean.append(tokens)

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_class)
    df_y = label_encoder.transform(df_class)
    
    new_doc = list()
    for doc in lines_clean:
        row= ' '.join(doc)
        new_doc.append(row)
    
    vectorizer = CountVectorizer(decode_error="replace", max_features=1000)
    X_input = vectorizer.fit_transform(new_doc)
    #Save tfidf_vectorizer.vocabulary_
    joblib.dump(vectorizer.vocabulary_,open("vectorizer.model","wb"))
    print('successfully save the features')
    #print(vectorizer.vocabulary_)
    #print(X_input.toarray())
    print('Learning MLP')
    mlp(X_input, df_y)
    
