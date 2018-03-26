
# coding: utf-8

#Importing required libraries
import pandas as pd
import nltk
import csv

#Reading the training data in the form of CSV file
file = pd.read_csv("train.dat", sep="\t", header=None)

#Importing the CounterVectorizer transformer
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
count = vector.fit_transform(file[1])

#Importing the TfidfTransformer transformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(count)

#Building the MultinomialNB classifier , although this was not used in final calculation
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_tfidf, file[0])

#Importing other classifiers
from sklearn.svm import LinearSVC
from sklearn.svm import SVC , NuSVC
from sklearn.linear_model import LogisticRegression

#Building the sklearn pipeline with corresponding transformers and a classifier along with passing ngrams as the parameters
from sklearn.pipeline import Pipeline
txt_classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4), analyzer='char')),
                     ('tfidf', TfidfTransformer()),
                     ('dtc', LinearSVC()),
])


#Training the model
txt_classifier = txt_classifier.fit(file[1], file[0])


#Testing the model on test data
t = open("test.dat")

#Generating the predict file
predicted = txt_classifier.predict(t)
print(predicted)

#Storing the predicted output in result
result = pd.DataFrame(predicted)

#Making a prediction.dat file for the final result
result.to_csv('prediction.dat', index=False, header=None)

