""" Import necessary libraries """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

"""Data Preprocessing"""

#Load dataset using pandas

raw_mail_data=pd.read_csv('spam_ham_dataset.csv')

#replacing the null values with null string

mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.head()


#label spam mail as 0 and ham mail as 1

mail_data.loc[mail_data['label'] == 'spam', 'label',] = 0
mail_data.loc[mail_data['label'] == 'ham', 'label',] = 1

X = mail_data['text']
Y = mail_data['label']


"""Train Test Split"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, train_size=0.8,test_size=0.2,random_state=3)


"""Feature Extraction"""

#transform text to feature vectors to feed SVM model using TfidVectorizer
#convert all text to lower case

feature_extract = TfidfVectorizer( min_df=1, stop_words ='english', lowercase='True')
X_train_features = feature_extract.fit_transform(X_train)
X_test_features  = feature_extract.transform(X_test)


#convert Y values to int


Y_train = Y_train.astype('int')
Y_test  = Y_test.astype('int')


"""Training the SVM model"""


model=LinearSVC().fit(X_train_features, Y_train)


"""Evaluation of model"""


predict_on_train_data=model.predict(X_train_features)
accuracy_train = accuracy_score(Y_train, predict_on_train_data)
print(accuracy_train)

predict_on_test_data = model.predict(X_test_features)
accuracy_on_test = accuracy_score(Y_test, predict_on_test_data)
print(accuracy_on_test)


"""Testing model by user"""


mail=input()
mail=[mail]

#convert to feature vector

mail_feature = feature_extract.transform(mail)
predict_mail=model.predict(mail_feature)
if predict_mail == 0:
  print('Spam Mail')
else:
  print('Ham Mail')