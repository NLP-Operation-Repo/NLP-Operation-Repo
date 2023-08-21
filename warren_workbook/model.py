import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer

import sys
# allow modules from parent directory to be imported
sys.path.append('..')


import prepare as p


def run_logistic_model(X_train = p.X_y_split()[0], y_train = p.X_y_split()[1], X_validate= p.X_y_split()[2], y_validate = p.X_y_split()[3], feature_column = 'stem' ):
    '''This function takes the split data and feature column (stem or lemmatized), vectorizes the features, fits and tranforms a Logistic Regression Model,
    and prints out the scores for train and validate.'''

    #set column for modeling
    X_train_model = X_train[feature_column]
    X_validate_model = X_validate[feature_column]

    #make the vectorizer
    tfidf = TfidfVectorizer()

    #fit and transform the vectorizer
    X_train_model = tfidf.fit_transform(X_train_model)
    X_validate_model = tfidf.transform(X_validate_model)

    #create df of actual values
    train_lm = pd.DataFrame(dict(actual=y_train))
    validate_lm = pd.DataFrame(dict(actual=y_validate))
    

    #fit the model
    lm = LogisticRegression(random_state=123).fit(X_train_model, y_train)

    #add the predicted values to df
    train_lm['predicted'] = lm.predict(X_train_model)
    validate_lm['predicted'] = lm.predict(X_validate_model)
    

    #print train
    print('Train')
    print('Accuracy: {:.2%}'.format(accuracy_score(train_lm.actual, train_lm.predicted)))
    print('---')
    print(classification_report(train_lm.actual, train_lm.predicted))
    print('---')


    #print validate
    print('Validate')
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_lm.actual, validate_lm.predicted))) 
    print('---')
    print(classification_report(validate_lm.actual, validate_lm.predicted))
    print('---')



def run_decisiontree_model(X_train = p.X_y_split()[0], y_train = p.X_y_split()[1], X_validate= p.X_y_split()[2], y_validate = p.X_y_split()[3], feature_column = 'stem'):
    '''This function takes the split data and feature column (stem or lemmatized), vectorizes the features, fits and tranforms a Decision Tree Model,
    and prints out the scores for train and validate.'''

    #set column for modeling
    X_train_model = X_train[feature_column]
    X_validate_model = X_validate[feature_column]

    #make the vectorizer
    tfidf = TfidfVectorizer()

    #fit and transform the vectorizer
    X_train_model = tfidf.fit_transform(X_train_model)
    X_validate_model = tfidf.transform(X_validate_model)

    #create df of actual values
    train_clf = pd.DataFrame(dict(actual=y_train))
    validate_clf = pd.DataFrame(dict(actual=y_validate))
    

    #fit the model
    clf = DecisionTreeClassifier(max_depth=3, random_state=123).fit(X_train_model, y_train)

    #add the predicted values to df
    train_clf['predicted'] = clf.predict(X_train_model)
    validate_clf['predicted'] = clf.predict(X_validate_model)
    

    #print train
    print('Train')
    print('Accuracy: {:.2%}'.format(accuracy_score(train_clf.actual, train_clf.predicted)))
    print('---')
    print(classification_report(train_clf.actual, train_clf.predicted))
    print('---')


    #print validate
    print('Validate')
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_clf.actual, validate_clf.predicted))) 
    print('---')
    print(classification_report(validate_clf.actual, validate_clf.predicted))
    print('---')


def run_randomforest_model(X_train = p.X_y_split()[0], y_train = p.X_y_split()[1], X_validate= p.X_y_split()[2], y_validate = p.X_y_split()[3], feature_column = 'stem'):
    '''This function takes the split data and feature column (stem or lemmatized), vectorizes the features, fits and tranforms a RandomForest Model,
    and prints out the scores for train and validate.'''    

   #set column for modeling
    X_train_model = X_train[feature_column]
    X_validate_model = X_validate[feature_column]

    #make the vectorizer
    tfidf = TfidfVectorizer()

    #fit and transform the vectorizer
    X_train_model = tfidf.fit_transform(X_train_model)
    X_validate_model = tfidf.transform(X_validate_model)

    #create df of actual values
    train_rf = pd.DataFrame(dict(actual=y_train))
    validate_rf = pd.DataFrame(dict(actual=y_validate))
    

    #fit the model
    rf = RandomForestClassifier(max_depth=3, min_samples_leaf=3, random_state=123).fit(X_train_model, y_train)

    #add the predicted values to df
    train_rf['predicted'] = rf.predict(X_train_model)
    validate_rf['predicted'] = rf.predict(X_validate_model)
    

    #print train
    print('Train')
    print('Accuracy: {:.2%}'.format(accuracy_score(train_rf.actual, train_rf.predicted)))
    print('---')
    print(classification_report(train_rf.actual, train_rf.predicted))
    print('---')


    #print validate
    print('Validate')
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_rf.actual, validate_rf.predicted))) 
    print('---')
    print(classification_report(validate_rf.actual, validate_rf.predicted))
    print('---')


def run_naivebayes_model(X_train = p.X_y_split()[0], y_train = p.X_y_split()[1], X_validate= p.X_y_split()[2], y_validate = p.X_y_split()[3], feature_column = 'stem'):
    '''This function takes the split data and feature column (stem or lemmatized), vectorizes the features, fits and tranforms a Multinomial Naive Bayes,
    and prints out the scores for train and validate.'''    

   #set column for modeling
    X_train_model = X_train[feature_column]
    X_validate_model = X_validate[feature_column]

    #make the vectorizer
    tfidf = TfidfVectorizer()

    #fit and transform the vectorizer
    X_train_model = tfidf.fit_transform(X_train_model)
    X_validate_model = tfidf.transform(X_validate_model)

    #create df of actual values
    train_nb = pd.DataFrame(dict(actual=y_train))
    validate_nb = pd.DataFrame(dict(actual=y_validate))
    

    #fit the model
    nb = MultinomialNB().fit(X_train_model, y_train)

    #add the predicted values to df
    train_nb['predicted'] = nb.predict(X_train_model)
    validate_nb['predicted'] = nb.predict(X_validate_model)
    

    #print train
    print('Train')
    print('Accuracy: {:.2%}'.format(accuracy_score(train_nb.actual, train_nb.predicted)))
    print('---')
    print(classification_report(train_nb.actual, train_nb.predicted))
    print('---')


    #print validate
    print('Validate')
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_nb.actual, validate_nb.predicted))) 
    print('---')
    print(classification_report(validate_nb.actual, validate_nb.predicted))
    print('---')


def run_final_test_model(X_train = p.X_y_split()[0], y_train = p.X_y_split()[1], X_test=p.X_y_split()[4], y_test=p.X_y_split()[5], feature_column = 'stem'):
    '''This function runs the top performing model Logistic Regression'''

    #set column for modeling
    X_train_model = X_train[feature_column]
    X_test_model = X_test[feature_column]

    #make the vectorizer
    tfidf = TfidfVectorizer()

    #fit and transform the vectorizer
    X_train_model = tfidf.fit_transform(X_train_model)
    X_test_model = tfidf.transform(X_test_model)

    #fit the model
    lm = LogisticRegression(random_state=123).fit(X_train_model, y_train)

    #make df of test values
    test_lm = pd.DataFrame(dict(actual=y_test))

    #add the predicted values to df
    test_lm['predicted'] = lm.predict(X_test_model)

    #save to csv
    test_lm.to_csv('test_pred.csv',index=False)

    # accuracy of rf on test data
    print('Accuracy of logistic regression classifier on test set: {:.2f}'
     .format(lm.score(X_test_model, y_test)))