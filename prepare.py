import pandas as pd
import numpy as np

import unicodedata
import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import string as st

from sklearn.model_selection import train_test_split


# Must run acquire to obtain the json file 
# Specify the path to your JSON file


######################################### PREPARE #########################################

def prepare_data(df = pd.read_json('data2.json')):
    '''This function takes in a df and returns a dataframe with cleaned, stemmed,
      and target columns added.'''
    
    # Cleans the text by removing characters, stopwords and tokenizing
    df['clean_text'] = df['readme_contents'].apply(lambda string: remove_stopwords(tokenize(clean_strings(string))))
    # 
    df['stem'] = df['clean_text'].apply(lambda string: stem(string))

    df['lemmatize'] = df['clean_text'].apply(lambda string: lemmatize(string))

    df['target'] = df['language'].apply(lambda val: 1 if val == 'Python' else (2 if val == 'JavaScript' else 0))

    df = df.drop_duplicates()

    df.reset_index(drop=True, inplace=True)

    return df


def clean_strings(string):
    '''This function takes in a string makes the characters lowercase 
    and removes non alphanumeric characters.'''

    clean_str = string.lower()
    # Remove punctuation
    clean_str = clean_str.translate(str.maketrans('', '', st.punctuation))
    # Remove numbers
    clean_str = re.sub(r'\d+', '', clean_str)
    # Remove extra whitespaces
    clean_str = ' '.join(clean_str.split())
    clean_str = unicodedata.normalize('NFKD', clean_str)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    clean_str = re.sub(r"[^a-z0-9'\s]", '', clean_str)

    return clean_str

def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    # make our tokenizer, taken from nltk's ToktokTokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # apply our tokenizer's tokenization to the string being input, ensure it returns a string
    string = tokenizer.tokenize(string, return_str = True)
    
    return string

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    ADDITIONAL_STOPWORDS = []

    # assign our stopwords from nltk into stopword_list
    stopword_list = stopwords.words('english') + ADDITIONAL_STOPWORDS
    # utilizing set casting, i will remove any excluded stopwords
    stopword_set = set(stopword_list) - set(exclude_words)
    # add in any extra words to my stopwords set using a union
    stopword_set = stopword_set.union(set(extra_words))
    # split our document by spaces
    words = string.split()
    # every word in our document, as long as that word is not in our stopwords
    filtered_words = [word for word in words if word not in stopword_set]
    # glue it back together with spaces, as it was so it shall be
    string_without_stopwords = ' '.join(filtered_words)
    # return the document back
    return string_without_stopwords


def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # create our stemming object
    ps = nltk.porter.PorterStemmer()
    # use a list comprehension => stem each word for each word inside of the entire document,
    # split by the default, which are single spaces
    stems = [ps.stem(word) for word in string.split()]
    # glue it back together with spaces, as it was before
    string = ' '.join(stems)
    
    return string


def lemmatize(string):
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    # create our lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    # use a list comprehension to lemmatize each word
    # string.split() => output a list of every token inside of the document
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    # glue the lemmas back together by the strings we split on
    string = ' '.join(lemmas)
    #return the altered document
    return string


####################################### SPLIT THE DATA #######################################


    
def split_data(df = prepare_data(), test_size=.10,
               validate_size=.10, stratify_col=None, random_state=123):
    '''
    take in a DataFrame and return train, validate, and test DataFrames;
    return train, validate, test DataFrames.
    '''
    
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, test_size=test_size, random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, test_size=validate_size/(1-test_size),
                                                                           random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df, test_size=test_size,
                                                random_state=random_state, stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, test_size=validate_size/(1-test_size),
                                           random_state=random_state, stratify=train_validate[stratify_col])       
    return train, validate, test


def X_y_split(train = split_data()[0],
              validate = split_data()[1],
              test = split_data()[2]):

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['target'])
    y_validate = validate['target']
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['target'])
    y_test = test['target']
    
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


