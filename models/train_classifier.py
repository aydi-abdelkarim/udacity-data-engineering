# import libraries

import sys

# Data wrangling
import numpy as np
import pandas as pd
import pickle


# Machine Learning

## Text processing
import nltk
nltk.download(['punkt', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

## sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):

    """
    Load data from data base and split feautrs and labels in seprate arrayas
    
    Args:
        database_filepath: str, data base path
    Return
    """
    
    
    # load data from database
    df = pd.read_sql_table('MessageToCategories', con='sqlite:///DisasterResponse.db')
    categories = df.columns[3:]
    X = df[['message']].values[:,0]
    y = df[categories].values


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()