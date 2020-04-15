# import libraries

import sys

# Data wrangling
import numpy as np
import pandas as pd
import pickle

# Machine Learning
## Text processing
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

## sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):

    """
    Load data from data base and split feautrs and labels in seprate arrayas

    Args:
        database_filepath: str, data base path
    Return:
        X: np.array, features matrix
        y: np.array, labels
        categories: list, label names
    """

    # load data from database
    df = pd.read_sql_table('MessageToCategories', con=f'sqlite:///{database_filepath}')

    categories = df.columns[3:]
    X = df[['message']].values[:,0]
    y = df[categories].values

    return X, y, categories

def tokenize(text, lemmatize=True):
    """
    Custom tokenizer for sklearn CountVectorizer

    Args:
    text: str,  raw text
    Return:
    tokens: list of str, list of lemmatized tokens
    """

    # Post-processing pipeline

    # Text normalization: Lower case
    text = text.lower()

    # Text cleaning: Delete URLs

    regex = "(http[s]?://\S+)"
    text = re.sub(regex,"[URL]",text)

    # Text Tokenization
    tokens = word_tokenize(text)

    # Text cleaning

    ## Delete punctuation
    regex = re.compile("\W")
    tokens = [t for t in tokens if not regex.match(t)]

    ## Delete single caracters
    regex = re.compile("^[a-z]$")
    tokens = [t for t in tokens if not regex.match(t)]

    ## Delete numbers
    regex = re.compile("^[0-9]+\W*$")
    tokens = [t for t in tokens if not regex.match(t)]

    ## Delete stopwords
    tokens = [t for t in tokens if not t in stopwords.words("english")]

    # Text Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def custom_scorer(y_pred, y_true):
    """
    Wrapper of evaluate function to use as metric in GridSearchCV
    The metric is the median across outputs of F1-score
    Args:
        y_true: np.array: ground truth labels
        y_pred: np.array: predicted labels
    Returns:
        score: float: metric

    """

    report = {"output":[],"f1":[]}
    n_output = y_true.shape[1]

    for i in range(n_output):

        y_pred_i = y_pred[:, i]
        y_true_i = y_true[:, i]
        report["output"].append(i)
        report["f1"].append(f1_score(y_pred_i,y_true_i, average='weighted'))

    score = pd.DataFrame(report).f1.median()
    return score



def build_model():
    """
    Build a processing and fine tuned machine Learning model

    Args:
    Return: pipeline, sklearn Estimator, ML model
    """

    pipeline = Pipeline([('extractor', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))])

    
    parameters = {'clf__estimator__max_depth': [5,10]}

    score = make_scorer(custom_scorer, greater_is_better=True)

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameters,
                               cv=3,
                               verbose=4,
                               n_jobs=-1,
                               scoring=score)

    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate score of model based on a test set

    Args:
        model: sklearn Estimator
        X_test: np.array, features
        Y_test: np.array, one hot labels
        category_names: list of str: named labels
    Returns
    """

    # Predict using model
    y_pred = model.predict(X_test)

    report = {"output":[],
              "accuracy":[],
              "precision":[],
              "recall":[],
              "f1":[]}

    for i in range(len(category_names)):

        y_pred_i = y_pred[:, i]
        y_true_i = Y_test[:, i]
        report["output"].append(category_names[i])
        report["accuracy"].append(accuracy_score(y_pred_i,y_true_i))
        report["precision"].append(precision_score(y_pred_i,y_true_i, average='weighted'))
        report["recall"].append(recall_score(y_pred_i,y_true_i, average='weighted'))
        report["f1"].append(f1_score(y_pred_i,y_true_i, average='weighted'))

    return pd.DataFrame(report)



def save_model(model, model_filepath):
    """
    Save fine tuned model for inference
    Args:
        model: sklearn estimator - full CV gridsearch pipeline
        model_filepath: str, serialized model path
    Returns
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model.best_estimator_, file)


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
        test_report = evaluate_model(model, X_test, Y_test, category_names)

        print("Test score: {:.4f}\n\n".format(test_report.f1.median()))
        print("Test: detailed evaluation")
        print(test_report)

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
