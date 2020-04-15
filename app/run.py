import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageToCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    categories = df.columns[3:]
    data = df[categories].sum().sort_values(ascending=True)
    labels = data.index.values
    freqs = data.values
    
    categories = df.columns[3:]
    data = df[categories].sum().sort_values(ascending=True)
    labels = data.index.values
    freqs = data.values
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=labels,
                    y=freqs
                )
            ],

            'layout': {
                'title': 'Labels distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                }
            }
        },
           {
  "data": [
    {
      "values": genre_counts,
      "labels": genre_names,
      "domain": {"x": [0, .5]},
      "name": 'Label',
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":'Genre distribution'
    }
}
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()