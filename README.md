# Disaster Response Pipeline Project

## Acknowledgements
Many thanks to Figure Eight and Udacity for curating such data sets and offering this hands-on impactful project.

## Motivation:
This repository contains my work to adress disaster response by building a web app that can ease emergency worker load to filter disaster messages. The figures below give a taste of the final project.

![Figure 1](https://raw.githubusercontent.com/aydi-abdelkarim/udacity-data-engineering/master/figures/figure.png)
![Figure 2](https://raw.githubusercontent.com/aydi-abdelkarim/udacity-data-engineering/master/figures/figure_2.png)


## Project structure
```
├── app                                                    # Web app files
│   ├── run.py                                             # Scrip to run to execute the app
│   └── templates
│       ├── go.html
│       └── master.html
├── data                                                    # Data directory
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── process_data.py
├── figures                                                 # Figure directory
│   ├── figure_2.png
│   └── figure.png
├── models                                                  
│   ├── classifier.pkl                                      # Serialized model
│   └── train_classifier.py                                 # SCript for training
├── notebooks
│   ├── ETL Pipeline Preparation.ipynb
│   └── ML Pipeline Preparation.ipynb
├── README.md

```

## Dependencies
In this work, I used these libraries along with Python=3.7.4:
- pandas
- matplotlib
- scikit-learn
- nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to clearrun your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

