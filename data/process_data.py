import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data sets
    
    Args:
    messages_filepath: str: message file path
    categories_filepath: str: categories file path
    
    Returns:
    df: pd.DataFrame: data set of messages and categories
    """
    # load messages dataset
    df_messages = pd.read_csv(messages_filepath, index_col="id")
    
    # load categories dataset
    df_categories = pd.read_csv(categories_filepath,index_col="id")

    # merge datasets
    df = pd.merge(df_messages,df_categories, left_index=True, right_index=True)
    
    return df


def clean_data(df):
    """
    Clean the data set by:
        - Parsing categories as separate columns,
        - Renaming columns
        - Concatenating the resulting data frame with original one
    
    Args:
    df: pd.DataFRame, raw dataset
    Returns:
    df: pd.DataFRame, cleaned dataset
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # Extracting column names
    row = categories.iloc[0]
    category_colnames = list(map(lambda col: col[:-2] , row))
    
    # Renaming added columns
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)

    # Replace categories column with new ones
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # Removing duplicates
    df = df.drop_duplicates()
    
    return df

    
def save_data(df, database_filename):
    """
    Save data set df in SQL data base

    Args:
    df: pd.DataFrame: cleaned data set
    database_filename: str: data base file path
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('MessageToCategories', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()