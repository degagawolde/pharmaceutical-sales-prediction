
import numpy as np
import pandas as pd
import logging

logging.basicConfig(filename='../logfile.log', filemode='a',
                    encoding='utf-8', level=logging.DEBUG)

#read csv file and convert to dataframe
def load_data(path: str):
    try:
        df = pd.read_csv(path)
    except BaseException:
        logging.warning('file not found or wrong file format')
    return df
#return shaoe of a dataframe
def data_shape(df):
    print(f" There are {df.shape[0]} rows and {df.shape[1]} columns")

"""how many missing values exist or better still what is the % of missing values in the dataset?"""

#return data types in a dataframe
def data_types(df):
    t = df.dtypes.value_counts()
    return t

#get the n rows from a datframe value counts
def top_values(df: pd.DataFrame, column: str, top: int) -> None:
    values = df[column].value_counts()
    topdf = df.loc[df[column].isin(list(values[:top].index))]
    return topdf


def get_percentage_counts(df: pd.DataFrame, column: str):
    count_dict = df[column].value_counts()

    return pd.DataFrame(count_dict)
