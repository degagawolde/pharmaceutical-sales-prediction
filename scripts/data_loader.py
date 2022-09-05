
import numpy as np
import pandas as pd
import logging

logging.basicConfig(filename='./logfile.log', filemode='a',
                    encoding='utf-8', level=logging.DEBUG)

def load_data(path: str):
    try:
        df = pd.read_csv(path)
    except BaseException:
        logging.warning('file not found or wrong file format')
    return df

def data_shape(df):
    print(f" There are {df.shape[0]} rows and {df.shape[1]} columns")

"""how many missing values exist or better still what is the % of missing values in the dataset?"""


def data_types(df):
    t = df.dtypes.value_counts()
    return t

def percent_missing(df):

    # Calculate total number of cells in dataframe
    totalCells = np.product(df.shape)

    # Count number of missing values per column
    missingCount = df.isnull().sum()

    # Calculate total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    print("The dataset contains", round(
        ((totalMissing/totalCells) * 100), 2), "%", "missing values.")

def top_values(df: pd.DataFrame, column: str, top: int) -> None:
    values = df[column].value_counts()
    topdf = df.loc[df[column].isin(list(values[:top].index))]
    return topdf
