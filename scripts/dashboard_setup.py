from io import BytesIO, StringIO
import traceback
import pickle
# import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# from scripts.fetch_data import DataLoader
#import custome modules
# sys.path.append('../')

from scripts.get_missing_information import MissingInformation
from scripts.get_dataframe_information import DataFrameInformation
from scripts.ploting_utils import Plotters
from scripts.data_clean_handler import CleanData
from scripts import data_loader
from scripts.dvc_data_fetch import DataLoader
from scripts.feature_engineering import FeatureEngineering
from scripts.preprocess import encode

feng = FeatureEngineering()
dvc_load = DataLoader()
cleaner = CleanData()
minfo = MissingInformation()
dinfo = DataFrameInformation()
pltu = Plotters(6, 4)

import dvc.api


def merge_with_store(df: pd.DataFrame) -> pd.DataFrame:
    try:
        content = dvc.api.read(path="data/raw/store.csv",
                               repo='.',
                               rev='store_v1')
        store_df = pd.read_csv(StringIO(content))
        df = df.merge(store_df, on='Store', how='left')

    except:
        pass

    return df


def add_train_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = feng.transform(df)
    except:
        pass

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    count, cols = minfo.get_missing_entries_count(df)
    df = cleaner.replace_missing(df, cols, 'median')
    
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df = clean(df)
    df = merge_with_store(df)
    df = add_train_columns(df)
    df = clean(df)

    return df


def plot_predictions(date, sales):
    fig = plt.figure(figsize=(20, 7))
    ax = sns.lineplot(x=date, y=sales)
    ax.set_title("Predicted Sales", fontsize=24)
    ax.set_xlabel("Row index", fontsize=18)
    ax.set_ylabel("Sales", fontsize=18)

    return fig


def load_model(model_path: str = None):
    
    with dvc.api.open("models/model.pkl", "../", "lstm_v1", mode='rb') as model_file:
       
        model_file = BytesIO(model_file)
        model = pickle.load(open(model_file, 'rb'))

    return model

# convert series to supervised learning
def series_to_supervised(dataset, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(dataset) is list else dataset.shape[1]
    df = pd.DataFrame(dataset)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#read data from database
def get_features(engine,database=False,dvc=False):
    if database:
        features = pd.read_sql_table(
            'feutures',
            con=engine.connect())
        return features
    if dvc:
        repo = '../'
        version = 'trained_v3'
        data_path = '../data/cleaned/train.csv'
        return dvc_load.dvc_get_data(data_path, version, repo)
    else:
        return pd.read_csv('./data/cleaned/train.csv')
        
    return features
