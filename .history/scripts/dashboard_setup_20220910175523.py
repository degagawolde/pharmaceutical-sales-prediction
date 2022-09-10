from io import StringIO
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
    count, cols = analyzer.get_missing_entries_count(df)
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
    # model_file = dataloader.dvc_get_data("models/model.pkl", 'rf-reg-v1', '.')
    with dvc.open("models/model.pkl", "github.com/Hen0k/Rossmann-Pharmaceuticals-Sales-Forcast", "rf-reg-v1", mode='rb') as model_file:
        # print(type(model_file))
        # model_file = BytesIO(model_file)
        # with open(model_path, 'rb') as f:
        model = pickle.load(model_file)
        # model = pickle.load(BytesIO(model_file))

    return model
