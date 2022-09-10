import numpy as np
import pandas as pd

import sys, os

import logging
from scipy.stats.mstats import winsorize

from scripts.setting_logs import get_rotating_log

class CleanData:
    def __init__(self):
        self.logger = get_rotating_log(
            filename='data_cleaner.log', logger_name='CleanDataFrameLogger')

    
    def convert_dtype(self, df: pd.DataFrame, columns, dtype):
        for col in columns:
            df[col] = df[col].astype(dtype=dtype)
        return df
    
    def format_float(self,value):
        return f'{value:,.2f}'
    
    def fix_missing_ffill(self, df: pd.DataFrame,col):
        df[col] = df[col].fillna(method='ffill')
        return df[col]
  
    def fix_missing_bfill(self, df: pd.DataFrame, col):
        df[col] = df[col].fillna(method='bfill')
        return df[col]
    
    def drop_column(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        for col in columns:
            df = df.drop([col], axis=1)
        return df
    
    def fill_mode(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        for col in columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df
    def fill_median(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        for col in columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def fix_outlier(self,df:pd.DataFrame, columns):
        for column in columns:
            df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
            
        return df

    def handle_outliers(self, df: pd.DataFrame,lower,upper):
       
        selected_columns = df.select_dtypes(include='float64').columns
        for col in selected_columns:
            df[col] = winsorize(df[col], (lower, upper))
        return df
    

    def get_mct(self,series: pd.Series, measure: str):
        """
        get mean, median or mode depending on measure
        """
        measure = measure.lower()
        if measure == "mean":
            return series.mean()
        elif measure == "median":
            return series.median()
        elif measure == "mode":
            return series.mode()[0]
        elif measure == 'zero':
            return 0

    def replace_missing(self,df: pd.DataFrame, columns: str, method: str=None, replace_with: any=None) -> pd.DataFrame:

        for column in columns:
            nulls = df[column].isnull()
            indecies = [i for i, v in zip(nulls.index, nulls.values) if v]
            if not replace_with:
                replace_with = self.get_mct(df[column], method)
            df.loc[indecies, column] = replace_with
            self.logger.info(f"Replacing missing values in column: {column} with method: {method}")

        return df