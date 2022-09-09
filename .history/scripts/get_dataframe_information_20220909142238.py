import logging
import pandas as pd
import numpy as np

import logging
import sys, os
import re

sys.path.append(os.path.abspath(os.path.join("./script")))
from scripts.get_missing_information import MissingInformation
 
class DataFrameInformation:
    
    def __init__(self):
        logging.basicConfig(filename='../logfile.log', filemode='a',
                            encoding='utf-8', level=logging.DEBUG)
        
    #calculate the skewness of the dataframe first
    def get_skewness(self,data:pd.DataFrame):
        skewness = data.skew(axis=0, skipna=True)
        df_skewness = pd.DataFrame(skewness)
        df_skewness = df_skewness.rename(
            columns={0: 'skewness'})
        
        return df_skewness

    #calculate skewness and missing value table
    def get_skewness_missing_count(self,data:pd.DataFrame):
        df_skewness = self.get_skewness(data)

        minfo = MissingInformation()

        mis_val_table_ren_columns = minfo.missing_values_table(data)
        df = pd.concat([df_skewness, mis_val_table_ren_columns], axis=1)
        df['Dtype'] = df['Dtype'].fillna('float64')
        df['% of Total Values'] = df['% of Total Values'].fillna(0.0)
        df['Missing Values'] = df['Missing Values'].fillna(0)
        df = df.sort_values(by='Missing Values', ascending=False)
        return df

    def get_column_with_string(self,df: pd.DataFrame, text):
        return [col for col in df.columns if re.findall(text, col) != []]

    def get_dataframe_information(self,df: pd.DataFrame):
        columns = []
        counts = []
        i = 0

        for key, item in df.isnull().sum().items():
            if item != 0:
                columns.append(key)
                counts.append(item)
                i += 1
        logging.info(
            'the dataset contain {} columns with missing values'.format(i))
        return pd.DataFrame({'column name': columns, 'counts': counts})

    def check_date_range(self,df: pd.DataFrame) -> None:
        """This function assumes df has a Date column and checks if 
        there are missing dates by counting unique dates.
        """
        assert 'Date' in df.columns, "`Date` is not a column in df"
        df['Date'] = pd.to_datetime(df['Date'])
        start_date, end_date = df['Date'].aggregate([min, max])
        print(
            f"start_date: {start_date.date()} ----> end_date: {end_date.date()}")
        unique_dates = df['Date'].unique()
        print(f"There are {len(unique_dates)} unique dates in the data.\n\
                The number of days between the end and start date is {(end_date-start_date).days}")
    def get_column_with_type(self,df:pd.DataFrame,dtype:list)->list:
        return df.select_dtypes(include=dtype).columns.tolist()
    
    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        categorical_columns = df.select_dtypes(
            include=['object', 'bool']).columns.tolist()
        return categorical_columns
