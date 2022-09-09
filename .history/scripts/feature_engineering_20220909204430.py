from types import NoneType
import pandas as pd
from scripts.setting_logs import get_rotating_log


logger = get_rotating_log(
    filename='preprocessing.log', logger_name='FeatureEngineeringLogger')


class FeatureEngineering:
    def __init__(self, df=None, holidays=None):
        self.df = df
        self.holidays = holidays
        self.columns_to_drop = ['Customers']

    def drop_columns(self) -> None:
        self.df = self.df.drop(self.columns_to_drop,axis=1)
        logger.info(
            f"Dropped {len(self.columns_to_drop)} columns since they are not in the test data")
        
    def transform(self, df: pd.DataFrame = None):
        if not isinstance(df, NoneType):
            self.df = df.copy()
        assert 'Date' in self.df.columns
        self.drop_columns()
        self._set_holidays()
        self.generate_columns()
        
        logger.info("Feature enginerring completed")

        return self.df

    def generate_columns(self,df:pd.DataFrame) -> None:
        """Adds date related categorical columns to the dataframe"""

        self.create_holiday_distance_cols()
        df.loc[:, ['Year']] = df['Date'].dt.year
        df.loc[:, ['Month']] = df['Date'].dt.month
        df.loc[:, ['WeekOfYear']] = df['Date'].dt.isocalendar().week
        df.loc[:, ['is_month_end']] = df['Date'].dt.is_month_end
        df.loc[:, ['is_month_start']] = df['Date'].dt.is_month_start
        df.loc[:, ['is_quarter_end']] = df['Date'].dt.is_quarter_end
        df.loc[:, ['is_quarter_start']
                    ] = df['Date'].dt.is_quarter_start
        df.loc[:, ['is_year_end']] = df['Date'].dt.is_year_end
        df.loc[:, ['is_year_start']] = df['Date'].dt.is_year_start
        logger.info("9 new columns added to the dataframe")

    def create_holiday_distance_cols(self,df:pd.DataFrame) -> None:
        df.loc[:, ['DistanceToNextHoliday']] = pd.NA
        df.loc[:, ['DistanceFromPrevHoliday']] = pd.NA
        unique_dates = pd.to_datetime(self.df.Date, format = '%Y-%m-%d') .unique()
        for date in unique_dates:
            after_holiday, to_next_holiday = self._get_holiday_distances(date)
            indecies = df[df['Date'] == date].index
            df.loc[indecies, 'DistanceToNextHoliday'] = to_next_holiday
            df.loc[indecies, 'DistanceFromPrevHoliday'] = after_holiday
        df.loc[:, ['DistanceToNextHoliday']] = df['DistanceToNextHoliday'].astype('int')
        df.loc[:, ['DistanceFromPrevHoliday']] = df['DistanceFromPrevHoliday'].astype('int')

    def _set_holidays(self) -> None:
        """Filters the holiday dates from a given dateframe"""
        self.holidays = pd.to_datetime(self.df.query(
            "StateHoliday in ['a', 'b', 'c']")['Date'], format='%Y-%m-%d').dt.date.unique()

        self.holidays.sort()
        logger.info(
            f"generatd holidays")

    def _get_holiday_distances(self, date) -> list[int, int]:
        """takes in a date, then tells me it's distance on both dxns for the closest holiday"""
        previous, upcoming = self._get_neighbors(date)

        after_holiday = date - previous

        to_next_holiday = upcoming - date
        logger.info(
            f"generated holidays distance")

        return int(after_holiday.days), int(to_next_holiday.days)

    def _get_neighbors(self, date) -> list[pd.to_datetime, pd.to_datetime]:
        """uses a sorted list of dates to get the neighboring 
        dates for a date. 
        """
        date = pd.to_datetime(date)
        original_year = None
        if date.year >= self.holidays[-1].year:
            original_year = date.year
            # Assume the date given is in 2014
            date = pd.to_datetime(f"2014-{date.month}-{date.day}")
        previous, upcoming = None, None
        for i, d in enumerate(self.holidays):
            if d >= date.date():
                previous = pd.to_datetime(self.holidays[i-1])
                upcoming = pd.to_datetime(self.holidays[i])
                if original_year:
                    previous = pd.to_datetime(
                        f"{original_year}-{previous.month}-{previous.day}")
                    upcoming = pd.to_datetime(
                        f"{original_year}-{upcoming.month}-{upcoming.day}")
                return previous, upcoming
