import dvc.api

from scripts.dvc_data_fetch import DataLoader

dvc_load = DataLoader()

# First load the cleaned stores data
data_path = 'data/cleaned/store.csv'
version = 'store_v2'
repo = './'

store_df = dvc_load.dvc_get_data(data_path, version, repo)
