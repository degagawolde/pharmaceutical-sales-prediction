from tkinter.filedialog import test
import unittest

import pandas as pd
import logging

import sys, os

sys.path.append(os.path.abspath(os.path.join("./scripts")))

from dvc_data_fetch import DataLoader

dvc_load = DataLoader()


# First load the cleaned stores data
data_path = 'data/cleaned/store.csv'
version = 'eaf6d2e8828a365523b9b8800a79d45c6b236368'
repo = 'https://github.com/degagawolde/pharmaceutical-sales-prediction'

store_df = dvc_load.dvc_get_data(data_path, version, repo)

# Then load the raw sales data
data_path = 'data/merged/train.csv'
version = '2c43bd5e314858f9b692a7d67fc894e8c04b282a'
train_df = dvc_load.dvc_get_data(data_path, version, repo)

# Finally load the test data
data_path = 'data/merged/test.csv'
version = 'ecc063bb453b6ca9e10fb2341d58ebbfe40b2514'
test_df = dvc_load.dvc_get_data(data_path, version, repo)

class TestGetInformations(unittest.TestCase):
    # def setUp(self):
    
    def test_load_store(self):
       self.assertIsInstance(store_df,pd.DataFrame)
       
    def test_load_train(self):
       self.assertIsInstance(train_df,pd.DataFrame)
       
    def test_load_test(self):
       self.assertIsInstance(test_df,pd.DataFrame)
        
        
if __name__ == "__main__":
    unittest.main()
