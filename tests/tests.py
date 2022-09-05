from tkinter.filedialog import test
import unittest

import pandas as pd
import logging

import sys, os

sys.path.append(os.path.abspath(os.path.join("./scripts")))

import data_loader
import preprocess

logging.basicConfig(filename='./logfile.log', filemode='a',
                    encoding='utf-8', level=logging.DEBUG)

try:
    test_data = pd.read_csv('./data/AdSmartABdata.csv')
except BaseException:
    logging.warning('file not found or wrong file format')

class TestGetInformations(unittest.TestCase):
    # def setUp(self):
        
    def test_top_values(self):
        self.assertIsInstance(data_loader.top_values(
            test_data, 'yes',3),pd.DataFrame )
    
    def test_load_data(self):
       self.assertIsInstance(data_loader.load_data(
           './data/AdSmartABdata.csv'),pd.DataFrame)
    def test_encode(self):
        self.assertIsInstance(preprocess.encode(
            test_data,'browser'),pd.DataFrame)
    
    def test_hot_encode(self):
        self.assertIsInstance(preprocess.hot_encode(
            test_data,'browser'),pd.DataFrame)
    def test_nonResponse(self):
       self.assertIsInstance(preprocess.nonResponse(
           test_data), pd.DataFrame)
        
    def test_remove(self):
        self.assertEqual(len(preprocess.remove(
            test_data,['auction_id','date']).columns),len(test_data.columns)-2)
        
if __name__ == "__main__":
    unittest.main()