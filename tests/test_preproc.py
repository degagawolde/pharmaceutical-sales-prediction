from dvc_data_fetch import DataLoader
import preprocess as pr
import data_loader as dl
from tkinter.filedialog import test
import unittest

import pandas as pd
import logging

import sys
import os

sys.path.append(os.path.abspath(os.path.join("./scripts")))


dvc_load = DataLoader()

logging.basicConfig(filename='./logfile.log', filemode='a',
                    encoding='utf-8', level=logging.DEBUG)
