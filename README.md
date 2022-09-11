# pharmaceutical-sales-prediction-for-multiple-stores
![rossmann](https://searchlogovector.com/wp-content/uploads/2020/04/rossmann-mein-drogeriemarkt-logo-vector.png)

The finance team at ***Rossman*** wants to forecast sales in all thier stores across several cities six weeks ahead of time. This project is all about building and serving an end-to-end product that delivers this prediction to analysts in the finance team.

# Exploration of customer purchasing behavior
### Store data exploration
- EDA regarding the store.csv data is done in [***notebooks/StoreDataExploration.ipynb***](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/notebooks)
### Sales data exploration
- EDA regarding the train.csv and test.csv data is done in [***notebooks/SalesDataExploration.ipynb***](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/notebooks)
- EDA for the questions guided analysis in done in [***notebooks/ DataExploration.ipynb***](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/notebooks)
### Logging
[Logging](https://docs.python.org/3/howto/logging.html) is a means of tracking events that happen when some software runs.
```
import logging
logging.warning('Watch out!')  # will print a message to the console
logging.info('I told you so')
```
# Prediction of store sales
### Preprocessing
Othor features like year, month, distance(after or before) from a holiday and othor features can be generated. [notebooks/FeatureEngineering.ipynb](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/notebooks) and [scripts/feature_engineering.py](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/scripts)
### Building models with sklearn pipelines
notebooks/RFRegressor.ipynb](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/notebooks) and [scripts/training_pipeline.py](https://github.com/degagawolde/pharmaceutical-sales-prediction/tree/main/scripts)
### Choose a loss function
The loss function used in this project is ***root means square error and R2***. we have to use this loss function as the problem is regression.
### Post Prediction analysis
### Serialize models
### Building model with deep learning 
### Using MLFlow to serve the prediction
# Serving predictions on a web interface
