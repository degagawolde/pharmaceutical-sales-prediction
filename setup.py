from pyexpat import features, model
import streamlit as st
import pandas as pd
from math import sqrt
from numpy import concatenate
from pandas import concat
import numpy as np
import pickle 
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from scripts.dashboard_setup import (
    load_model,
    preprocess,
    plot_predictions,
    series_to_supervised,
    get_features
)

connections_path = "mysql+pymysql://root:root#123@localhost/sales_data"
engine = create_engine(connections_path)





st.image('images/ROSSMAN.jpg')
st.header("Rossmann Pharmaceuticals Sales Prediction")

input_data = st.file_uploader(label="Upload a CSV or excel file",
                              type=['csv', 'xlsx'],
                              accept_multiple_files=False)

# model = load_model(model_path='models/model.pkl')
model = pickle.load(open('models/model.pkl', 'rb'))
data = pd.DataFrame()
if input_data is not None:
    # Can be used wherever a "file-like" object is accepted:
    data = pd.read_csv(input_data)
    
else:
    data = get_features(engine,database=False,dvc=False)

columns = [i for i in range(data.shape[1])]
columns.remove(3)
print(data.head())
values = data.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

# frame as supervised learning
st.write('generateing data with sliding window...')
reframed = series_to_supervised(dataset=values, n_in = 60,n_out = 26)

# drop columns we don't want to predicti
reframed = scaler.fit_transform(reframed)
reframed = pd.DataFrame(reframed)

reframed.drop(
        reframed.columns[columns], axis=1, inplace=True)
st.dataframe(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = (2*365+365//2)*1115
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
    
st.write(f"---\n# Predictions")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
st.write('Test RMSE: %.3f' % rmse)

r2 = r2_score(inv_y, inv_yhat)
st.write('Test R2: %.3f' % r2)

preds = model.predict()
print(preds.shape)
pred_fig = plot_predictions(date=[*range(len(preds))], sales=preds)

st.write(f"---\n**Filtering by stores coming soon**")
