import pandas as pd
from sklearn.preprocessing import LabelEncoder

def nonResponse(df):
    # excluding who did not respond
    clean_data = df[df['yes'] == 1].append(df[df['no'] == 1])
    return clean_data

def remove(df, torem: list):
    df = df.drop(torem, axis=1)
    return df

def hot_encode(df, column):
    coldum = pd.get_dummies(df[column], drop_first = True)
    df = pd.concat([df, coldum], axis = 1)
    df.rename(columns={list(df.columns)[-1]: f"{column} {list(df.columns)[-1]}"}, inplace=True)

    return df

def encode(df, col):
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    inv = encoder.inverse_transform(df[col])
    # dic = pd.DataFrame(data= {'Label': list(inv), 'Code': list(df[col])})
    return df

  