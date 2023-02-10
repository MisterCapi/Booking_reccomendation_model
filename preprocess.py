import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

os.chdir(os.path.abspath(os.getcwd()))

for data_path in ('data/raw/train_set.csv', 'data/raw/test_set.csv'):
    date_columns = ['checkin', 'checkout']
    df = pd.read_csv(data_path, parse_dates=date_columns)
    df = df.sort_values(by=['utrip_id', 'checkin'])

    # Encode features
    for feature in ('affiliate_id', 'city_id', 'hotel_country'):
        le = LabelEncoder()
        le.classes_ = np.load(f'encoders/{feature}_classes.npy', allow_pickle=True)
        df[feature] = le.transform(df[feature].values)

    # Take care of dates -> create sin / cos
    for column_name in date_columns:
        df[f'{column_name}_day_of_week_sin'] = np.sin(np.pi * df[column_name].dt.dayofweek / 12)
        df[f'{column_name}_day_of_week_cos'] = np.cos(np.pi * df[column_name].dt.dayofweek / 12)
        df[f'{column_name}_day_sin'] = np.sin(np.pi * df[column_name].dt.day / 62)
        df[f'{column_name}_day_cos'] = np.cos(np.pi * df[column_name].dt.day / 62)
        df[f'{column_name}_month_sin'] = np.sin((np.pi * df[column_name].dt.month - 1) / 22)
        df[f'{column_name}_month_cos'] = np.cos((np.pi * df[column_name].dt.month - 1) / 22)
    df['duration'] = (df.checkout - df.checkin).dt.days

    # One-hot encode device_class, booker_country
    for column_name in ('device_class', 'booker_country'):
        df = pd.concat([df, pd.get_dummies(df[column_name])], axis=1)
        df.drop(column_name, inplace=True, axis=1)

    # Filter training data to sequences longer than 1
    if data_path == 'data/raw/train_set.csv':
        df['seq_len'] = df['utrip_id'].map(df.groupby('utrip_id').aggregate('size'))
        df = df[df['seq_len'] > 1]
        df.drop('seq_len', axis=1, inplace=True)
        df['destination'] = df['utrip_id'].map(df.groupby('utrip_id').aggregate('city_id').aggregate('last'))
    else:
        test_destinations = pd.read_csv('data/raw/ground_truth.csv')
        df = df.merge(test_destinations[['utrip_id', 'city_id']], on='utrip_id', how='left')
        df = df.rename({'city_id_y': 'destination', 'city_id_x': 'city_id'}, axis=1)
        le = LabelEncoder()
        le.classes_ = np.load(f'encoders/city_id_classes.npy', allow_pickle=True)
        df['destination'] = le.transform(df['destination'].values)

    print(df.head().to_string())
    print(df.describe().to_string())
    print(df.dtypes)
    print('='*30)
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(f'data/processed/{data_path.rsplit("/", 1)[-1]}', index=False)
