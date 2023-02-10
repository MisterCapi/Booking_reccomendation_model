import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from analysis import N_cities

for data_path in ('data/processed/train_set.csv', 'data/processed/test_set.csv'):
    df = pd.read_csv(data_path)
    print(df.head(20).to_string())

    # Get X_cities for RNN
    # Limit city sequences to 20
    df = df.groupby('utrip_id').head(21).reset_index(drop=True)
    X_cities = df.groupby('utrip_id').city_id.apply(lambda x: np.array(np.array(x)[:-1])).values

    def get_dimensions(array, level=0):
        yield level, len(array)
        try:
            for row in array:
                yield from get_dimensions(row, level + 1)
        except TypeError:  # not an iterable
            pass

    def get_max_shape(array):
        dimensions = defaultdict(int)
        for level, length in get_dimensions(array):
            dimensions[level] = max(dimensions[level], length)
        return [value for _, value in sorted(dimensions.items())]

    def iterate_nested_array(array, index=()):
        try:
            for idx, row in enumerate(array):
                yield from iterate_nested_array(row, (*index, idx))
        except TypeError:  # final level
            yield (*index, slice(len(array))), array

    def pad(array, fill_value):
        dimensions = get_max_shape(array)
        result = np.full(dimensions, fill_value)
        for index, value in iterate_nested_array(array):
            result[index] = value
        return result

    X_cities = pad(X_cities, N_cities)
    y_cities = df.groupby('utrip_id')['destination'].aggregate('first').values
    print(y_cities.shape)
    print(X_cities.shape)

    print(df.describe().to_string())
    continous_features = ["utrip_id", "checkin_day_of_week_sin", "checkin_day_of_week_cos", "checkin_day_sin",
                          "checkin_day_cos", "checkin_month_sin", "checkin_month_cos", "checkout_day_of_week_sin",
                          "checkout_day_of_week_cos", "checkout_day_sin", "checkout_day_cos", "checkout_month_sin",
                          "checkout_month_cos", "duration", "desktop", "mobile", "tablet", "Bartovia", "Elbonia",
                          "Gondal", "Tcherkistan", "The Devilfire Empire"]
    X_features_continuous = df[continous_features]
    X_features_continuous['total_duration'] = X_features_continuous['utrip_id'].map(
        X_features_continuous.groupby('utrip_id').aggregate('duration').aggregate('sum'))
    # print(X_features_conticous['total_duration'].quantile(0.95)) => 20
    X_features_continuous['total_duration'] = X_features_continuous['total_duration'].map(lambda x: min(x, 20))
    X_features_continuous.drop('duration', axis=1, inplace=True)
    X_features_continuous = X_features_continuous.groupby('utrip_id').aggregate('last').values

    if data_path == 'data/processed/train_set.csv':
        scaler = MinMaxScaler().fit(X_features_continuous)
        os.makedirs('scalers', exist_ok=True)
        with open('scalers/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open('scalers/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    X_features_continuous = scaler.transform(X_features_continuous)
    print(X_features_continuous.shape)

    X_features_categorical = df[["utrip_id", "affiliate_id"]].groupby('utrip_id').aggregate('last').values
    print(X_features_categorical.shape)

    os.makedirs('data/model_ready', exist_ok=True)
    if data_path == 'data/processed/train_set.csv':
        np.save('data/model_ready/X_train_cities.npy', X_cities)
        np.save('data/model_ready/y_train_cities.npy', y_cities)
        np.save('data/model_ready/X_train_features_continuous.npy', X_features_continuous)
        np.save('data/model_ready/X_train_features_categorical.npy', X_features_categorical)
    else:
        np.save('data/model_ready/X_test_cities.npy', X_cities)
        np.save('data/model_ready/y_test_cities.npy', y_cities)
        np.save('data/model_ready/X_test_features_continuous.npy', X_features_continuous)
        np.save('data/model_ready/X_test_features_categorical.npy', X_features_categorical)
