import os

os.chdir(os.path.abspath(os.getcwd()))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/raw/train_set.csv')
df2 = pd.read_csv('data/raw/test_set.csv')

df = pd.concat([df, df2], axis=0)

N_cities = df.nunique()['city_id']
N_affiliate = df.nunique()['affiliate_id']

"""
user_id           259159 -> delete
checkin              425 -> to cos and sin of dt.dayofweek and dt.day and dt.month
checkout             425 -> to cos and sin of dt.dayofweek and dt.day and dt.month
city_id            39902 -> label encode, because not all int values are used (many are skipped in fact)
device_class           3 -> ['desktop' 'mobile' 'tablet'] -> easy to one-hot encode
affiliate_id        3611 -> label encode
booker_country         5 -> ['desktop' 'mobile' 'tablet'] -> easy to one-hot encode
hotel_country        195 -> drop
utrip_id          288348 -> just an identification id for each sample -> skip
"""
if __name__ == '__main__':
    print(df.head(10).to_string())
    print(df.dtypes)
    print(df.describe())
    print(df.nunique())
    print(df.device_class.unique())
    print(N_cities)

    os.makedirs('encoders', exist_ok=True)
    le = LabelEncoder().fit(df.city_id.values)
    np.save('encoders/city_id_classes.npy', le.classes_)
    le = LabelEncoder().fit(df.hotel_country.values)
    np.save('encoders/hotel_country_classes.npy', le.classes_)
    le = LabelEncoder().fit(df.affiliate_id.values)
    np.save('encoders/affiliate_id_classes.npy', le.classes_)
