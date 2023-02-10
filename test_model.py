import tensorflow
import numpy as np

X_cities = np.load('data/model_ready/X_test_cities.npy')
y_cities = np.load('data/model_ready/y_test_cities.npy')
X_features_continuous = np.load('data/model_ready/X_test_features_continuous.npy')
X_features_categorical = np.load('data/model_ready/X_test_features_categorical.npy')

model: tensorflow.keras.Model = tensorflow.keras.models.load_model('models/model_v3.h5')

model.evaluate([X_cities, X_features_categorical, X_features_continuous], y_cities)

