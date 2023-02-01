import numpy as np
from tensorflow import keras
from analysis import N_cities, N_affiliate


X_cities = np.load('data/model_ready/X_train_cities.npy')
y_cities = np.load('data/model_ready/y_train_cities.npy')
X_features_continuous = np.load('data/model_ready/X_train_features_continuous.npy')
X_features_categorical = np.load('data/model_ready/X_train_features_categorical.npy')

print(X_cities.shape)
print(y_cities.shape)
print(X_features_continuous.shape)
print(X_features_categorical.shape)

# RNN
input_city_seq = keras.layers.Input(shape=(X_cities.shape[1]))
city_embedding = keras.layers.Embedding(N_cities+1, 24, trainable=True)(input_city_seq)

lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(city_embedding)
gru = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=False))(lstm)

# MLP
input_categorical = keras.layers.Input(shape=(X_features_categorical.shape[1]))
embedding_categorical = keras.layers.Embedding(N_affiliate, 6, trainable=True)(input_categorical)
dense_categorical = keras.layers.Dense(512)(embedding_categorical)
flatten_categorical = keras.layers.Flatten()(dense_categorical)

input_continuous = keras.layers.Input(shape=(X_features_continuous.shape[1]))

# Final MLP
all_together = keras.layers.concatenate([gru, flatten_categorical, input_continuous], axis=1)
mlp = keras.layers.Dense(512, activation='relu')(all_together)
mlp = keras.layers.Dense(256, activation='relu')(mlp)
mlp_out = keras.layers.Dense(N_cities, activation='softmax')(mlp)

model = keras.Model([input_city_seq, input_categorical, input_continuous], mlp_out)

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=4)])
model.fit([X_cities, X_features_categorical, X_features_continuous],
          y_cities,
          epochs=1000,
          batch_size=8042,
          validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_sparse_top_k_categorical_accuracy',
                                                   patience=3,
                                                   mode='max')])

model.save('models/model.h5')
