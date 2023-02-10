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


def create_model():
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
    return model


# 20k fold validation
k = 20
fold_size = X_cities.shape[0] // k
validation_accuracies = []
validation_losses = []

for i in range(k):
    # Get the start and end index for the current validation fold
    print(f"Training {i+1}/20 fold")
    start = i * fold_size
    end = (i + 1) * fold_size

    # Get the current validation fold
    X_cities_valid = X_cities[start:end]
    X_features_categorical_valid = X_features_categorical[start:end]
    X_features_continuous_valid = X_features_continuous[start:end]
    y_cities_valid = y_cities[start:end]

    # Get the training data by concatenating all the other folds
    X_cities_train = np.concatenate([X_cities[:start], X_cities[end:]])
    X_features_categorical_train = np.concatenate([X_features_categorical[:start], X_features_categorical[end:]])
    X_features_continuous_train = np.concatenate([X_features_continuous[:start], X_features_continuous[end:]])
    y_cities_train = np.concatenate([y_cities[:start], y_cities[end:]])

    model = create_model()
    history = model.fit([X_cities_train, X_features_categorical_train, X_features_continuous_train],
                        y_cities_train,
                        epochs=1000,
                        batch_size=2008,
                        validation_data=([X_cities_valid, X_features_categorical_valid, X_features_continuous_valid], y_cities_valid),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_sparse_top_k_categorical_accuracy',
                                                                 patience=3,
                                                                 mode='max')])
    validation_accuracies.append(history.history['val_sparse_top_k_categorical_accuracy'][-1])
    validation_losses.append(history.history['val_loss'][-1])

    model.save('models/model.h5')

mean_validation_accuracy = np.mean(validation_accuracies)
print(f'Mean Validation Accuracy: {mean_validation_accuracy}')
std_validation_accuracy = np.std(validation_accuracies)
print(f'Standard deviation of Validation Accuracy: {std_validation_accuracy}')

mean_validation_loss = np.mean(validation_losses)
print(f'Mean Validation Accuracy: {mean_validation_loss}')
std_validation_loss = np.std(validation_losses)
print(f'Standard deviation of Validation Accuracy: {std_validation_loss}')
