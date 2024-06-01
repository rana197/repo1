
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import csv

# Importing the dataset - preprocessed already

df = pd.read_csv('C:Downloads/10_normalized_file.csv',sep=',', index_col=False)

## Set the proportion of the dataset to use for training (e.g., 80% for training, 20% for testing)
train_proportion = 0.95     #.98
# Calculate the split index based on the proportion
split_index = int(len(df) * train_proportion)
df_train = df[:split_index]
df_test = df[split_index:]

spots_train = df_train[['timestep_time', 'Del_V','vehicle_angle', 'vehicle_x','vehicle_y']].values
spots_test = df_test[['timestep_time', 'Del_V','vehicle_angle', 'vehicle_x','vehicle_y']].values

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window[:, :])  # Input columns: 'timestep_time', 'vehicle_angle', 'vehicle_speed', 'Del_V', 'vehicle_x','vehicle_y'
        y.append(after_window[-3:])  # Output columns: 'vehicle_x','vehicle_y', angle

    return np.array(x), np.array(y)

SEQUENCE_SIZE = 10
x_train, y_train = to_sequences(SEQUENCE_SIZE, spots_train)
x_test, y_test = to_sequences(SEQUENCE_SIZE, spots_test)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    #outputs = layers.Dense(2)(x)  # 2 output columns: 'x cordinate' and 'y cordinate'
    outputs = layers.Dense(3)(x)  # 3 output columns: 'x cordinate',y cordinate and 'angle'
    return keras.Model(inputs, outputs)


input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=64,      # , 256
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=2,   #4
    mlp_units=[64],        #128
    #mlp_dropout=0.4,
    #dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-5)
    #optimizer = keras.optimizers.SGD(learning_rate=1e-5)
)

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=1,
    batch_size=10,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

pred = model.predict(x_test)

score = np.sqrt(metrics.mean_squared_error(pred[:, 2], y_test[:, 2]))  # RMSE for 'y cordinate'
print("Score (RMSE)-norm for 'y cordinate': {}".format(score))

score = np.sqrt(metrics.mean_squared_error(pred[:, 0], y_test[:, 0]))  # RMSE for 'angle'
print("Score (RMSE)-norm for 'angle': {}".format(score))

score = np.sqrt(metrics.mean_squared_error(pred[:, 1], y_test[:, 1]))  # RMSE for 'x cordinate'
print("Score (RMSE)-norm for 'x cordinate': {}".format(score))

score = np.sqrt(metrics.mean_squared_error(pred, y_test))
print("Combined RMSE: {}".format(score))


with open('C:Downloads/pred_norm.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pred)

with open('C:Downloads/y_test_norm.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(y_test)
