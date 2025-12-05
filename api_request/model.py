import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Conv1D, Bidirectional,
                                     TimeDistributed, Concatenate, Dot, Activation,
                                     BatchNormalization, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def directional_loss(y_true, y_pred):
    sign_true = K.sign(y_true)
    sign_pred = K.sign(y_pred)

    direction_correct = sign_true * sign_pred >= 0
    penalty = tf.where(direction_correct, 1.0, 1.5)

    squared_error = K.square(y_true - y_pred)
    penalized_error = squared_error * penalty

    return K.mean(penalized_error)


def attention_block(hidden_states, context_vector):
    score = Dot(axes=[1, 2])([hidden_states, context_vector])
    score = Activation('softmax')(score)
    context = Dot(axes=[1, 1])([hidden_states, score])
    return context, score


def build_hybrid_model(input_timesteps, input_features, output_horizon):
    encoder_inputs = Input(shape=(input_timesteps, input_features), name='encoder_input')

    cnn = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(encoder_inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.2)(cnn)
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    encoder_lstm = Bidirectional(LSTM(128, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(cnn)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(output_horizon, 1), name='decoder_input')

    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    attention = tf.keras.layers.Attention()([decoder_outputs, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention])

    dense = TimeDistributed(Dense(64, activation='relu'))(decoder_combined_context)
    dense = Dropout(0.2)(dense)
    outputs = TimeDistributed(Dense(1, activation='linear'))(dense)

    model = Model([encoder_inputs, decoder_inputs], outputs)

    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)  # Giảm learning rate
    model.compile(optimizer=optimizer, loss=directional_loss, metrics=['mae', 'mse'])

    return model

def train_model(model, X_encoder_train, X_decoder_train, y_train,
                X_encoder_test, X_decoder_test, y_test,
                model_save_path='best_model.keras',
                epochs=100, batch_size=32):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    history = model.fit(
        [X_encoder_train, X_decoder_train], y_train,
        validation_data=([X_encoder_test, X_decoder_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history

def evaluate_model(model, X_encoder_test, X_decoder_test, y_test, scaler_y):

    predictions = model.predict([X_encoder_test, X_decoder_test])

    if predictions.ndim == 3:
        predictions = predictions.squeeze(-1)

    if y_test.ndim == 3:
        y_test = y_test.squeeze(-1)

    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)

    mae = np.mean(np.abs(y_test_original - predictions_original))
    rmse = np.sqrt(np.mean((y_test_original - predictions_original) ** 2))
    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # plot_predictions_timeseries(y_test_original, predictions_original)

    # plot_predictions_samples(y_test_original, predictions_original, num_samples=5)

    return predictions_original

# def predict_next_days()