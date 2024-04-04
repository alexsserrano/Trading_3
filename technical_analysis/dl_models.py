# dl_models.py

import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump, load  # Para guardar y cargar modelos entrenados
import os


def train_dnn_model(X, y, model_path, epochs=30, batch_size=32):
    """
    Entrena un modelo Deep Neural Network con los datos proporcionados.

    Parameters:
    - X (pd.DataFrame): Características de entrada.
    - y (pd.Series): Variable objetivo.
    - model_path (str): Ruta para guardar el modelo entrenado.
    - epochs (int): Número de épocas para el entrenamiento.
    - batch_size (int): Tamaño del batch para el entrenamiento.

    Returns:
    - model: El modelo DNN entrenado.
    """
    # Definición del modelo DNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')  # Usar 'softmax' si tienes más de 2 clases
    ])

    # Compilación del modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Ajustar según sea necesario

    # Entrenamiento del modelo
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # Guardar el modelo entrenado
    model.save(model_path)
    print(f"Modelo DNN guardado en {model_path}")
    return model


def train_lstm_model(X, y, model_path, epochs=30, batch_size=32):
    """
    Entrena un modelo LSTM con los datos proporcionados.

    Parameters:
    - X (pd.DataFrame): Características de entrada en formato 3D [samples, timesteps, features].
    - y (pd.Series): Variable objetivo.
    - model_path (str): Ruta para guardar el modelo entrenado.
    - epochs (int): Número de épocas para el entrenamiento.
    - batch_size (int): Tamaño del batch para el entrenamiento.

    Returns:
    - model: El modelo LSTM entrenado.
    """
    # Asegúrate de que X está en el formato correcto para LSTM [samples, timesteps, features]
    if len(X.shape) != 3:
        raise ValueError("X debe tener 3 dimensiones [samples, timesteps, features]")

    # Definición del modelo LSTM
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compilación del modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Ajustar según sea necesario

    # Entrenamiento del modelo
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # Guardar el modelo entrenado
    model.save(model_path)
    print(f"Modelo LSTM guardado en {model_path}")
    return model


def load_and_predict(model_path, X):
    """
    Carga un modelo entrenado y realiza predicciones con un nuevo conjunto de datos.

    Parameters:
    - model_path (str): Ruta del modelo entrenado a cargar.
    - X (pd.DataFrame): Nuevo conjunto de datos para realizar predicciones.

    Returns:
    - predictions: Las predicciones del modelo.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"El modelo en {model_path} no existe.")

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X)
    return predictions


def train_cnn_model(X, y, model_path, epochs=30, batch_size=32):
    """
    Entrena un modelo Convolutional Neural Network con los datos proporcionados.

    Parameters:
    - X (pd.DataFrame): Características de entrada en formato 3D [samples, timesteps, features] para CNN 1D.
    - y (pd.Series): Variable objetivo.
    - model_path (str): Ruta para guardar el modelo entrenado.
    - epochs (int): Número de épocas para el entrenamiento.
    - batch_size (int): Tamaño del batch para el entrenamiento.

    Returns:
    - model: El modelo CNN entrenado.
    """
    if len(X.shape) != 3:
        raise ValueError("X debe tener 3 dimensiones [samples, timesteps, features] para CNN.")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    model.save(model_path)
    print(f"Modelo CNN guardado en {model_path}")
    return model

def train_conv_lstm_model(X, y, model_path, epochs=30, batch_size=32):
    """
    Entrena un modelo ConvLSTM con los datos proporcionados.

    Parameters:
    - X (pd.DataFrame): Características de entrada en formato 5D [samples, timesteps, rows, columns, features] para ConvLSTM.
    - y (pd.Series): Variable objetivo.
    - model_path (str): Ruta para guardar el modelo entrenado.
    - epochs (int): Número de épocas para el entrenamiento.
    - batch_size (int): Tamaño del batch para el entrenamiento.

    Returns:
    - model: El modelo ConvLSTM entrenado.
    """
    if len(X.shape) != 5:
        raise ValueError("X debe tener 5 dimensiones [samples, timesteps, rows, columns, features] para ConvLSTM.")

    model = tf.keras.models.Sequential([
        tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3], X.shape[4]), return_sequences=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    model.save(model_path)
    print(f"Modelo ConvLSTM guardado en {model_path}")
    return model
