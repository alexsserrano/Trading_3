# optimize2.py
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from technical_analysis.dl_models import train_dnn_model, train_lstm_model, train_cnn_model, train_conv_lstm_model, load_and_predict
from technical_analysis.backtest import backtest
import os


def optimize_strategy(data, features_columns, target_column, model_paths, n_trials=10):
    """
    Utiliza Optuna para optimizar los hiperparámetros de modelos de deep learning y los parámetros del backtest.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene los datos de mercado.
    - features_columns (list): Lista de columnas a usar como características.
    - target_column (str): Nombre de la columna objetivo.
    - model_paths (str): Ruta base para guardar los modelos optimizados.
    - n_trials (int): Número de ensayos para la optimización con Optuna.
    """

    # Preprocesamiento: Normaliza las características
    X = data[features_columns]
    y = data[target_column]

    # Dividir los datos
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        # Elegir el tipo de modelo
        model_type = trial.suggest_categorical('model_type', ['DNN', 'LSTM', 'CNN', 'ConvLSTM'])

        # Definir hiperparámetros específicos del modelo
        epochs = trial.suggest_int('epochs', 5, 30)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        # Definir parámetros específicos de cada modelo
        if model_type == 'DNN':
            model = train_dnn_model(X_train, y_train, f"{model_paths}_{trial.number}.h5", epochs=epochs,
                                    batch_size=batch_size)
        elif model_type == 'LSTM':
            # Asegúrate de que los datos estén en el formato correcto para LSTM
            X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            model = train_lstm_model(X_train_lstm, y_train, f"{model_paths}_{trial.number}.h5", epochs=epochs,
                                     batch_size=batch_size)
        elif model_type == 'CNN':
            # Asegúrate de que los datos estén en el formato correcto para CNN
            X_train_cnn = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            model = train_cnn_model(X_train_cnn, y_train, f"{model_paths}_{trial.number}.h5", epochs=epochs,
                                    batch_size=batch_size)
        elif model_type == 'ConvLSTM':
            # Asegúrate de que los datos estén en el formato correcto para ConvLSTM
            # Ejemplo: ConvLSTM necesita datos en formato 5D [samples, timesteps, rows, columns, features]
            raise NotImplementedError("La preparación de datos para ConvLSTM debe ser implementada")

        # Realizar predicciones con el modelo entrenado y calcular la precisión en el conjunto de validación
        # La implementación específica dependerá del modelo y de tus datos
        # Aquí va tu código para evaluar el modelo y calcular la precisión o cualquier métrica que desees usar
        accuracy = np.random.rand()  # Esto es un placeholder

        # Opcional: Puedes eliminar el modelo después de evaluarlo para ahorrar espacio
        os.remove(f"{model_paths}_{trial.number}.h5")

        return accuracy  # La métrica a optimizar

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study
