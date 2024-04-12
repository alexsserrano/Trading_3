# generate_buy_signals2.py
import pandas as pd
from technical_analysis.dl_models import load_and_predict


def generate_buy_signals2(data, features_columns, model_paths):
    """
    Genera señales de compra basadas en las predicciones de los modelos de deep learning.
    Una señal de compra se genera si la predicción del modelo supera un umbral determinado.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los nuevos datos para generar las señales.
    - features_columns (list of str): Lista de columnas usadas como características.
    - model_paths (list of str): Lista con las rutas de los modelos entrenados a cargar.

    Returns:
    - pd.DataFrame: DataFrame con señales de compra para cada modelo, donde 1 indica una señal de compra y 0 indica no comprar.
    """
    # Preparar los datos de entrada para la predicción
    X = data[features_columns]

    # Inicializar el DataFrame de señales de compra
    buy_signals = pd.DataFrame(index=data.index)

    # Generar señales de compra para cada modelo
    for idx, model_path in enumerate(model_paths):
        # Cargar el modelo y realizar predicciones
        predictions = load_and_predict(model_path, X)

        # Convertir las predicciones en señales de compra (suponiendo clasificación binaria)
        # Aquí, por ejemplo, consideramos una señal de compra si la predicción es mayor a 0.5
        buy_signals[f'model_{idx}'] = (predictions > 0.5).astype(int)

    return buy_signals


# Ejemplo de uso
if __name__ == "__main__":
    # Suponiendo que 'data' es tu DataFrame que contiene los datos de mercado y 'features_columns' las columnas que quieres utilizar como características
    data = pd.read_csv("tu_archivo_de_datos.csv")
    features_columns = ['caracteristica1', 'caracteristica2', 'caracteristica3']
    model_paths = ['ruta_al_modelo_dnn.h5', 'ruta_al_modelo_lstm.h5', 'ruta_al_modelo_cnn.h5',
                   'ruta_al_modelo_convlstm.h5']

    buy_signals = generate_buy_signals2(data, features_columns, model_paths)
    print(buy_signals.head())
