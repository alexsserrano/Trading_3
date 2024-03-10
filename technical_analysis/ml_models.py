#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, accuracy_score
#import pandas as pd
#import xgboost as xgb
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#
#
#def prepare_data_and_train_models(data, features_columns, target_column):
#    """
#    Entrena los modelos SVC, XGBoost, y Logistic Regression con el dataset proporcionado
#    y devuelve las predicciones de cada modelo.
#
#    Parameters:
#    - data (pd.DataFrame): El DataFrame con los datos.
#    - features_columns (list of str): Lista de columnas usadas como características.
#    - target_column (str): La columna objetivo.
#
#    Returns:
#    Tuple de Series con predicciones de SVC, XGBoost, y Logistic Regression.
#    """
#    # Dividir el dataset en características y objetivo
#    X = data[features_columns]
#    y = data[target_column]
#
#    # Dividir en entrenamiento y prueba
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#    # Entrenar SVC
#    svc = SVC(C=1.0, kernel='rbf', gamma='scale')
#    svc.fit(X_train, y_train)
#    svc_predictions = svc.predict(X_test)
#
#    # Entrenar XGBoost
#    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False,
#                                  eval_metric='logloss')
#    xgb_model.fit(X_train, y_train)
#    xgb_predictions = xgb_model.predict(X_test)
#
#    # Entrenar Logistic Regression
#    lr = LogisticRegression(C=1.0, solver='liblinear')
#    lr.fit(X_train, y_train)
#    lr_predictions = lr.predict(X_test)
#
#    # Convertir a Series para facilitar operaciones posteriores
#    svc_predictions_series = pd.Series(svc_predictions, index=X_test.index)
#    xgb_predictions_series = pd.Series(xgb_predictions, index=X_test.index)
#    lr_predictions_series = pd.Series(lr_predictions, index=X_test.index)
#
#    return svc_predictions_series, xgb_predictions_series, lr_predictions_series

# ml_models.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import dump, load  # Para guardar y cargar modelos

# Esta función ya no será necesaria en su forma actual
# def prepare_data_and_train_models(...)

def train_and_save_models(data, features_columns, target_column, model_paths):
    """
    Entrena los modelos SVC, XGBoost, y Logistic Regression con el dataset proporcionado
    y guarda los modelos entrenados en las rutas especificadas.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los datos.
    - features_columns (list of str): Lista de columnas usadas como características.
    - target_column (str): La columna objetivo.
    - model_paths (dict): Diccionario con las rutas para guardar cada modelo entrenado.
    """
    X = data[features_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar y guardar SVC
    svc = SVC(C=1.0, kernel='rbf', gamma='scale')
    svc.fit(X_train, y_train)
    dump(svc, model_paths['svc'])

    # Entrenar y guardar XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    dump(xgb_model, model_paths['xgb'])

    # Entrenar y guardar Logistic Regression
    lr = LogisticRegression(C=1.0, solver='liblinear')
    lr.fit(X_train, y_train)
    dump(lr, model_paths['lr'])

def load_models_and_predict(data, features_columns, model_paths):
    """
    Carga los modelos entrenados y hace predicciones con un nuevo conjunto de datos.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los nuevos datos para predicción.
    - features_columns (list of str): Lista de columnas usadas como características.
    - model_paths (dict): Diccionario con las rutas de los modelos entrenados a cargar.

    Returns:
    Tuple de pd.Series con predicciones de SVC, XGBoost, y Logistic Regression.
    """
    X = data[features_columns]

    # Cargar y predecir con SVC
    svc = load(model_paths['svc'])
    svc_predictions = svc.predict(X)

    # Cargar y predecir con XGBoost
    xgb_model = load(model_paths['xgb'])
    xgb_predictions = xgb_model.predict(X)

    # Cargar y predecir con Logistic Regression
    lr = load(model_paths['lr'])
    lr_predictions = lr.predict(X)

    return pd.Series(svc_predictions, index=data.index), pd.Series(xgb_predictions, index=data.index), pd.Series(lr_predictions, index=data.index)
