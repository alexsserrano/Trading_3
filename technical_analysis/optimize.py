# optimize.py
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from joblib import dump

def optimize(data, features_columns, target_column, model_paths, n_trials=100):
    """
    Utiliza Optuna para optimizar los hiperparámetros de los modelos ML.

    Parameters:
    - data (pd.DataFrame): El DataFrame completo con los datos.
    - features_columns (list): Lista de columnas utilizadas como características.
    - target_column (str): Nombre de la columna objetivo.
    - model_paths (dict): Rutas donde se guardarán los modelos optimizados.
    - n_trials (int): Número de trials para la optimización con Optuna.

    Returns:
    - Un estudio de Optuna con los resultados de la optimización.
    """

    def objective(trial):
        # Dividir los datos en entrenamiento y prueba
        X = data[features_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Parámetros para SVC
        svc_C = trial.suggest_loguniform('svc_C', 1e-10, 1e10)
        svc_kernel = trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        svc_gamma = trial.suggest_categorical('svc_gamma', ['scale', 'auto'])

        # Parámetros para XGBoost
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 100, 1000)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
        xgb_learning_rate = trial.suggest_loguniform('xgb_learning_rate', 1e-10, 1.0)

        # Parámetros para Logistic Regression
        lr_C = trial.suggest_loguniform('lr_C', 1e-10, 1e10)

        # Entrenar SVC
        svc = SVC(C=svc_C, kernel=svc_kernel, gamma=svc_gamma)
        svc.fit(X_train, y_train)
        svc_pred = svc.predict(X_test)
        svc_accuracy = accuracy_score(y_test, svc_pred)

        # Entrenar XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)

        # Entrenar Logistic Regression
        lr = LogisticRegression(C=lr_C, solver='liblinear')
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)

        # Calcular la precisión media de los tres modelos
        average_accuracy = (svc_accuracy + xgb_accuracy + lr_accuracy) / 3

        # Guardar los modelos entrenados
        dump(svc, model_paths['svc'])
        dump(xgb_model, model_paths['xgb'])
        dump(lr, model_paths['lr'])

        return average_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study
