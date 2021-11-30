import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.metrics import mean_absolute_percentage_error as fmape, mean_squared_error as fmse, mean_absolute_error as fmae, r2_score as fr2, f1_score as ff1, precision_score as fprecision, recall_score as frecall, accuracy_score as faccuracy

def tunar_hiperparams(
    tipo_modelo, 
    selec_hiperparams: Tuple[pd.Index],
    X_train: np.array or pd.DataFrame or pd.Series,
    X_test: np.array or pd.DataFrame or pd.Series,
    y_train: np.array or pd.Series,
    y_test: np.array or pd.Series,
    tipo_dado: str = 'quantitativo',
    scaler = None,
    *func_metricas_args,
    **func_metricas_kwargs
) -> pd.DataFrame:

    if tipo_dado not in [ 'quantitativo', 'qualitativo']:
        raise NameError("'tipo_dado' deve ser 'qualitativo' ou 'quantitativo'.")

    if scaler is not None:
        sc = scaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    # create test cases

    comb_idx = pd.MultiIndex.from_product(selec_hiperparams)

    # create apply function
    def calc_scores(s, nomes_hiperparams):
        modelo_hiperparams = { nome: valor for nome, valor in zip(nomes_hiperparams, s.name)}
        
        modelo = tipo_modelo(**modelo_hiperparams)
        modelo.fit(X_train, y_train)
        s['modelos'] = modelo
        
        y_pred = modelo.predict(X_test)

        metricas_kwargs = dict(
            y_true = y_test,
            y_pred = y_pred
        )

        metricas_kwargs.update(func_metricas_kwargs)

        if tipo_dado == 'quantitativo':

            r2 = fr2(*func_metricas_args, **metricas_kwargs)
            mape = fmape(*func_metricas_args, **metricas_kwargs)
            rmse = fmse(squared = False, *func_metricas_args, **metricas_kwargs)
            mae = fmae(*func_metricas_args, **metricas_kwargs)
            
            s['r2'] = r2
            s['mape'] = mape
            s['rmse'] = rmse
            s['mae'] = mae
        
        else:
            f1 = ff1(*func_metricas_args, **metricas_kwargs)
            recall = frecall(*func_metricas_args, **metricas_kwargs)
            precision = fprecision(*func_metricas_args, **metricas_kwargs)

            s['f1'] = f1
            s['precision'] = precision
            s['recall'] = recall

        return s
        
    # create dataframe
    if tipo_dado == 'quantitativo':
        score_cols = [ 'modelos', 'mape', 'rmse', 'mae', 'r2']
        ascending = False
    else:
        score_cols = [ 'modelos', 'f1', 'recall', 'precision']
        ascending = False

    scores = pd.DataFrame([], columns = score_cols, index = comb_idx)
    comb_nomes = scores.index.names
    scores = scores.apply(calc_scores, nomes_hiperparams = comb_nomes, axis = 1)
    scores = scores.sort_values(by = list(scores.columns[1:]), ascending = ascending)
    return scores