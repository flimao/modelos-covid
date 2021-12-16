#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# default
import pickle
import pandas as pd
import numpy as np
import os

# modelo
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# tooling
from sklearn.model_selection import train_test_split


# preproc pipeline
def acertar_tipos(df):
    df = df.copy()

    df = df.astype(pd.Int8Dtype())  # porque não 'int64' normal do numpy? Porque o 'Int' do pandas acomoda a presença de NaNs (o do numpy não); o '8' é só para economizar memória (e tornar a análise mais ágil)
    cols_minus_age = list(set(df.columns) - set(['age']))
    df[cols_minus_age] = df[cols_minus_age].astype('category')

    return df

def remover_outliers(df):
    df = df.copy()

    outliers_mask = (
        ((df.age < 12) & (df.pregnancy == 1)) | # crianca gravida
        ((df.sex == 0) & (df.pregnancy == 1)) | # homem gravido
        ((df.patient_type == 1) & (df.intubed == 1))  # paciente dispensado e intubado
    )

    df = df.drop(index = df.index[outliers_mask])

    return df

def remover_duplicatas(df):
    # não vamos remover duplicatas, mas mantemos a função no pipe caso mudemos de ideia
    return df

def tratar_valores_faltantes(df):
    df = df.copy()

    # dropando colunas 'intubed', 'icu'
    df = df.drop(columns = ['intubed', 'icu'])

    # preenchendo pregnancy = 0 para homens (sex == 0)
    df.loc[df.sex == 0, 'pregnancy'] = 0
    
    # dropando missings das outras colunas
    df = df.dropna(subset = set(df.columns) - set(['contact_other_covid']))

    # agora que não há mais dados faltantes na coluna de 'age', podemos reverter para o tipo 'int8' do numpy
    # (ao invés do tipo 'Int8' do pandas, que aceita NaN)
    df['age'] = df['age'].astype('int8')

    # codificando NaN da coluna 'contact_other_covid' como -1    
    for c in ['contact_other_covid',]:
        df[c] = df[c].cat.codes
        df[c] = df[c].astype('category')

    return df


# classe para construir Pipelines com predict em função do threshold
class Pipeline_threshold(Pipeline):
    def __init__(self, threshold = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.threshold = threshold
    
    def predict(self, threshold = None, *args, **kwargs):
        yprobs = self.predict_proba(*args, **kwargs)
        
        t = threshold
        if t is None:
            t = self.threshold

        return yprobs[:, 1] > t


def montar_modelo(estimador, base, threshold = 0.5, opt_params = None, fit = True):

    X = covid.drop(columns = 'covid_res')
    y = covid['covid_res']

    # pipeline das colunas numéricas
    pipeline_numericas = Pipeline(steps = [
        ('scaler_std', StandardScaler()),
    ])

    # pipeline das colunas categoricas
    pipeline_categoricas = Pipeline( steps = [
        ('onehot', OneHotEncoder(handle_unknown = 'ignore', drop = 'first')),
    ])

    # nome das colunas numericas e categoricas
    features_numericas = X.select_dtypes(exclude = ['object', 'category']).columns
    features_categoricas = X.select_dtypes(include = ['object', 'category']).columns

    # separador: os dados entram no pipeline e são separados em dois, cada um vai em um pipeline diferente
    separador = ColumnTransformer(transformers = [
        ('numericas', pipeline_numericas, features_numericas),
        ('categoricas', pipeline_categoricas, features_categoricas)
    ])

    # ... montar os passos do pipeline final de cada modelo ...
    steps = [ 
        ('separador', separador),
        estimador
    ]

    # ... e criar a pipeline
    model = Pipeline_threshold(
        steps = steps,
        threshold = threshold
    )

    # setar os parametros otimizados
    if opt_params is not None:
        model.set_params(**opt_params)

    # fit se a opção for True
    if fit:
        model.fit(
            X,
            y.astype('int8')
        )
    
    return model

def preproc_pipe(covid_raw):
    # preproc
    covid = (covid_raw
        .pipe(acertar_tipos)
        .pipe(remover_outliers)
        .pipe(remover_duplicatas)
        .pipe(tratar_valores_faltantes)
    )
    
    return covid
#%%
if __name__ == '__main__':

    # arquivos
    MODELSCRIPTFILE = os.path.realpath(__file__)
    MODELDIR = os.path.dirname(MODELSCRIPTFILE)
    DBFILE = os.path.join(MODELDIR, r'../data/COVID.csv')
    MODELBINFILE = os.path.join(MODELDIR, r'modelo.model')

    # modelo final
    estimador_final = ('xgboost', XGBClassifier())
    opt_params = {
        'xgboost__base_score': 0.5,
        'xgboost__booster': 'gbtree',
        'xgboost__colsample_bylevel': 1,
        'xgboost__colsample_bynode': 1,
        'xgboost__colsample_bytree': 1,
        'xgboost__criterion': 'mae',
        'xgboost__enable_categorical': False,
        'xgboost__eval_metric': 'logloss',
        'xgboost__gamma': 0,
        'xgboost__gpu_id': -1,
        'xgboost__importance_type': None,
        'xgboost__interaction_constraints': '',
        'xgboost__learning_rate': 0.1,
        'xgboost__max_delta_step': 0,
        'xgboost__max_depth': 3,
        'xgboost__max_features': 'log2',
        'xgboost__min_child_weight': 1,
        'xgboost__missing': np.nan,
        'xgboost__monotone_constraints': '()',
        'xgboost__n_estimators': 100,
        'xgboost__n_jobs': -1,
        'xgboost__num_parallel_tree': 1,
        'xgboost__objective': 'binary:logistic',
        'xgboost__predictor': 'auto',
        'xgboost__reg_alpha': 0,
        'xgboost__reg_lambda': 1,
        'xgboost__scale_pos_weight': 1,
        'xgboost__subsample': 1,
        'xgboost__tree_method': 'exact',
        'xgboost__use_label_encoder': True,
        'xgboost__validate_parameters': 1,
        'xgboost__verbosity': None
    }
    threshold = 0.24

    # import
    covid_raw = pd.read_csv(DBFILE, index_col = 'Unnamed: 0')
    covid_raw.index.name = 'id'

    # preproc
    covid = preproc_pipe(covid_raw)

    X = covid.drop(columns = 'covid_res')
    y = covid['covid_res']

    # model
    model = montar_modelo(
        base = covid,
        estimador = estimador_final,
        opt_params = opt_params,
        threshold = threshold,
        fit = True
    )

    # export para arquivo
    model_export = {}

    # modelo: Pipeline já fitado
    model_export['modelo'] = model

    # base: df covid já transformado
    model_export['base_treino'] = covid

    # salvar via pickle
    with open(MODELBINFILE, 'wb') as modelfile:
        pickler = pickle.Pickler(file = modelfile)
        pickler.dump(model_export)
        