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
from sklearn.linear_model import LogisticRegression
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

def montar_modelo(train_test_split_tuple, estimador, opt_params = None, fit = True):

    X_train, X_test, y_train, y_test = train_test_split_tuple

    # pipeline das colunas numéricas
    pipeline_numericas = Pipeline(steps = [
        ('scaler_std', StandardScaler()),
    ])

    # pipeline das colunas categoricas
    pipeline_categoricas = Pipeline( steps = [
        ('onehot', OneHotEncoder(handle_unknown = 'ignore', drop = 'first')),
    ])

    # nome das colunas numericas e categoricas
    features_numericas = X_train.select_dtypes(exclude = ['object', 'category']).columns
    features_categoricas = X_train.select_dtypes(include = ['object', 'category']).columns

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
    model = Pipeline(
        steps = steps,
    )

    # setar os parametros otimizados
    if opt_params is not None:
        model.set_params(**opt_params)

    # fit se a opção for True
    if fit:
        model.fit(
            X_train,
            y_train.astype('int8')
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

if __name__ == '__main__':

    MODELSCRIPTFILE = os.path.realpath(__file__)
    MODELDIR = os.path.dirname(MODELSCRIPTFILE)
    DBFILE = os.path.join(MODELDIR, r'../data/COVID.csv')
    MODELBINFILE = os.path.join(MODELDIR, r'modelo.model')

    # import
    covid_raw = pd.read_csv(DBFILE, index_col = 'Unnamed: 0')
    covid_raw.index.name = 'id'

    # preproc
    covid = preproc_pipe(covid_raw)

    X = covid.drop(columns = 'covid_res')
    y = covid['covid_res']

    train_test_split_tuple = train_test_split(
        X, y,
        test_size = 0.3,
        stratify = y
    )

    # model
    model = montar_modelo(
        train_test_split_tuple,
        estimador = ('teste', LogisticRegression()), # TODO: modelo
        opt_params = None, # TODO: opt_params
        fit = True
    )

    # export para arquivo
    model_export = {}

    # modelo: Pipeline já fitado
    model_export['modelo'] = model

    # train_test_split: tupla com 2 dataframes (X_train, X_test) e 2 series (y_train e y_test)
    model_export['train_test_split'] = train_test_split_tuple

    # base: df covid já transformado
    model_export['base'] = covid

    # salvar via pickle
    # with open(MODELBINFILE, 'wb') as modelfile:
    #     pickler = pickle.Pickler(file = modelfile)
    #     pickler.dump(model_export)
        