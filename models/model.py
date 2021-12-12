#!/usr/bin/env python3

import pandas as pd
import numpy as np

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
    return df

def tratar_valores_faltantes(df):
    df = df.copy()

    # dropando colunas 'intubed', 'icu', 'pregnancy'
    df = df.drop(columns = ['intubed', 'icu', 'pregnancy'])
    
    # dropando missings das outras colunas
    df = df.dropna(subset = set(df.columns) - set(['contact_other_covid']))

    # agora que não há mais dados faltantes na coluna de 'age', podemos reverter para o tipo 'int8' do numpy
    # (ao invés do tipo 'Int8' do pandas, que aceita NaN)
    df['age'] = df['age'].astype('int8')

    # codificando NaN da coluna 'contact_other_covid' como -1
    # for c in ['contact_other_covid',]:
    #     df[c] = df[c].cat.codes
    #     df[c] = df[c].astype('category')

    return df


# import
covid_raw = pd.read_csv(r'../data/COVID.csv')

# preproc

covid = (covid_raw
    .pipe(acertar_tipos)
    .pipe(remover_outliers)
    .pipe(remover_duplicatas)
    .pipe(tratar_valores_faltantes)
)