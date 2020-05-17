#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:22:43 2020

@author: ramonpuga
"""

# Plantilla de Pre Procesado de Datos - Datos categóricos

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
# Matriz X con todas las filas, y todas las columnas menos la última
X = dataset.iloc[:, :-1].values
# Vector y con todas las filas y la última columna
y = dataset.iloc[:, -1].values

# Codificar datos de categorías
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
# La columna 0 contine valores que son categorías
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#El OneHotEncoder en las nuevas versiones está OBSOLETO
# Convertimos esos valores en columnas dummy (tantas como categorías)
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)


# NUEVO Python 3.7 (otra posibilidad de código)
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import make_column_transformer
#onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
#X = onehotencoder.fit_transform(X)

X = np.array(ct.fit_transform(X), dtype=np.float)
#X = ct.fit_transform(X)
# Eliminar una columna dummy para evitar la multicolinealidad
# OneHotEncoder pone las columnas dummy al principio, por lo tanto se podrá elimnar la columna 0
X = X[:, 1:]

# La columna de resultados, tambien es una categória (yes or no)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)