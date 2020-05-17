#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:22:44 2020

@author: ramonpuga
"""

# Plantilla de Pre Procesado de Datos - Datos faltantes

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

# Tratamiento de los NAs (datos vacíos)
from sklearn.impute import SimpleImputer
# Aplicamos estrategia de la media de la columna para rellenar valores vacíos (nan)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0) 
# Ajustamos los valores de la columna 1 a la 2 (3 no incluida)
imputer = imputer.fit(X[:, 1:3])
# Copiamos los valores a la matriz X
X[:, 1:3] = imputer.transform(X[:, 1:3])