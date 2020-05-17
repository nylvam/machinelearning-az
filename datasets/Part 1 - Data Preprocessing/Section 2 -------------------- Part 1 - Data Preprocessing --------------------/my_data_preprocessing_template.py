#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:15:34 2020

@author: ramonpuga
"""

# Plantilla de Pre Procesado de Datos

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

# Dividir el data set en conjunto de training y de test
from sklearn.model_selection import train_test_split
# Aplicamos un porcentaje del 20% (0.2) para el test y un valor de selección alatoria de 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado (estandarización o normalización) de variables
""" Quitar comillas iniciales y fianles cuando se necesite este código
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Aplicamos y fijamos el metodo de estandarización a todas las columnas X
X_train = sc_X.fit_transform(X_train)
# Aplicamos el mismo metodo de estandarización que para los datos de training
X_test = sc_X.transform(X_test)
"""