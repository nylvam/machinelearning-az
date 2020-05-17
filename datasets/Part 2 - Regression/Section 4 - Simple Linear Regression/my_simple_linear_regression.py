#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:17:29 2020

@author: ramonpuga
"""

# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
# La columna de variable dependiente es la columna 1 o la última (-1)
y = dataset.iloc[:, -1].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# En la Regresión Lineal Simple no se necesita escalado (solo hay una variable)
# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Crear modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
# Todos los resultados de predicción forman parte de la recta
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
# Pintar una nube de puntos (scatter)
plt.scatter(X_train, y_train, color = "red")
# Pintar una recta de regresión (los puntos de la recta necesitan cada uno dos coordenadas)
# Eje x = valores de los años de experiencia X_train
# Eje y = valores de sueldo que el modelo de regresión predice para esos valores
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultados de test
# Pintar una nube de puntos (scatter)
plt.scatter(X_test, y_test, color = "red")
# Pintar una recta de regresión (los puntos de la recta necesitan cada uno dos coordenadas)
# OJO la recta es la misma, se puede dejar la misma instruccion de antes y pintarla con los datos de entrenamiento
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()