#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:41:22 2020

@author: ramonpuga
"""

# Clasificación de bosques aleatorios (Random Forest Classification)

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Matriz X con todas las filas, y todas las columnas menos la última
X = dataset.iloc[:, [2,3]].values
# Vector y con todas las filas y la última columna
y = dataset.iloc[:, -1].values

# Dividir el data set en conjunto de training y de test
from sklearn.model_selection import train_test_split
# Aplicamos un porcentaje del 25% (0.25) para el test y un valor de selección alatoria de 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

""" 
# Los arboles no se basan en la distancia, no se usa el concepto de distancia Euclídea
# No tendría sentido el escalado, se podría utilizar pero por otra cuestión
# La idea es que las decisiones se tomen en base a los valores y no a su escalado
# Escalado (estandarización o normalización) de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Aplicamos y fijamos el metodo de estandarización a todas las columnas X
X_train = sc_X.fit_transform(X_train)
# Aplicamos el mismo metodo de estandarización que para los datos de Training
X_test = sc_X.transform(X_test)
"""

# En la clase si hace el escalado pero para que el gráfico quede mejor con los step = 0.01
# Yo lo dejo como la clase anterior, sin escalar y con step logicos a los valores sin escalar

# Ajustar el clasificador en el Conjunto de Training
from sklearn.ensemble import RandomForestClassifier
# Por defecto toma 10 arboles
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Representación gráfica de los resultados del algoritmo
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
# Dado que no hemos escalado, no tiene sentido hacer el step de 0.01, generaría millones de puntos
# En este caso para la edad de añoa en año --> step = 1
# Y para el sueldo de 500$ en 500$ --> step = 500
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Conjunto de Training)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

# Aparecen cajitas muy pequeñas de otro color, que son debidas al overfitting
# Está demadiado ajustado al entrenamiento, a individuos concretos

# Representación gráfica de los resultados del algoritmo
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()