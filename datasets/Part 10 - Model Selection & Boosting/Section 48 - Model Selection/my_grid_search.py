#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:29:19 2020

@author: ramonpuga
"""

# Grid Search

# Hay parámetros que son hiperparámetros que el algoritmo no aprende y normalmente son argumentos
# a indicar a la hora de ejecutar el algoritmo. Se trata de encontrar estos hiperparámetros óptimos 
# Otra duda, es saber que modelo elegir, según el problema: Clusteing, Regresión, Clasificación
# Grid Search también nos ayuda a saber si elegir un modelo lineal o no lineal

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(X_train, y_train)


# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()



# Aplicar la mejora de Grid Search Cross Validation para optimizar el modelo y sus parámetros
from sklearn.model_selection import GridSearchCV
# Declaramos una variable en modo diccionario python para evaluar todos los parámetros del algoritmo SVC usado en el ajuste
# Una lista con los identificadores en formato clave, el nombre de los parámetros y los valores que queremos que evalue
# En este caso si no sabemos si usar lineal o no, lo podemos evaluar según un parametro que existe en este SVC
"""
parameters = [{'C': [1, 10, 100, 1000], 
               'kernel': ['linear', 'rbf']}]
"""
# Si queremos que se han combinaciones completas es mejor crear varios diccionarios separados por comas
# De esta manera por ejemplo en el segundo con rbf podemos especificar parámetros propios de éste: gamma
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}
              ]
# Llamamos a la función con parametros, como el objeto de ajuste, el diccionario, 
# la métrica de rendimiento 'scoring' para medir la eficacia usamos la precisión 'accuracy' 
# al ser una métrica de clasificación, y la evaluación cruzada 'cv', igual que en k-fold
# La métrica es para que el GridSearch evalue los resultados de cada combinación
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
# Ojo si el dataset es muy grande se hacen muchas combinaciones de parámetros 
# más el propio cv, puede ser interesante usar njobs para utilizar n cores
grid_search = grid_search.fit(X_train, y_train)

# Ver el mejor resultado, accuracy del objeto grid_search
best_accuracy = grid_search.best_score_
# La mejor selección de párametros posibles nos va a dar una precisión del 90%

# Averiguamos cuales son
best_parameters = grid_search.best_params_
# C = 1000, gamma = 0.1, kernel = rbf

# Con estos resultados se podría volver a ejecutar pero con parámetros cercanos a estos
"""
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0,07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]}
              ]
"""


# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Kernel (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
