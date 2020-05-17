#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:44:59 2020

@author: ramonpuga
"""

# k-Fold Cross Validation

# Hasta ahora hemos hecho un conjunto de entrenamiento y otro de test, y con
# eso evaluabamos el redimiento del modelo. Pero podemos tener un problema de varianza de los datos
# Pero puede ocurrir que luego utilicemos otro conjunto de test y de resultado totalmente diferentes

# Está técnica k-Fold Cross Validation mejora el problema de la varianza

# Vamos a partir de un modelo ya usado durante el curso el Kernel-SVM

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


# Es necesario cambiar la manera de dividir el conjunto de entrenamiento y test
# Se crean 10 conjuntos y devuelve 10 precisiones. Cada combinación crea 9 para entrenar y 1 para validar
# Aplicar k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
# El primer parametro es el estimador que será el modelo a evaluar es decir el objeto classifier usado en el ajuste
# El segundo el conjunto de datos, y el tercero la variable a predecir, cv = grupos a crear, njobs permite hacer calculos en varias cpus
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# Vamoa a sacar la media de las precisiones -> Obtenemos un 90%
accuracies.mean()
# Podemos calcular la desviación estandar -> Nos dirá si hay una gran varianza entre las diferenctes iteracciones
# Se obtiene un 0,065. Una varianza del 6,5% que es bastante baja, podría llegar a ser del 90%
accuracies.std()
# Es decir sobre el 90% +/- 6,5% arriba o abajo: 90-6= 84 <-> 90+6=96


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