#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:58:11 2020

@author: ramonpuga
"""

# ACP (Análisis de Componentes Principales)

# Es un problema de Clasifición, y partimos del algoritmo de Regresión Logistica, 
# lo importate no es la regresión sino la reducción de variables

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Wine.csv')
# Son 178 filas y 14 variables (la última es la dependiente) -> es un ejemplo tipico del ML
# Esta en 'UCI repository wine dataset' (google.com)
# Se trata combinar las variables para establecer una única que represente a todas

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Pare reducir la dimensión el cambio de escala es muy recomendable
# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Reducir la dimensión del dataset con ACP
from sklearn.decomposition import PCA
# Agumento principal componentes de la varianza, 
# Seleccionamos 2 para poder visualizar en un grafíco, pero se puede poner más componentes
# El número de componentes debe crear un equlibrio entre cuantas son y el % de la varianza que queremos explicar
# Si ponemos = none, se hará un análisis de componentes con todas y poder visualizar 
# la varianza acumulada que va explicando cada una de las componentes, con 1 x%, con 2 x%, etc.
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# En el ACP de Python hay una función que devuelve todas las componente principales y el % de varianza explicada
# para no tener que calcularlo a mano
explained_variance = pca.explained_variance_ratio_
# Devuelve las 13 componentes principales de nuestro dataset (= nº variables independientes)
# La 1ª de ellas que será una transformación de las 13 variables originales y explica el 36,9% de la varianza
# Si tomasemos 2 componentes principales explicariamos el 56% (37+19) de la varianza global del dataset

# Vamos a coger 2 para poder respresentarlo aunque solo llega al 56%, lo mejor sería llegar a un 70% u 80%
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
# Hay que volver a cargar las variables, lineas previas


# Ajustar el modeo de Regresión Logística en el Conjunto de Entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión, para evaluar el modelo
# Sale una matriz 3x3 porque hay tres valores(1,2,3) y no solo 0s y 1s
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Ha salido 35 aciertos/36 = 97,22% de fiabilidad


# Hay que adaptarlo, porque el código original estaba preparadao para 2 categorias y ahora son 3
# Solo es necesario añadir un color para la 3ª categoria en ListedColormap

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
# CP = Componente principal
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()

# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()
