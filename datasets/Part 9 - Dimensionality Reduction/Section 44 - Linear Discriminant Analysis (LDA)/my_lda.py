#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:25:42 2020

@author: ramonpuga
"""

# LDA Linear Discriminant Analysis

# Copiado del PCA

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


# Reducir la dimensión del dataset con LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# Cuantas direcciones de separación máxima de nuestras categorías
# Ponemos 2 para poder visualizar
# A diferencia de PCA, añadimos las variables dependientes porque es Supervisado
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
# En el conjunto de test no necesito la VD porque se trata de que prediga con los datos de training
X_test = lda.transform(X_test)


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
# Ha salido 36 aciertos/36 = 100% de fiabilidad


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
# DL = Discriminante lineal
plt.xlabel('DL1')
plt.ylabel('DL2')
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
plt.xlabel('DL1')
plt.ylabel('DL2')
plt.legend()
plt.show()