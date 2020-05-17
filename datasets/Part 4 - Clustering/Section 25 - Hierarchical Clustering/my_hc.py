#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:28:37 2020

@author: ramonpuga
"""

# Clustering Jerárquico

# Importar librerías de trabajo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar los datos del centro comercial con pandas
dataset = pd.read_csv("Mall_Customers.csv")

# Obtenmos del dataser un conjunto con todas las filas y las columnas que nos interesen
# Ponemos .values para seleccionar el valor y no solo la posición
X = dataset.iloc[:, [3, 4]].values


# Utilizar el dendrograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
# Invocamos el dendrogram, usamos linkage para utilizar el algoritmo aglomerativo
# Hay que indicar para unir los objetos, varios parámetros
# ward = método que minimiza la varianza que existe entre los puntos de los clusteres
# Al igual que en kmeans usamos el wcss para minimizar la suma, en clustering jerárquico 
# se hace lo mismo pero minimizando la varianza entre los clusteres
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# La instrucción anterior ya genera un gráfico, lo que hacemos a continuación es poner títulos
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# La línea vertical más a la derecha sería la de mayor distancia, sin pasar por encima de ninguna horizontal
# El número de verticales que corta sería de 5 y por lo tanto tendremos K = 5

# Ajustar el clustering jerárquico a nuestro conjunto de datos 
# Técnica aglomerativo cómo la más común, la otra era la divisitivo
from sklearn.cluster import AgglomerativeClustering
# El parámetro affinity es para indicar el método de distancia
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
# Ajustamos obteniendo la predicción de a que cluster pertenece cada cliente
y_hc = hc.fit_predict(X)

# Visualización de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Estándar")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de gasto (1-100)")
plt.legend()
plt.show()

# Con el gráfico, podemos hacer análisis de los datos y los grupos creados
# Si hacemos clustering para más de 2 dimensiones, no podremos utilizar la parte de visualización
