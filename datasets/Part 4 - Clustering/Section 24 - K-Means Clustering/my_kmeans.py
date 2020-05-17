#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:22:12 2020

@author: ramonpuga
"""

# K-Means

# Importar librerías de trabajao
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos con pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Método del codo para averiguar el número de clusters
from sklearn.cluster import KMeans
# Vamos a calcular 10 KMeans, la WCSS y dibujar el gráfico
# Calcular WCSS
wcss = []
# 10 segmentos range 1 to 11, Python no incluye el último valor del rango 11
for i in range(1, 11):
    # Para no caer en la trampa de la inicialización aleatoria, usaremos kmeans++
    # Ponemos un valor máximo, por si el algoritmo no acaba nunca, por defecto max_iter = 300
    # Inicialización alteatoria n_init, por defecto = 10
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    # Calcular la suma de los cuadrados, parámetro inertia_ ya lo calcula el algoritmo
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title("Métocod del codo")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS(k)")
plt.show()

# El número óptimo seria 5
# Aplicar el método de k-means para segmentar el dataset
kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
# Necesitamos hacer el ajuste de kmeans, pero también la predicción de a que cluster pertenece cada punto
# Varemos en cada fila a que cluster pertenece del 0 al 4
y_kmeans = kmeans.fit_predict(X)

# Visualización de los clusters
# Pintamos los puntos, la nube de puntos, cada una con su sector al que corresponde
# Seleccionamos los puntos cuyo cluster == 0 para pintarlos de un color, 1 para otro color, etc.
# Seleccionamos de la matriz X solo las filas cuyo cluster == 0, y en el eje de las x(0) ira la primera columna (0), 
# de la matriz de caracteristicas X y en el eje de las y(1), la columnas número 1
# Elegimos el tamaño de la bolita s = 100, el color y una etiqueta para diferenciarlo
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cluster 1")
# Repertimos la línea por cada cluster
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
# Pintamos los baricentros (centro geométricos de cada cluster), todas las filas columna 0 y todas las filas columna 1
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Baricentro")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de gasto (1-100)")
# Añadimos la leyenda de colores
plt.legend()
plt.show()

# Podríamos poner nombre a cada cluster --> usuarios que gastan mucho y ganan poco, etc.
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Estándar")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Baricentro")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de gasto (1-100)")
plt.legend()
plt.show()

# Aunque el gráfico es de 2 dimensiones, el algoritmo es genérico y permite poner todas las características que sean necesarias
