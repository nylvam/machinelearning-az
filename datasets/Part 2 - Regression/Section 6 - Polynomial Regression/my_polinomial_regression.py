#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:48:55 2020

@author: ramonpuga
"""

# Regresión Polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
# Esta instruccion elige la columna 1 y crea un vector. En Variable explorer Size = (10,)
#X = dataset.iloc[:, 1].values
# Sin embargo esta instrucción columna de la 1 a la 2, lo que crea es una matriz Size =(10,1)
# El algoritmo de ML espera una Matriz de características
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

"""En este caso solo con 10 filas no merece ls pena dividir training y testing, perderiasmos información
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Tampoco haremos escalado nos interesa ver como es esa relacion no lineal de los datos
# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Vamos antes a ajustar con regresión lineal para ver las diferencias
# Ajustar la Regresión Lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la Regresión Polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
# En este caso necesitamos introducir no solo la columna de X, sino X2 (cuadrado), X3 (cubo), etc.
# Por defeco degree = 2 es decir X y X2
poly_reg = PolynomialFeatures(degree = 2)
# No solo ajustamos, sino que transformamos y luego ajustamos
X_poly = poly_reg.fit_transform(X)
# En realidad crea 3 columnas, la primera todos '1' se corresponde al término indepentiente
# La segunda es nuestra X
# La tercera es el cuadrado de X

# La clase que se utiliza para hacer una Regresión Polinómica es la misma que la lineal
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualización de los resultados del Modelo Líneal
# Creamos la nube de puntos
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del Modelo Polinómico
# Creamos la nube de puntos
plt.scatter(X, y, color = "red")
# Pintamos la regresión, en este caso una curva
#plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
# Usamos este código para poder reutilizarlo, es igual a la línea anterior
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Qué pasaria si ajustasemo con un grado mayor, por ejemplo hasta X al cubo
# Tendriamos un modelo mejor?
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Visualizamos
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Grado 4
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Visualizamos
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# NUEVO con arange creamos una secuenciua de datos entre el valor mínimo y máximo y a intervalos especificado
# Con esto haremos que la grafica no sean secciones de recta
# Primero creamos una matriz entre 1 y 10 a intervalos de 0,1, es decir 90 elementos
X_grid = np.arange(min(X), max(X), 0.1)
# Ahora es un vector fila, para convertirlo en vector columna, hacemos reshape
X_grid = X_grid.reshape(len(X_grid), 1)
# Ahora con este grid tendremos más puntos para representar la curva

# Creamos la nube de puntos
plt.scatter(X, y, color = "red")
# Pintamos la regresión, en este caso una curva
#plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
# Usamos este código para poder reutilizarlo, es igual a la línea anterior
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Predicción de nuestros modelos
# Predecir los datos para un valor de entrada, en este caso con un nivel entre el 6 y el 7 (6.5)
# Primero con el modelo de regresión lineal
# La función predcit ahora necesita un parámetro en formato array 2D, por eso el doble '['
lin_reg.predict([[6.5]])

# Para predecir el modelo de regresión polinómica, necesitamos preparar grado 4
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

