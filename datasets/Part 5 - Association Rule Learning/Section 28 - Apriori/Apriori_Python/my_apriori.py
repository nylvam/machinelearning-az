#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:37:30 2020

@author: ramonpuga
"""

# Apriori

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
# Por defecto toma la primera lína como títulos de la columnas en este caso salen
# cabeceras, que son en realidad los datos del primer cliente, 7500 filas
#dataset = pd.read_csv('Market_Basket_Optimisation.csv')
# Arreglamos la carga sin cabeceras header = None, 7501 filas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# Cada línea es un ciente con los productos comprados
# Es un dataframe cuadrado, si un cliente ha comprado 3 items, el resto de columnas son 'nan'
# Se trata de preprocesar el dataframe para hacer listas de productos por cliente
transactions = []
# Bucle para recorrer todas las transacciones de clientes, van de la 0 a la 7500
# Con este conseguimos una lista de listas
for i in range(0, 7501):
    # Añadimos un '[' al principio y al final para crear una lista dentro de cada transacción
    # Cogemos los values, pero dado que con textos los convertimos en string str()
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])


# Entrenar el algoritmo de Apriori 
# Importamos el código de Apyori (librería externa)
from apyori import apriori
# Argumentos soporte, confianza, lift, lenght = número de items mínimo en la lista
# El dataset corresponde a 1 semana de ventas del supermercado francés
# Soporte: queremos elementos que se compren mínimo 3 veces al dia: 3 x 7 = 21 / 7500 = 0,0028 -> 0.003
# Confianza: confianza alta no saldrán reglas, confianza muy baja serán relaciones sin valor
# Confianza: Nivel de confianza mínimo del 20%, las reglas se cumplen al menos el 20% de las veces que se compra el primer item
# Lift: confianza / soporte -> queremos reglas relevantes, por ejemplo queremos que el cociente sea superior a 3 veces
# Mirar ejercicios en R
# Min_Lenght = al menos 2 itmes en la cesta de la compra
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, 
                min_lift = 3, min_lenght = 2)

# Visualización de los resultados
results = list(rules)
# Obtenemos 154 reglas de asociación, las ordena de la más relevante a la menos (lift)
# No parece que el resultado salga ordenado por lift, habria que aplicar un sort
# Por consola podemos ver resultados para la relación 0 --> pollo y salsa ligera
# En consola, se puede ver el soporte, la confianza y el lift
results[0]
# Otra fila salsa de champiñones y escalope, el 30% de las veces se compran juntos
results[1]
# Siguiente fila pasta y escalope, con un soporte del 0.005 -> 0.5% compra pasta
# Que se compren a la vez es del 37% (confianza)
results[2]
# Con estas reglas se ponen a prueba en el supermercado y se analizan el impacto

