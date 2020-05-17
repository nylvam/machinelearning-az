#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:33:31 2020

@author: ramonpuga
"""

# Upper Confidence Bound (UCB)

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de UCB (mirar pasos en slides curso)
# Este código esta condicionado por la estructura de los datos, en formato dataframe (tabla)
# Lo normal seria tener anuncio mostrado y si se ha pulsado o no, incluso en tiempo real
import math
# Inicializar variables N (número de rondas), d (número de anuncios)
N = 10000
d = 10
# Paso 1: A cada ronda 'n' : El número de veces que el anuncio se selecciona y la suma de recompensas
# Crear un vector de tamaño inicial 'd' (número de anuncios a mostrar)
# Inicializar el vector con ceros: truco poner 0 dentro y multiplicar por el tamaño 'd'
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0
# Paso 2: A partir de estos dos números calculamos: La recompensa media y el intervalo de confianza
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        # Para poder actuar necesitamos que por lo menos el anuncio se hay mostrado 1 vez
        # Las 10 primeras rondas (la primera aparición de cada anuncio) no se tendrán en cuenta
        if(number_of_selections[i]>0):
            # Recompensa media
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            # Intervalo de confianza
            # Primero calculamos el delta
            # Ojo como la ronda 'n' comienza en 0 y no en 1, el log(0) dará error, así que sumamos 1
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            # Intervalo de confianza superior
            upper_bound = average_reward + delta_i
        else:
            # Establecemos un número muy grande 10 eleveado a 400
            # En la ronda 0 será el 0, en la ronda 1 el 1, etc.
            # Asi aseguramos que las 10 primeras rondas no se usan en los cálculos
            # y forzamos que cada anuncio pasa en las primeras rondas: 0, 1, 2, 3, 4,..
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            # Guardamos el intevalo de confianza mayor y que anuncio es
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
    
# Histogrma de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
