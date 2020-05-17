#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:00:26 2020

@author: ramonpuga
"""

# Natural Language Processing

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
# Contiene valoraciones de un restaurante, y una columna si es positiva o negativa 
# para el entrenamiento del algoritmo
# En este caso es un tsv (separado por tabuladores),
# para evitar confusión con ',' en el propio texto de las valoraciones del restaurante
# En el texto hay comillas doble ", con el parametro quoting = 3 ignoramos las comillas dobles
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Limpieza de texto
# Eliminar palabras que no aportan (conjunciones, articulos,...), números,
# formas verbales a infinitivos, pasar todo a minúsculas, etc.
# Importamos librería de Expresiones Regulares
import re
# Usamos las máscaras a eliminar, con ^ indicamos lo contrario a, lo que no queremos eliminar, que será más corto
# Eliminamos todo lo que no sea una letra por un espacio en blanco
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
# Ponemos todo en minúsculas
review = review.lower()
# Eliminar palabras irrelevantes (artículos, conjunciones, preposiciones, ...)
# Importamos la libreía de NLP natural language tool kit
import nltk
# Descargamos la lista de palabras inúlites (el trabajo ya lo ha hecho la librería) -> stopwords
# Al ejecutar, en la consola se ve donde se ha descargado y descomprimido la lista
nltk.download('stopwords')
# Del cuerpo de palabras necesito importar las stopwords
from nltk.corpus import stopwords
# Dividimos la cadena de caracteres en un arry de palabras
review = review.split()
# Recorremos la palabras de la lista y la eliminaremos si está en la lista de stopwords
# Hacemos un for en la propia sentencia
# Accedemos a la lista de palbras del diccionario inglés, 
# Es una lista, le ponemos set para que sea un conjunto con las palabras en inglés
# En una lista hay un orden y sería necesario recorrer toda la lista por cada palabra,
# si es un conjunto se busca directamente
review = [word for word in review if not word in set(stopwords.words('english'))]
# Vamos a eliminar declinaciones, y poner vernos en infinitivo
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
# Se puede rehacer la instrucción anterior de eliminar stopwords, solo para las que quedan
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
# Ahora lo convertimos de nuevo en una cadena de texto con las palabras que han quedado
# La parte ' '. indica que las una con un espacio de separación
review = ' '.join(review)

# Lo hemos hecho para una frase, ahora se trata de generarlizarlo para todo las líneas del dataset
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Crear el Bag of Words
# Después de limpiar las valoraciones y crear el corpus, cada palabra diferente llegará a ser una columna
# Tendremos una matriz dispersa, con tantas columnas como palabras diferentes 
# y por cada columna (palabra) un 1 si existe en en esa valoración y un o si no lo está
# Será una matriz llena de ceros, con mil filas y miles de columnas -> tokenización
# Transformar las frases en vectores de frecuencia, tantas veces aparece una palabra en una frase
from sklearn.feature_extraction.text import CountVectorizer
# Tiene muchos parametros que hacen lo que hemos hecho antes, minúsculas, stopwords, etc.
# Nosotros lo hemos hecho paso a paso, y así hay más control, y hay más opciones
# Salen 1565, pero con max_features el algoritmo coge el máximo que le digamos por prioridad
# así puede quitar palabras que no hemos eliminado, como por ejemplo nombres propios, que solo salndrán una vez y no son relevantes
cv = CountVectorizer(max_features = 1500)
# Crear la matriz dispersa, sparse
# Ajustar, mirar todas las palabras y transformar para crear la matriz sparse
# El objeto fit_transform que ha generado CountVectorizer, lo convertimos a una matriz para los siguientes pasos
X = cv.fit_transform(corpus).toarray()
# Esta será la matriz de características
# La variables dependiente será la columna de si es positiva o no
y = dataset.iloc[:, 1].values

# Ahora toca usar un algotirmo de clasifiación de los vistos en el curso

# Utilizamos el de Naïve Bayes
# Dividimos en conjunto de entrenamiento y test: 800 - 200 (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# No hacemos el escalado, porque ya están escaladas entre 0 y 1

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(55+91)/200
# Da una fiabilidad del 73%, con más entrenamiento sería más preciso

# No hacemos la represnetación gráfica del algoritmo porque tenemos 1500 columnas

