#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:06:30 2020

@author: ramonpuga
"""

# Regresion Lineal Múltiple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
X = np.array(ct.fit_transform(X), dtype=np.float)

# Eliminar una columna dummy para evitar la multicolinealidad
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
# Podría ser que lo que sobra es el término independiente 'b0'
# Se agrega una columna al conjunto de datos toda llena de 1 (unos) y así poder calcular su p-valor
# Así podemos determinar si ese término independiente debe ser 0 (cero)
# Añadimos una columna al inicio con todo 1s, en este caso una dupla de 50 filas en 1 columna
# La librería statsmodels requiere que el término independiente sea la primera columna
# Al aajustar el modelo de RLM se determinará si el coficiente del término independiente es 0 o no, y poder medir su p-valor
# Forzamos que sean enteros, sino por defecto serían float
# Lo último es indicar si queremos añadir en fila (Axis = 0) o columna axis = 1
#X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)
# Por defecto append lo añade al final, para ponerlo al inicio le damos la vuelta
# Genermamos un array de 1s y le añadimos la X
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# Crear el SL (nivel de significación)
SL = 0.05

# Crear una matriz de características óptimas
# En la Eliminación hacia atras, partimos de todas ellas
# Podemos todas las columnas, para visualmente ir viendo el proceso de eliminado
X_opt = X[:, [0, 1, 2, 3, 4, 5]] 

# La librería statsmodel necesita volver a ajustar el modelo, un nuevo regressor, además tiene una nueva columna
# Se llama OLS (Ordinary Least Squares), técnica de los mínimos cuadrados ordinarios
# Variable endogena, y exogena
# El fit vuelve a generar la RLM
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Aplicamos la funicón summary() que nos devolverá p-valor (P>[t]) para cada variable y otros parámetros
regression_OLS.summary()

# La variable con el p-valor más grande es x2 (0.990) y superior a SL
# Eliminamosla columna 2 del X original
X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

# La variable con el p-valor más grande es x1 (0.940) y superior a SL
# Eliminamos la columna 1 del X original
X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

# La variable con el p-valor más grande es x2 (0.602) y superior a SL
# Eliminamos la columna 4 del X original
X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

# La variable con el p-valor más grande es x2 (0.060) y superior a SL
# Eliminamos la columna 5 del X original
X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

# Ya no hay ningún p-valor superior a SL = 0.05
# Al final nos ha quedado un modelo de Regresión Lineal Simple: una constante y una variable 'R+D Spend'
# Esto es porque estamos en modo práctico y siendo rigurosos con la Eliminación hacia atrás
# A veces no basta con aplicar solo el criterio de p-valor
# En la práctica habría que adoptar un criterio de eliminación (Akaike o Bayesiano)

# REVISAR en esta sección las dos lecciones de texto de 1 minuto

# Automatización de la Elimnación hacia atrás

"""Se ha adaptado el código para que utilice la transformación .tolist() 
sobre el ndarray y así se adapte a Python 3.7"""

# Eliminación hacia atrás utilizando solamente p-valores
# import statsmodels.formula.api as sm -> Ya no está en formula
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars): 
        # El for pasa por 0,1,2,3,4 y 5 (no llega hasta el NumVars = 6)
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)  
        # Comprobar si algún p-valor (el máximo) superó SL
        if maxVar > sl:            
            for j in range(0, numVars - i):  
                # El primer i es 0, por lo tanto el for ira de range(0, 6 - 0)
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    print(regressor_OLS.summary())   
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# Eliminación hacia atrás utilizando  p-valores y el valor de R Cuadrado Ajustado
# import statsmodels.formula.api as sm -> Ya no está en formula
import statsmodels.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar): 
                    # Guardamos la columna en temp para poder restarurarla
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):
                        # hstack concatena matrices en orden secuencial
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
