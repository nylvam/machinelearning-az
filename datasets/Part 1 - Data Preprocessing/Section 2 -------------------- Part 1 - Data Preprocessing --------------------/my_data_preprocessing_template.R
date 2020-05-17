# Plantilla para el Pre Procesado de Datos

# Importar el dataset
dataset = read.csv('Data.csv')
# Ejemplo de seleccionar un conjunto de datos del dataset original
# dataset = dataset[, 2:3]

# Dividir los datos en conjunto de training y conjunto de test
# install.packages("caTools")
library(caTools)
# Establecer un valor de semilla para la selección de datos
set.seed(123)
# Establecemos un 80% de las filas como ratio de división (training)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores para las columnas 2 y 3 (2ª y 3ª)
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])
