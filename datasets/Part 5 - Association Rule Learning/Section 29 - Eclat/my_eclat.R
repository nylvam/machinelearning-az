# Eclat

# Preprocesado de Datos
#install.packages("arules")
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Entrenar algoritmo Eclat con el dataset
# En este algoritmo no hay nivel de confianza
# Añadimos que como mínimo haya 2 items (minlen)
rules = eclat(data = dataset, 
                parameter = list(support = 0.004, minlen = 2))

# Visualización de los resultados
inspect(sort(rules, by = 'support')[1:10])
