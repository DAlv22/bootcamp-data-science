
# # Proyecto Megaline

# La compañia Megaline requiere de un modelo que pueda analizar el comportamiento de los clientes y pueda recomendar uno de los nuevos planes disponibles:
#
#     - Smart
#     - Ultra
#

# ## Objetivo
#
#     Generar un modelo que pueda sugerir a los clientes qué plan le iría mejor acorde a sus necesidades.
#
# ### Inicialización
#
#     La compañia entrega un archivo CSV donde se observa el comportamiento de los suscriptores que ya se han cambiado a los planes nuevos. Con esta información se procede a trabajar el modelo solicitado.
#
# ### Carga de librerias y de datos


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("/datasets/users_behavior.csv")
df.info()
display(df)


# ### Revisión de los datos
#
#     Se da una revisión a los datos. Este archivo cuenta con 5 columnas, las 4 primeras en tipo de datos float y la última en integer. Se describe el nombre de la columna y la información que se encuentra en ella.
#
#     •	сalls — número de llamadas
#     •	minutes — duración total de la llamada en minutos
#     •	messages — número de mensajes de texto
#     •	mb_used — Tráfico de Internet utilizado en MB
#     •	is_ultra — plan para el mes actual (Ultra - 1, Smart - 0)
#
#     Y tenemos un total de 3214 registros.
#
#     Se procede a una revisión más específica de los datos, con el objetivo de tener la certeza de estar trabajando con datos veraces.

# In[3]:


print(
    f'El número total de filas duplicadas en este archivo es de {df.duplicated().sum()} filas.')


# In[4]:


print(df.describe())


# Se observa que este archivo no cuenta con filas duplicadas. Revisando algunas estadisticas descriptivas de los datos se observa lo siguiente:
#
#     - Columna calls: El promedio de llamadas es de aproximadamente 63, con una variabilidad considerable (desviación estándar de aproximadamente 33). La cantidad mínima de llamadas es 0, lo que sugiere que hay usuarios que no realizaron ninguna llamada. El máximo de llamadas realizadas es de 244.
#
#     - Columna minutes:El promedio de minutos es de aproximadamente 438, con una variabilidad considerable (desviación estándar de aproximadamente 234). La cantidad mínima de minutos es 0, lo que sugiere que hay usuarios que no utilizaron minutos, lo que concuerda con los datos anteriores, al haber clientes que no realizaron ninguna llamada. El máximo de minutos utilizados es de 1632.
#
#     - Columnas messages: El promedio de mensajes es de aproximadamente 38, con una variabilidad considerable (desviación estándar de aproximadamente 36). La cantidad mínima de mensajes es 0, lo que sugiere que hay usuarios que no enviaron mensajes. El número máximo de mensajes enviados es de 224.
#
#     - Columnas mb_used: El promedio de megabytes utilizados es de aproximadamente 17207, con una variabilidad considerable (desviación estándar de aproximadamente 7570). La cantidad mínima de megabytes utilizados es 0, lo que sugiere que hay usuarios que no usaron datos. El valor máximo es 49745.
#
#     - Columna is_ultra: El promedio es de aproximadamente 0.31, lo que indica que alrededor del 31% de los usuarios tienen el plan ultra (1) y el 69% tienen el plan Smarth (0). La desviación estándar es de aproximadamente 0.46, lo que sugiere cierta variabilidad en la distribución de los planes.


# ## Segmentación de los datos
#
#     Una vez revisados los datos del dataset, se procede a la segmentación de los datos para poder trabajar con ellos.
#     Al tener un solo dataset y no contar con un dataset de prueba a futuro, se dividirá el total del dataset de la siguiente manera:
#     - 60% dataset de entrenamiento
#     - 20% dataset de validación
#     - 20% Dataset de prueba


features = df.drop(['is_ultra'], axis=1)
target = df['is_ultra']

features_train, x_valid, target_train, y_valid = train_test_split(
    features, target, test_size=0.4, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(
    x_valid, y_valid, test_size=0.5, random_state=12345)


# ## Modelos
#
# Una vez segmentados los datos acorde a lo anteriormente indicado, se procede a realizar 3 modelos de regresión logística para encontrar el modelo y los hiperparámetros que arrojen un resultado más acertado:
#
# - Árbol de decisión
# - Bosque aleatorio
# - Regresión logística
#
# Se utilizará 'accuracy' como métrica general que mide la proporción de predicciones correctas en relación con el total de predicciones.

# ### Árbol de decisión
#
# Se elabora árbol de decisión, ejecutando dentro del modelo, un verificador que nos ayuda a saber los mejores hiperparámetros para el mismo modelo.

# In[6]:


best_model_tree = None
best_result_tree = 0
best_depth_tree = 0
for depth in range(1, 6):
    model_tree = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    train_tree = model_tree.fit(features_train, target_train)
    predictions_tree = model_tree.predict(features_valid)
    result_tree = accuracy_score(target_valid, predictions_tree)
    if result_tree > best_result_tree:
        best_model_tree = model_tree
        best_result_tree = result_tree
        best_depth_tree = depth

print("Exactitud del mejor modelo en el conjunto de validacion:", best_result_tree)
print("Resultado alcanzado con una profundidad de", best_depth_tree)


# ### Bosque aleatorio
#
# Se elabora un bosque aleatorio, ejecutando dentro del modelo, un verificador que nos ayuda a saber los mejores hiperparámetros para el mismo modelo.

# In[7]:


best_model_forest = None
best_result_forest = 0
best_depth_forest = 0
for depth in range(1, 6):
    model_forest = RandomForestClassifier(random_state=12345, max_depth=depth)
    train_forest = model_forest.fit(features_train, target_train)
    predicition_forest = model_forest.predict(features_valid)
    results_forest = accuracy_score(target_valid, predicition_forest)
    if results_forest > best_result_forest:
        best_model_forest = model_forest
        best_result_forest = results_forest
        best_depth_forest = depth

print("Exactitud del mejor modelo en el conjunto de validacion:", best_result_forest)
print("Resultado alcanzado con una profundidad de", best_depth_forest)

# ### Regresión logística
#
# Se elabora un modelo de regresión logística.

# In[8]:


model_regression = LogisticRegression(random_state=12345, solver='liblinear')
train_regresson = model_regression.fit(features_train, target_train)
score_train = model_regression.score(features_train, target_train)
score_valid = model_regression.score(features_valid, target_valid)

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)

# ### Observaciones modelos
#
# Después de haber implementado 3 modelos diferentes, obtuvimos los siguientes resultados:
#
#     - Árbol de decisión
#         Con una exactitud de 0.785 se obtuvo el mejor modelo, con una profundidad de 3.
#     - Bosque aleatorio
#         Con una exactitud de 0.794 se obtuvo el mejor modelo, con una profundidad de 5.
#     - Regresión logística
#         Con una exactitud de 0.758 se obtuvo el mejor modelo.
#
# Para el presente estudio, se solicitaba un umbral de exactitud del 0.75, por lo tanto los 3 modelos pueden ser capaces de dar buenos resultados.
#
#
# Nombre | Precisión | Velocidad
# --- | --- | ---
# Árbol de decisión | Bajo | Alto
# Bosque aleatorio | Alto | Bajo
# Regresión Logística | Medio | Bajo
#
#
# Al ser un dataset de 3214 registros, podríamos elegir el modelo de "Bosque Aleatorio", debido a que es preciso y por el tamaño del dataset, no representará retrazos de calculo en la velocidad.

# ## Prueba Calidad del modelo
#
# Los 3 modelos han superado el umbral de exactitud, por lo cual, a los 3 se les medirá la calidad del modelo usando el conjunto de prueba previamente considerado y apartado de nuestro dataset original.

# ### Árbol de decisión

# In[9]:


predictions_tree_test = model_tree.predict(features_test)
result_tree_test = accuracy_score(target_test, predictions_tree_test)

print("Precisión del modelo de Árbol de Decisión en el conjunto de prueba:", result_tree_test)


# ### Bosque aleatorio

# In[10]:


predictions_forest_test = model_forest.predict(features_test)
result_forest_test = accuracy_score(target_test, predictions_forest_test)

print("Precisión del modelo del bosque aleatorio en el conjunto de prueba:",
      result_forest_test)


# ### Regresión logística

# In[11]:


result_regression_test = model_regression.score(features_test, target_test)

print("Precisión del modelo del regresión logística en el conjunto de prueba:",
      result_regression_test)


# ### Observaciones
#
# Se realiza la prueba de calidad de los modelos utilizando el apartado de prueba para verificar que tanta exactitud tenian los modelos previamente entrenados, se observó lo siguiente:
#
#     - El modelo de árbol de decisión tuvo una exactitud en su predicción del 78.3%
#     - El modelo del bosque aleatorio tuvo una exactitud en su predicción del 79%
#     - El modelo de Regresión logística tuvo una exactitud en su predicción del 74%
#
# Por lo que podríamos confiar más en el modelo del bosque aleatorio ha sido el más exacto.

# ## Prueba de cordura al modelo
#
# Se realiza una prueba de cordura al modelo por medio de un dummy clasifier, utilizando la estrategia que dió los mejores resultados, en este caso 'most_frequent'.

# In[12]:


dummy_model = DummyClassifier(strategy='most_frequent')

dummy_model.fit(features_train, target_train)
predictions_dummy_valid = dummy_model.predict(features_valid)
accuracy_dummy_valid = accuracy_score(target_valid, predictions_dummy_valid)

predictions_dummy_test = dummy_model.predict(features_test)
accuracy_dummy_test = accuracy_score(target_test, predictions_dummy_test)

print("Precisión del modelo Dummy en el conjunto de validación:", accuracy_dummy_valid)
print("Precisión del modelo Dummy en el conjunto de prueba:", accuracy_dummy_test)


# Aún utilizando la mejor estrategia, los resultados arrojados por DummyClassifier es inferior a la calidad de nuestros modelos previamente trabajados. Logrando unicamente una precisión del 70% en el conjunto de validación y un 68% en el conjunto de prueba.

# ## Conclusión
#
# Se trabajó la creación de un modelo para la empresa Megaline partiendo de un documento que nos indica el comportamiento de los suscriptores que ya se han cambiado al plan nuevo. El modelo debe ser capaz de sugerir el tipo de plan que vaya mejor con los comportamientos del cliente.
#
# Al tener un solo Dataset, se dividieron los datos disponibles en 3, con la siguiente proporción:
#     - 60% dataset de entrenamiento
#     - 20% dataset de validación
#     - 20% Dataset de prueba
#
# Se entrenan 3 tipos de modelos, con una exactitud en sus predicciones de:
#     - Árbol de decisión, 78.5%
#     - Bosque aleatorio, 79.4 %
#     - Regresión logística, 75.8%
#
# Posteriormente se realizaron los test con los datos de prueba, logrando la siguiente exactitud en las predicciones:
#     - Árbol de decisión, 78.3%
#     - Bosque aleatorio, 79 %
#     - Regresión logística, 74%
#
# Teniendo un umbral de exactitud del 75%, podríamos indicar que el tanto el árbol de decisión como el bosque aleatorio superaron el umbral indicado. El modelo de regresión logistica podría no tener la precisión que estamos esperando, pero dependiendo del la cantidad de datos que deba trabajar el modelo en un futuro, probablemente podría ser la regresión logística una alternativa.
#
# Por último, con el objetivo de realizar una prueba de cordura, se utilizó el modelo dummy, con estrategia de 'most_frequent'. Teniendo como resultado que en su conjunto de validación tuvo un 70.6% de precisión y un 68.4% de precisión en la prueba.
# Revisando así que nuestros modelos muestra un tiene un rendimiento superior al de la prueba dummy.
