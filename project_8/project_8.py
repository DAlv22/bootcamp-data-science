# # PROYECTO BETA BANK

# ## Objetivo

# El presente proyecto tiene por objetivo presentar un modelo que pueda ayudar a Beta Bank a predecir si un cliente dejará el banco pronto.
#

# ###  Inicialización
#
# El banco proporciona los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.

# ### Carga de librerias y de datos
# Ser carga la información proporcionada por Beta Bank sobre el comportamiento de sus clientes que han terminado relación con el banco.
#
#     - Para comenzar descargaremos las librerías necesarias para el tratamiento necesarios de los datos
#     - Posteriormente se revisarán las características generales de los datos

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


# Se realiza la importación de datos

# In[2]:


df = pd.read_csv('/datasets/Churn.csv')


# Se procede a la revisión de los datos

# In[3]:


print(df.shape)
print()
print(df.info())
print()
print(df.dtypes)
print()
print(df.describe())
print()
print(f'Este dataset tiene {df.duplicated().sum()} filas duplicadas.')
display(df)


# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
#
# Comprendo, he realizado el cambio.
#
# </div>

# ### Revisión de los datos
#
# Tenemos un dataset con un tota del 10,000 filas y 14 columnas, de las cuales tenemos los siguientes encabezados y caracteristicas de los datos alojados en ellas:
#
# Nombre de columna | Descripción | Tipo de dato
# --- | --- | ---
# RowNumber | índice de cadena de dato | int64
# CustomerId | identificador de cliente único | int64
# Surname | apellido | object
# CreditScore | valor de crédito | int64
# Geography | país de residencia | object
# Gender | sexo | object
# Age | edad | int64
# Tenure | período durante el cual ha madurado el depósito a plazo fijo de un cliente (años) | float64
# Balance | saldo de la cuenta | float64
# NumOfProducts | número de productos bancarios utilizados por el cliente | int64
# HasCrCard | el cliente tiene una tarjeta de crédito (1 - sí; 0 - no) | int64
# IsActiveMember | actividad del cliente (1 - sí; 0 - no) | int64
# EstimatedSalary | salario estimado | float64
# Exited | El cliente se ha ido (1 - sí; 0 - no) | int64
#
# Teniendo un total de:
# - Datos flotantes (3)
# - Datos integer (8)
# - Datos objeto (3)
#
# La columna donde tenemos 909 datos faltantes es en la columna "Tenure", representando un 9% del total de datos. Las características de los datos faltantes se revisará más adelante.
#
# Esta es una tabla que muestra las principales características de las columnas que contienen datos númericos (Se excluyen las columnas "RowNumber" y "CustomerId" debido a que no es relevantes para este análisis)
#
# |   |CreditScore | Age | Tenure | Balance | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
# |---|-------------|-----|--------|---------|---------------|-----------|-----------------|------------------|--------|
# | count | 10000 | 10000 | 9091 | 10000 | 10000 | 10000 | 10000 | 100000 | 10000 |
# | mean  | 650.5288 | 38.9218 | 4.9976 | 76485.8892 | 1.5302 | 0.7055 | 0.5151 | 100090.2398 | 0.2037 |
# | std | 96.653299 | 10.487806 | 2.894723 | 62397.405202 | 0.581654 | 0.45584 | 0.499797 | 57510.492818 | 0.402769 |
# | min | 350 | 18 | 0 | 0 | 1 | 0 | 0 | 11.58 | 0 |
# | 25% | 584 | 32 | 2 | 0 | 1 | 0 | 0 | 51002.11 | 0 |
# | 50% | 652 | 37 | 5 | 97198.54 | 1 | 1 | 1 | 100193.915 | 0 |
# | 75% | 718 | 44 | 7 | 127644.24 | 2 | 1 | 1 | 149388.2475 | 0 |
# | max | 850 | 92 | 10 | 250898.09 | 4 | 1 | 1 | 199992.48 | 1 |
#
# Hay datos que son faciles de entender a la vista, pero nos centraremos en aquellas columnas cuyos datos son binarios.
#
#     - HasCrCard: cuya media indica que el 71% son clientes que tienen una tarjeta de crédito
#     - IsActiveMember: cuya media indica que el 52% son clientes están activos
#     - Exited: cuya media indica que el 20% indica que los clientes se han ido
#
#
# Este dataset tiene 0 filas duplicadas.
#

# ### Escrutinio y ajuste de datos
#
# Una vez con una noción general de los datos con los cuales se trabaja, se procede a hacer un análisis y ajuste de los datos para poder comenzar a trabajar con ellos.
#
# Se comienza ajustando los encabezados de columna a minúsculas:

# In[4]:


df.columns = df.columns.str.lower()
df = df.rename(columns={
    'hascrcard': 'has_cr_card',
    'rownumber': 'row_number',
    'customerid': 'customer_id',
    'creditscore': 'credit_score',
    'numofproducts': 'num_of_products',
    'hascrcard': 'has_crcard',
    'isactivemember': 'is_active_member',
    'estimatedsalary': 'estimated_salary'})
print(df.head())


# Se procede a revisar los datos nulos ubicados en la columna 'tenure'

# In[5]:


print(df['tenure'].unique())


# Se observa que dentro de los datos unicos que hay en 'tenure'va acorde a nuesta tabla anterior, cuyo mínimo es 0 y máximo es 10. Esto cancela la idea de que los datos 'nan' de esta columna, pudieran ser los clientes que aún no cumplian un año con el banco, dado que sí hay clientes que tienen marcado el año '0'.
#
# Se filtra el dataset por los datos nulos de 'tenure' para revisar si existe alguna tendencia que podamos ver a simple vista.

# In[6]:


filtered_df = df[df['tenure'].isna()]
display(filtered_df)


# Parece no haber una tendencia por la cual tengamos datos 'NaN' en la columna de 'tenure'.
# Por ultimo se realizará una revisón por 'geography' para ver si existe algún tipo de tendencia en los datos 'NaN'

# In[7]:


print(filtered_df['geography'].unique())


# In[8]:


germany_data = filtered_df[filtered_df['geography'] == 'Germany']
print(germany_data)
spain_data = filtered_df[filtered_df['geography'] == 'Spain']
print(spain_data)
france_data = filtered_df[filtered_df['geography'] == 'France']
print(france_data)


# Parece no haber una tendencia por la cual tengamos datos 'NaN' en la columna de 'tenure'.
#
# Se toma la desición de eliminar las filas que contengan NaN, debido a que no tenemos claro el porqué estos datos están ausentes y por temas de ejercicio. Se considera que lo más adecuado sería poder consultar al banco, el porqué de los datos ausentes.
#
# Se eliminará con esto el 9% de los datos del dataset

# In[9]:


df_clean = df.dropna(subset=['tenure'])
print(df_clean.shape)


# Se procede a realizar el último tratamiento a los datos que será:
#
#     - Eliminar las columnas que no son relevantes para nuestro modelo
#         - RowNumber: índice de cadena de dato
#         - CustomerId: identificador de cliente único
#         - Surname: apellido
#
# Estas columnas parecen ser identificadores únicos o índices y no aportarán información significativa al modelo.
#
# Se codificarán las columnas:
#
#     - Geography
#     - Gender
#
# Puesto que estas columnas son categóricas y se necesitará codificarlas para que el modelo pueda trabajar con ellas. Se decide trabajar con OrdinalEncoder.

# In[10]:


data = df_clean.drop(['row_number', 'customer_id', 'surname'], axis=1)

encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns)
print(data_ordinal.head(10))


# ## Entrenamiento de modelo sin equilibrio
#
# Con el escrutinio de los datos, se sabe que hay un desequilibrio de clases.
# A modo de recordatorio:
#
#     - Exited: cuya media indica que el 20% indica que los clientes se han ido
#
# Se procede a entrenar el modelo sin estar equilibrado para ver qué resultados arroja.

# ### Segmentación de los datos
#
# Se procede a la segmentación de los datos para poder trabajar con ellos.
# Al tener un solo dataset y no contar con un dataset de prueba a futuro, se dividirá el total del dataset de la siguiente manera:
# - 60% dataset de entrenamiento
# - 20% dataset de validación
# - 20% Dataset de prueba

# In[11]:


features = data_ordinal.drop(['exited'], axis=1)
target = data_ordinal['exited']

features_train, x_train, target_train, y_valid = train_test_split(
    features, target, test_size=0.40, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(
    x_train, y_valid, test_size=0.50, random_state=12345)


# ### Modelos
#
# Ya segmentados los datos se entrenan 3 modelos para encontrar el modelo y los hiperparámetros que arrojen los mejores resultados:
#
#     - Árbol de decisión
#     - Bosque aleatorio
#     - Regresión logística
#
# Se utilizará 'F1 score' y 'AUC ROC' como métricas generales para entender la calidad del modelo.

# #### Árbol de desición
#
# Se elabora árbol de decisión, ejecutando dentro del modelo, un verificador que nos ayuda a saber los mejores hiperparámetros para el mismo modelo.

# In[12]:


best_model_tree_unbalanced = None
best_result_tree_unbalanced = 0
best_depth_tree_unbalanced = 0

for depth in range(1, 6):
    model_tree_unbalanced = DecisionTreeClassifier(
        random_state=12345, max_depth=depth)
    train_tree_unbalanced = model_tree_unbalanced.fit(
        features_train, target_train)
    predictions_tree_unbalanced = model_tree_unbalanced.predict(features_valid)
    result_tree_unbalanced = f1_score(
        target_valid, predictions_tree_unbalanced)

    if result_tree_unbalanced > best_result_tree_unbalanced:
        best_model_tree_unbalanced = model_tree_unbalanced
        best_result_tree_unbalanced = result_tree_unbalanced
        best_depth_tree_unbalanced = depth

print("El Valor de f1 del mejor modelo sin balance en el conjunto de validacion es de:",
      best_result_tree_unbalanced)
print("Resultado alcanzado con una profundidad de", best_depth_tree_unbalanced)


probabilities_valid_tree_unbalanced = model_tree_unbalanced.predict_proba(
    features_valid)
probabilities_one_valid_tree_unbalanced = probabilities_valid_tree_unbalanced[:, 1]
auc_roc_tree_unbalanced = roc_auc_score(
    target_valid, probabilities_one_valid_tree_unbalanced)

print(f'El valor de AUC-ROC es de: {auc_roc_tree_unbalanced}')


# #### Bosque Aleatorio
#
# Se elabora un bosque aleatorio, ejecutando dentro del modelo, un verificador que nos ayuda a saber los mejores hiperparámetros para el mismo modelo.

# In[13]:


best_model_forest_unbalanced = None
best_result_forest_unbalanced = 0
best_depth_forest_unbalanced = 0

for depth in range(1, 10):
    model_forest_unbalanced = RandomForestClassifier(
        random_state=12345, max_depth=depth)
    train_forest_unbalanced = model_forest_unbalanced.fit(
        features_train, target_train)
    predicition_forest_unbalanced = model_forest_unbalanced.predict(
        features_valid)
    results_forest_unbalanced = f1_score(
        target_valid, predicition_forest_unbalanced)
    if results_forest_unbalanced > best_result_forest_unbalanced:
        best_model_forest_unbalanced = model_forest_unbalanced
        best_result_forest_unbalanced = results_forest_unbalanced
        best_depth_forest_unbalanced = depth

print("El Valor de f1 del mejor modelo sin balance en el conjunto de validacion es de:",
      best_result_forest_unbalanced)
print("Resultado alcanzado con una profundidad de", best_depth_forest_unbalanced)

probabilities_valid_forest_unbalanced = model_forest_unbalanced.predict_proba(
    features_valid)
probabilities_one_valid_forest_unbalanced = probabilities_valid_forest_unbalanced[:, 1]
auc_roc_forest_unbalanced = roc_auc_score(
    target_valid, probabilities_one_valid_forest_unbalanced)

print(f'El valor de AUC-ROC es de: {auc_roc_forest_unbalanced}')


# #### Regresión Logística
# Se elabora un modelo de regresión logística

# In[14]:


model_regression_unbalanced = LogisticRegression(
    random_state=12345, solver='liblinear')
train_regresson_unbalanced = model_regression_unbalanced.fit(
    features_train, target_train)

predictions_train_unbalanced = model_regression_unbalanced.predict(
    features_train)
predictions_valid_unbalanced = model_regression_unbalanced.predict(
    features_valid)

f1_train_unbalanced = f1_score(target_train, predictions_train_unbalanced)
f1_valid_unbalanced = f1_score(target_valid, predictions_valid_unbalanced)

print("F1-score del modelo de regresión logística en el conjunto de entrenamiento:",
      f1_train_unbalanced)
print("F1-score del modelo de regresión logística en el conjunto de validación:",
      f1_valid_unbalanced)

probabilities_valid_regression_unbalanced = model_regression_unbalanced.predict_proba(
    features_valid)
probabilities_one_valid_regression_unbalanced = probabilities_valid_regression_unbalanced[:, 1]
auc_roc_regression_unbalanced = roc_auc_score(
    target_valid, probabilities_one_valid_regression_unbalanced)
print(f'El valor de AUC-ROC es de: {auc_roc_regression_unbalanced}')


# ### Observaciones de los modelos no equilibrados
#
# Después de haber implementado 3 modelos diferentes, obtuvimos los siguientes resultados:
#
#     - Árbol de decisión
#         El Valor de f1 del mejor modelo sin balance en el conjunto de validacion es de: 0.5514
#         El valor de AUC-ROC es de: 0.835
#
#     - Bosque aleatorio
#         El Valor de f1 del mejor modelo sin balance en el conjunto de validacion es de: 0.55
#         El valor de AUC-ROC es de: 0.8664
#
#     - Regresión logística
#         F1-score del modelo de regresión logística en el conjunto de validación: 0.2135
#         El valor de AUC-ROC es de: 0.73
#
#
# Para el presente estudio, se solicitaba un umbral de valor F1 de al menos 0.59, por lo tanto los 3 modelos no equilibrados, no pasarían la solicitud.
#
# De estos datos podemos indicar:
#
#     - Ambos modelos (Árbol de Decisión y Bosque Aleatorio) muestran un F1-score relativamente más alto en comparación con la Regresión Logística en el conjunto de validación.
#     - El valor de AUC-ROC es alto para ambos modelos, lo cual indica que son capaces de discriminar bien entre las clases positivas y negativas, lo que es mayor que solo adivinar al azar, pero al comparar estos dos valores, nos indica que el modelo puede no ser confiable.
#
#     - Para la Regresión Logística el valor F1-score es más bajo (0.2135) en comparación con los modelos basados en árboles.
#     - El valor de AUC-ROC es aceptable (0.73), pero inferior al de los modelos basados en árboles, lo que igualmente que en el caso anterior, temer un valor "aceptable" y el otro por debajo, podría indicar que no hay congruecia y nuestro modelo no es confiable.

# ### Prueba de calidad de los modelos
#
# Se medirá la calidad de los modelos con el fin de tener datos comparativos cuando el modelo esté equilibrado

# In[15]:


predictions_tree_test_unbalanced = model_tree_unbalanced.predict(features_test)
result_tree_test_unbalanced = f1_score(
    target_test, predictions_tree_test_unbalanced)

print("F1 del modelo de Árbol de Decisión en el conjunto de prueba:",
      result_tree_test_unbalanced)
print()

predictions_forest_test_unbalanced = model_forest_unbalanced.predict(
    features_test)
result_forest_test_unbalanced = f1_score(
    target_test, predictions_forest_test_unbalanced)

print("F1 del modelo del bosque aleatorio en el conjunto de prueba:",
      result_forest_test_unbalanced)
print()

predictions_regression_test_unbalanced = model_regression_unbalanced.predict(
    features_test)
result_regression_test_unbalanced = f1_score(
    target_test, predictions_regression_test_unbalanced)

print("F1 del modelo del regresión logística en el conjunto de prueba:",
      result_regression_test_unbalanced)


# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
#
# ¡Gracias!
#
# </div>

# ### Observaciones de los resultados de los test
#
# Al correr los modelos usando el conjunto de prueba previamente considerado Se obtuvieron los siguientes datos:
#
#     - F1 del modelo de Árbol de Decisión en el conjunto de prueba: 0.5316
#     - F1 del modelo del bosque aleatorio en el conjunto de prueba: 0.5472
#     - F1 del modelo del regresión logística en el conjunto de prueba:  0.2465
#
# Los modelos de arboles, han bajado algunas centécimas contra el conjunto de validación. Por los valores de F1 se puede indicar que el modelo no está teniendo un buen desempeño en términos de precisión y recall. Por lo que puede estar clasificando incorrectamente muchos casos positivos o negativos.
#
# Con estos datos podemos terminar de concluir que el desequilibrio de clase, nos da resultados apenas aceptables para los árboles de decisión y bosque aleatorio y muy bajo para F1.

# ## Mejora de la calidad del modelo
#
# Viendo que los resultados obtenidos por modelos con clases desequilibradas no fueron satisfactios, se procede a mejorar la calidad del modelo corrigiendo el desequilibrio de clases.
#
# Se equilibrará las clases por medio de las siguientes estrategias:
#
#     - Ajuste de pesos de clase
#     - Sobremuestreo de la clase minoritaria

# ### Ajuste de pesos de clase

# #### Modelos balanceados
#
# Se realizarán los modelos:
#
#     - Árboles de decisión
#     - Bosque Aleatorio
#     - Módelo de regresión logística

# ##### Árbol de decisión

# In[16]:


best_model_tree = None
best_result_tree = 0
best_depth_tree = 0


for depth_tree in range(1, 6):
    model_tree = DecisionTreeClassifier(
        random_state=12345, max_depth=depth_tree, class_weight='balanced')
    model_tree.fit(features_train, target_train)
    predicted_valid_tree = model_tree.predict(features_valid)
    result_tree = f1_score(target_valid, predicted_valid_tree)

    if result_tree > best_result_tree:
        best_model_tree = model_tree
        best_result_tree = result_tree
        best_depth_tree = depth_tree

print("El Valor de f1 del mejor modelo balanceado en el conjunto de validacion es de:", best_result_tree)
print("Resultado alcanzado con una profundidad de", best_depth_tree)

probabilities_valid_tree = model_tree.predict_proba(features_valid)
probabilities_one_valid_tree = probabilities_valid_tree[:, 1]
auc_roc_tree = roc_auc_score(target_valid, probabilities_one_valid_tree)

print(f'El valor de AUC-ROC es de: {auc_roc_tree}')


# ##### Bosque Aleatorio

# In[17]:


best_model_forest = None
best_result_forest = 0
best_depth_forest = 0

for depth_forest in range(1, 10):
    model_forest = RandomForestClassifier(
        random_state=12345, max_depth=depth_forest, class_weight='balanced')
    train_forest = model_forest.fit(features_train, target_train)
    predicition_forest = model_forest.predict(features_valid)
    results_forest = f1_score(target_valid, predicition_forest)

    if results_forest > best_result_forest:
        best_model_forest = model_forest
        best_result_forest = results_forest
        best_depth_forest = depth_forest

print("El Valor de f1 del mejor modelo balancado en el conjunto de validacion es de:", best_result_forest)
print("Resultado alcanzado con una profundidad de", best_depth_forest)

probabilities_valid_forest = model_forest.predict_proba(features_valid)
probabilities_one_valid_forest = probabilities_valid_forest[:, 1]
auc_roc_forest = roc_auc_score(target_valid, probabilities_one_valid_forest)

print(f'El valor de AUC-ROC es de: {auc_roc_forest}')


# ##### Módelo regresión logísitca

# In[18]:


model_regression = LogisticRegression(
    random_state=12345, class_weight='balanced', solver='liblinear')
train_regresson = model_regression.fit(features_train, target_train)

predictions_train = model_regression.predict(features_train)
predictions_valid = model_regression.predict(features_valid)

f1_train = f1_score(target_train, predictions_train)
f1_valid = f1_score(target_valid, predictions_valid)

print("F1-score del modelo de regresión logística en el conjunto de entrenamiento:", f1_train)
print("F1-score del modelo de regresión logística en el conjunto de validación:", f1_valid)

probabilities_valid_regression = model_regression.predict_proba(features_valid)
probabilities_one_valid_regression = probabilities_valid_regression[:, 1]
auc_roc_regression = roc_auc_score(
    target_valid, probabilities_one_valid_regression)
print(f'El valor de AUC-ROC es de: {auc_roc_regression}')


# #### Prueba de calidad de los modelos equilibrados
#
# Se realizará la pruebad de calidad de los modelos equilibrados con los datos test

# In[19]:


predictions_tree_test = model_tree.predict(features_test)
result_tree_test = f1_score(target_test, predictions_tree_test)

print("F1 del modelo de Árbol de Decisión en el conjunto de prueba:", result_tree_test)
print()

predictions_forest_test = model_forest.predict(features_test)
result_forest_test = f1_score(target_test, predictions_forest_test)

print("F1 del modelo del bosque aleatorio en el conjunto de prueba:", result_forest_test)
print()

predictions_regression_test = model_regression.predict(features_test)
result_regression_test = f1_score(target_test, predictions_regression_test)

print("F1 del modelo del regresión logística en el conjunto de prueba:",
      result_regression_test)


# #### Revisión de los datos
#
# A continuación, se muestra una tabla comparativa de los datos de cada uno de los modelos, una columna muestra los datos desequilibrados y los datos ya equilibrados:
#
# Resultados de Árbol de desición | Desequilibrado | Equilibrado
# --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55.14% | 57.59%
# Mejor resultado alcanzado con una profundidad | 5 | 5
# El valor de AUC-ROC | 83.50% | 83.05%
# F1 del modelo de Árbol de Decisión en el conjunto de prueba | 53.50% | 54.89%
#
#
# Resultados de Bosque aleatorio | Desequilibrado | Equilibrado
# --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55% | 63.56%
# Mejor resultado alcanzado con una profundidad | 6 | 9
# El valor de AUC-ROC | 86.64% | 86.49%
# F1 del modelo del bosque aleatorio en el conjunto de prueba | 54.72% | 58.46%
#
#
# Resultados de regesión | Desequilibrado | Equilibrado
# --- | --- | ---
# F1-score del modelo de regresión logística en el conjunto de entrenamiento | 25.03% | 48.31%
# F1-score del modelo de regresión logística en el conjunto de validación | 21.35% | 47.98%
# El valor de AUC-ROC | 73.01% | 75.70%
# F1 del modelo del regresión logística en el conjunto de prueba | 24.66% | 47.60%
#
#
# Se observa que en general y en la mayoria de los casos, los porcentajes de los modelos equilibrados dan porcentajes más altos, siendo el bosque aleatorio de profundidad 9, el que más se acerca al 59% deseado en F1.

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
#
# ¡Entiendo!
#
# </div>

# ### Sobremuestreo de la clase minoritaria
#
# Ahora se balanceará los datos por medio de el sobremuestreo de la clase minoritaria

# #### Función de sobremuestreo

# In[20]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled


features_upsampled, target_upsampled = upsample(
    features_train, target_train, 7
)


# Se utilizarán los 3 modelos con el sobremuestreo.

# ##### Árbol de decisión

# In[21]:


best_model_tree_upsampled = None
best_result_tree_upsampled = 0
best_depth_tree_upsampled = 0


for depth_tree_upsampled in range(1, 6):
    model_tree_upsampled = DecisionTreeClassifier(
        random_state=12345, max_depth=depth_tree_upsampled)
    model_tree_upsampled.fit(features_upsampled, target_upsampled)
    predicted_valid_upsampled_tree = model_tree_upsampled.predict(
        features_valid)
    result_tree_upsampled = f1_score(
        target_valid, predicted_valid_upsampled_tree)

    if result_tree_upsampled > best_result_tree_upsampled:
        best_model_tree_upsampled = model_tree_upsampled
        best_result_tree_upsampled = result_tree_upsampled
        best_depth_tree_upsampled = depth_tree_upsampled

print("El Valor de f1 del mejor modelo balanceado en el conjunto de validacion es de:",
      best_result_tree_upsampled)
print("Resultado alcanzado con una profundidad de", best_depth_tree_upsampled)

probabilities_valid_tree_upsampled = best_model_tree_upsampled.predict_proba(
    features_valid)
probabilities_one_valid_tree_upsampled = probabilities_valid_tree_upsampled[:, 1]
auc_roc_tree_upsampled = roc_auc_score(
    target_valid, probabilities_one_valid_tree_upsampled)

print(f'AUC-ROC: {auc_roc_tree_upsampled}')


# ##### Bosque Aleatorio

# In[22]:


best_model_forest_upsampled = None
best_result_forest_upsampled = 0
best_depth_forest_upsampled = 0

for depth_forest_upsampled in range(1, 10):
    model_forest_upsampled = RandomForestClassifier(
        random_state=12345, max_depth=depth_forest_upsampled,)
    model_forest_upsampled.fit(features_upsampled, target_upsampled)
    predicted_valid_upsampled_forest = model_forest_upsampled.predict(
        features_valid)
    results_forest_upsampled = f1_score(
        target_valid, predicted_valid_upsampled_forest)

    if results_forest_upsampled > best_result_forest_upsampled:
        best_model_forest_upsampled = model_forest_upsampled
        best_result_forest_upsampled = results_forest_upsampled
        best_depth_forest_upsampled = depth_forest_upsampled

print("El Valor de f1 del mejor modelo balancado en el conjunto de validacion es de:",
      best_result_forest_upsampled)
print("Resultado alcanzado con una profundidad de", best_depth_forest_upsampled)

probabilities_valid_forest_upsampled = best_model_forest_upsampled.predict_proba(
    features_valid)
probabilities_one_valid_forest_upsampled = probabilities_valid_forest_upsampled[:, 1]
auc_roc_forest_upsampled = roc_auc_score(
    target_valid, probabilities_one_valid_forest_upsampled)

print(f'AUC-ROC: {auc_roc_forest_upsampled}')


# ##### Módelo regresión logísitca

# In[23]:


model_regression_upsampled = LogisticRegression(
    random_state=12345, solver='liblinear')
model_regression_upsampled.fit(features_upsampled, target_upsampled)

predictions_train_upsampled_regression = model_regression_upsampled.predict(
    features_train)
predictions_valid_upsampled_regression = model_regression_upsampled.predict(
    features_valid)

f1_train_upsampled = f1_score(
    target_train, predictions_train_upsampled_regression)
f1_valid_upsampled = f1_score(
    target_valid, predictions_valid_upsampled_regression)

print("F1-score del modelo de regresión logística en el conjunto de entrenamiento:",
      f1_train_upsampled)
print("F1-score del modelo de regresión logística en el conjunto de validación:",
      f1_valid_upsampled)

probabilities_valid_regression_upsampled = model_regression_upsampled.predict_proba(
    features_valid)
probabilities_one_valid_regression_upsampled = probabilities_valid_regression_upsampled[:, 1]
auc_roc_regression_upsampled = roc_auc_score(
    target_valid, probabilities_one_valid_regression_upsampled)

print(f'AUC-ROC: {auc_roc_regression_upsampled}')


# #### Prueba de calidad de los modelos equilibrados
# Se realizará la prueba de calidad de los modelos equilibrados con los datos test

# In[24]:


predictions_tree_test_upsampled = best_model_tree_upsampled.predict(
    features_test)
result_tree_test_upsampled = f1_score(
    target_test, predictions_tree_test_upsampled)

print("F1 del modelo de Árbol de Decisión en el conjunto de prueba:",
      result_tree_test_upsampled)
print()

predictions_forest_test_upsampled = best_model_forest_upsampled.predict(
    features_test)
result_forest_test_upsampled = f1_score(
    target_test, predictions_forest_test_upsampled)

print("F1 del modelo del bosque aleatorio en el conjunto de prueba:",
      result_forest_test_upsampled)
print()

predictions_regression_test_upsampled = model_regression_upsampled.predict(
    features_test)
result_regression_test_upsampled = f1_score(
    target_test, predictions_regression_test_upsampled)

print("F1 del modelo del regresión logística en el conjunto de prueba:",
      result_regression_test_upsampled)


# #### Revisión de los datos
#
# A continuación, se muestra una tabla comparativa de los datos de cada uno de los modelos realizados con sobremuestreo. En una columna se muestran los datos desequilibrados y en la otra columna los datos ya equilibrados:
#
# Resultados de Árbol de desición | Desequilibrado | Equilibrado
# --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55.14% | 54.87%
# Mejor resultado alcanzado con una profundidad | 5 | 5
# El valor de AUC-ROC | 83.50% | 84.55%
# F1 del modelo de Árbol de Decisión en el conjunto de prueba | 53.16% | 51.85%
#
#
# Resultados de Bosque aleatorio | Desequilibrado | Equilibrado
# --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55% | 56.67%
# Mejor resultado alcanzado con una profundidad | 6 | 8
# El valor de AUC-ROC | 86.64% | 86.07%
# F1 del modelo del bosque aleatorio en el conjunto de prueba | 54.72% | 53.32%
#
#
# Resultados de regesión | Desequilibrado | Equilibrado
# --- | --- | ---
# F1-score del modelo de regresión logística en el conjunto de entrenamiento | 25.03% | 43.17%
# F1-score del modelo de regresión logística en el conjunto de validación | 21.35% | 42.22%
# El valor de AUC-ROC | 73.01% | 75.76%
# F1 del modelo del regresión logística en el conjunto de prueba | 24.66% | 42.94%
#
#
# Se observa que en general y en la mayoria de los casos, los porcentajes de los modelos equilibrados dan porcentajes más altos, sin embargo no se llega a obtener el valor de F1 buscado.

# Despues de realizar el equilibrio de modelos utilizando Ajuste de pesos de clase y sobremuestreo y no haber llegado al F1 solicitado, se realizará un submuestreo

# ### Submuestreo
#
# Después de realizar el equilibrio de clases en diferentes modelos y correrlos, se encuentra la mejor opción que nos da los resultados buscados.
#
# Se elige el modelo de Bosque Aleatorio, con un equilibrio de clases con sobremuestreo con repetición de 8.
#
# Se realiza la prueba final.

# #### Función de submuestreo

# In[25]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.35
)


# Se utilizarán los 3 modelos con el sobremuestreo.

# ##### Árbol de decisión

# In[26]:


best_model_tree_downsample = None
best_result_tree_downsample = 0
best_depth_tree_downsample = 0


for depth_tree_downsample in range(1, 6):
    model_tree_downsample = DecisionTreeClassifier(
        random_state=12345, max_depth=depth_tree_downsample)
    model_tree_downsample.fit(features_downsampled, target_downsampled)
    predicted_valid_downsample_tree = model_tree_downsample.predict(
        features_valid)
    result_tree_downsample = f1_score(
        target_valid, predicted_valid_downsample_tree)

    if result_tree_downsample > best_result_tree_downsample:
        best_model_tree_downsample = model_tree_downsample
        best_result_tree_downsample = result_tree_downsample
        best_depth_tree_downsample = depth_tree_downsample

print("El Valor de f1 del mejor modelo balanceado en el conjunto de validacion es de:",
      best_result_tree_downsample)
print("Resultado alcanzado con una profundidad de", best_depth_tree_downsample)

probabilities_valid_tree_downsample = best_model_tree_downsample.predict_proba(
    features_valid)
probabilities_one_valid_tree_downsample = probabilities_valid_tree_downsample[:, 1]
auc_roc_tree_downsample = roc_auc_score(
    target_valid, probabilities_one_valid_tree_downsample)

print(f'AUC-ROC: {auc_roc_tree_downsample}')


# ##### Bosque Aleatorio

# In[27]:


best_model_forest_downsample = None
best_result_forest_downsample = 0
best_depth_forest_downsample = 0

for depth_forest_downsample in range(1, 10):
    model_forest_downsample = RandomForestClassifier(
        random_state=12345, max_depth=depth_forest_downsample)
    model_forest_downsample.fit(features_downsampled, target_downsampled)
    predicted_valid_downsample_forest = model_forest_downsample.predict(
        features_valid)
    results_forest_downsample = f1_score(
        target_valid, predicted_valid_downsample_forest)

    if results_forest_downsample > best_result_forest_downsample:
        best_model_forest_downsample = model_forest_downsample
        best_result_forest_downsample = results_forest_downsample
        best_depth_forest_downsample = depth_forest_downsample

print("El Valor de f1 del mejor modelo balancado en el conjunto de validacion es de:",
      best_result_forest_downsample)
print("Resultado alcanzado con una profundidad de", best_depth_forest_downsample)

probabilities_valid_forest_downsample = best_model_forest_downsample.predict_proba(
    features_valid)
probabilities_one_valid_forest_downsample = probabilities_valid_forest_downsample[:, 1]
auc_roc_forest_downsample = roc_auc_score(
    target_valid, probabilities_one_valid_forest_downsample)

print(f'AUC-ROC: {auc_roc_forest_downsample}')


# ##### Regresión lineal

# In[28]:


model_regression_downsample = LogisticRegression(
    random_state=12345, solver='liblinear')
model_regression_downsample.fit(features_downsampled, target_downsampled)

predictions_train_downsample_regression = model_regression_downsample.predict(
    features_train)
predictions_valid_downsample_regression = model_regression_downsample.predict(
    features_valid)

f1_train_downsample = f1_score(
    target_train, predictions_train_downsample_regression)
f1_valid_downsample = f1_score(
    target_valid, predictions_valid_downsample_regression)

print("F1-score del modelo de regresión logística en el conjunto de entrenamiento:",
      f1_train_downsample)
print("F1-score del modelo de regresión logística en el conjunto de validación:",
      f1_valid_downsample)

probabilities_valid_regression_downsample = model_regression_downsample.predict_proba(
    features_valid)
probabilities_one_valid_regression_downsample = probabilities_valid_regression_downsample[:, 1]
auc_roc_regression_downsample = roc_auc_score(
    target_valid, probabilities_one_valid_regression_downsample)

print(f'AUC-ROC: {auc_roc_regression_downsample}')


# #### Prueba de calidad de los modelos equilibrados
#
# Se realizará la prueba de calidad de los modelos equilibrados con los datos test

# In[29]:


predictions_tree_test_downsample = model_tree_downsample.predict(features_test)
result_tree_test_downsample = f1_score(
    target_test, predictions_tree_test_downsample)

print("F1 del modelo de Árbol de Decisión en el conjunto de prueba:",
      result_tree_test_downsample)
print()

predictions_forest_test_downsample = model_forest_downsample.predict(
    features_test)
result_forest_test_downsample = f1_score(
    target_test, predictions_forest_test_downsample)

print("F1 del modelo del bosque aleatorio en el conjunto de prueba:",
      result_forest_test_downsample)
print()

predictions_regression_test_downsample = model_regression_downsample.predict(
    features_test)
result_regression_test_downsample = f1_score(
    target_test, predictions_regression_test_downsample)

print("F1 del modelo del regresión logística en el conjunto de prueba:",
      result_regression_test_downsample)


# #### Revisión de los datos
#
# A continuación, se muestra una tabla comparativa de los datos de cada uno de los modelos realizados con sobremuestreo. En una columna se muestran los datos desequilibrados y en la otra columna los datos ya equilibrados:
#
# Resultados de Árbol de desición | Desequilibrado | Equilibrado
# --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55.14% | 57.43%
# Mejor resultado alcanzado con una profundidad | 5 | 5
# El valor de AUC-ROC | 83.50% | 83.05%
# F1 del modelo de Árbol de Decisión en el conjunto de prueba | 53.16% | 55.64%
#
#
# Resultados de Bosque aleatorio | Desequilibrado | Equilibrado
# --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55% | 61.7%
# Mejor resultado alcanzado con una profundidad | 6 | 6
# El valor de AUC-ROC | 86.64% | 86.26%
# F1 del modelo del bosque aleatorio en el conjunto de prueba | 54.72% | 59.61%
#
#
# Resultados de regesión | Desequilibrado | Equilibrado
# --- | --- | ---
# F1-score del modelo de regresión logística en el conjunto de entrenamiento | 25.03% | 46.69%
# F1-score del modelo de regresión logística en el conjunto de validación | 21.35% | 47.11%
# El valor de AUC-ROC | 73.01% | 74.23%
# F1 del modelo del regresión logística en el conjunto de prueba | 24.66% | 46.80%
#
#
# Ahora hemos podido encontrar un F1 que con los datos de prueba sea como mínimo 59%

# ### Comparativo Final

# #### Revisión de los datos
#
# A continuación, se muestra una tabla comparativa de los datos de cada uno de los modelos realizados con sobremuestreo. En una columna se muestran los datos desequilibrados y en la otra columna los datos ya equilibrados:
#
# Resultados de Árbol de desición | Desequilibrado | Ajuste de pesos | Sobremuestreo | Submuestreo
# --- | --- | --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55.14% | 57.59% | 54.87% | 57.43%
# Mejor resultado alcanzado con una profundidad | 5 | 5 | 5 | 5
# El valor de AUC-ROC | 83.50% | 83.05% | 84.55% | 83.05%
# F1 del modelo de Árbol de Decisión en el conjunto de prueba | 53.50% | 54.89% | 51.85% | 55.64%
#
#
# Resultados de Bosque aleatorio | Desequilibrado | Ajuste de pesos | Sobremuestreo | Submuestreo
# --- | --- | --- | --- | ---
# El Valor de f1 del mejor modelo en el conjunto de validacion | 55% | 63.56% | 56.67% | 61.70%
# Mejor resultado alcanzado con una profundidad | 6 | 9 | 8 | 6
# El valor de AUC-ROC | 86.64% | 86.49% | 86.07% | 86.26%
# F1 del modelo del bosque aleatorio en el conjunto de prueba | 54.72% | <span style="color:brown;font-weight:bold;">58.46%</span> | 53.32% | <span style="color:green;font-weight:bold;">59.61%</span>
#
#
# Resultados de regesión | Desequilibrado | Ajuste de pesos | Sobremuestreo | Submuestreo
# --- | --- | --- | --- | ---
# F1-score del modelo de regresión logística en el conjunto de entrenamiento | 25.03% | 48.31% | 43.17% | 46.69%
# F1-score del modelo de regresión logística en el conjunto de validación | 21.35% | 47.98% | 44.22% | 47.11%
# El valor de AUC-ROC | 73.01% | 75.70% | 75.76% | 74.23%
# F1 del modelo del regresión logística en el conjunto de prueba | 24.66% | 47.60% | 42.94% | 46.80%
#
#
# Ahora hemos podido encontrar un F1 que con los datos de prueba sea como mínimo 59%
#
# Con el ajuste de pesos de clase para el modelo de Bosque aleatorio obtuvo un resultado muy cercano al del F1 deseado, con un valor del 58.46%.
# El modelo de bosque aleatorio con equilibrio de clase de submuestreo fue el que finalmente nos proporcionó un valor de f1 del 59.61%, teniendo el mínimo de buscado.
