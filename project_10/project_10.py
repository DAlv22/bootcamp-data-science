# # Proyecto Zyfra
#
# ## Introducción
# Zyfra es una compañia que proporciona soluciones eficientes para la industria pesada.
# "Mejoramos la eficacia y la seguridad de las industrias minera, petrolera y del gas, química y de ingeniería."
#
# ##  Objetivo del proyecto
# Zyfra busca un prototipo de un modelo de Machine Learning.
#
# ###  Objetivo específico
# Proporcionar un modelo de Machine Learning que prediga la cantidad de oro extraido del mineral de oro.
#
# ##  Inicio
#
# ### Librerias
# La compañia proporciona los datos en bruto que fueron descargados del almacén de datos
#
#     - Se descargan las librerías necesarias para el tratamiento necesarios de los datos

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer


# ### Revisón de datos
#
#     Se importan los datos proporcionados

# In[2]:


gold_recovery_train = pd.read_csv('/datasets/gold_recovery_train.csv')
gold_recovery_test = pd.read_csv('/datasets/gold_recovery_test.csv')
gold_recovery_full = pd.read_csv('/datasets/gold_recovery_full.csv')


# Se revisan las características generales de cada uno de los archivos proporcionados

# In[3]:


print(
    f'El dataset train tiene: \n - {gold_recovery_train.duplicated().sum()} filas duplicadas \n - Filas y columnas: {(gold_recovery_train.shape)}')
train_null = gold_recovery_train.isnull().sum().sum(
) / (gold_recovery_train.shape[0] * gold_recovery_train.shape[1])*100
print(f' - Hay un {round(train_null,2)}% de datos ausentes \n \nLas características generales son las siguientes:\n ')


display(gold_recovery_train)
print()
print(gold_recovery_train.info())
gold_recovery_train.describe()


# In[4]:


print(
    f'El dataset test tiene: \n - {gold_recovery_test.duplicated().sum()} filas duplicadas \n - Filas y columnas: {(gold_recovery_test.shape)}')
test_null = gold_recovery_test.isnull().sum().sum(
) / (gold_recovery_test.shape[0] * gold_recovery_test.shape[1])*100
print(f' - Hay un {round(test_null,2)}% de datos ausentes \n \nLas características generales son las siguientes:\n ')

display(gold_recovery_test)
print()
print(gold_recovery_test.info())
gold_recovery_test.describe()


# In[5]:


print(
    f'El dataset full tiene: \n - {gold_recovery_full.duplicated().sum()} filas duplicadas \n - Filas y columnas: {(gold_recovery_full.shape)}')
full_null = gold_recovery_full.isnull().sum().sum(
) / (gold_recovery_full.shape[0] * gold_recovery_full.shape[1]) * 100
print(f' - Hay un {round(full_null,2)}% de datos ausentes \n \nLas características generales son las siguientes:\n ')

display(gold_recovery_full)
print()
print(gold_recovery_full.info())
print()
gold_recovery_full.describe()


# ### Descripción de datos
#
# Los datos representados en los dataset son los siguientes:
#
# Proceso tecnológico
#
#     •	Rougher feed — materia prima
#     •	Rougher additions (o adiciones de reactivos) - reactivos de flotación: xantato, sulfato, depresante
#         o	Xantato — promotor o activador de la flotación
#         o	Sulfato — sulfuro de sodio para este proceso en particular
#         o	Depresante — silicato de sodio
#     •	Rougher process — flotación
#     •	Rougher tails — residuos del producto
#     •	Float banks — instalación de flotación
#     •	Cleaner process — purificación
#     •	Rougher Au — concentrado de oro rougher
#     •	Final Au — concentrado de oro final
#
# Parámetros de las etapas
#
#     •	air amount — volumen de aire
#     •	fluid levels
#     •	feed size — tamaño de las partículas de la alimentación
#     •	feed rate
#
#
# A CONSIDERAR:
#
# La columna __"date"__ representa la __fecha y hora__ de adquisición de los datos.
#
# El resto de las columnas tiene un nombre acorde al siguiente formato: [stage].[parameter_type].[parameter_name]
#
# DONDE...
#
#     - Valores posibles para [stage]:
#         •	rougher — flotación
#         •	primary_cleaner — purificación primaria
#         •	secondary_cleaner — purificación secundaria
#         •	final — características finales
#
#     - Valores posibles para [parameter_type]:
#         •	input — parámetros de la materia prima
#         •	output — parámetros del producto
#         •	state — parámetros que caracterizan el estado actual de la etapa
#         •	calculation — características de cálculo (Involucra características derivadas o calculadas en lugar de medidas directas)
#


# ### Escrutineo de datos
#
# Algunos parámetros no están disponibles porque fueron medidos o calculados mucho más tarde. Eso podría llevar a que algunas de las características que están presentes en el conjunto de entrenamiento pueden estar ausentes en el de prueba.
#
# Los parámetros cercanos en el tiempo indicado en date suelen ser similares.
#
# El dataset fuente contiene los conjuntos de entrenamiento y prueba con todas las características.
#
# Los datos de los datasets se resumen de la siguiente forma:
#
#  Data | gold_recovery_train | gold_recovery_test | gold_recovery_full
# --- | --- | --- | ---
# Filas duplicadas | 0 | 0 | 0
# % datos ausentes | 2.07% | 0.76% | 1.85%
# Filas | 16860 | 5856 | 22716
# Columnas | 87 | 53 | 87
# D. float64 | 86 | 52 | 86
# D. object | 1 | 1 | 1
#
#
# Esta es la visualización general inicial de los datos

# ### Recuperación del concentrado rougher
#
# Se comprueba que el calculo de la recuperación de la característica __rougher.output.recovery__ es correcta, esto mediante el conjunto de entrenamiento. La fórmula es la siguiente:
#
# $Recuperación = \frac{C \cdot (F - T)}{F \cdot (C - T)} \times 100\%$
#
# Donde:
#
# Recuperación del concentrado rougher (C):
#
#     - Después de la flotación: rougher.output.concentrate_au
#     - Después de la purificación: final.output.concentrate_ag
#
#
# Proporción de oro en la alimentación (F):
#
#     - Antes de la flotación: rougher.input.feed_au
#     - En el concentrado justo después de la flotación: rougher.output.concentrate_au
#
#
# Proporción de oro en las colas rougher (T):
#
#     - Después de la flotación: rougher.output.tail_au
#     - Después de la purificación: final.output.tail_au
#
#
# Para trabajar con los datos, se decide eliminar las filas con datos nulos y posteriormente realizar la función

# In[6]:


gold_recovery_train_clean = gold_recovery_train.dropna()


# In[7]:


def recovery_calc(row):
    numerator = row['rougher.output.concentrate_au'] * \
        (row['rougher.input.feed_au']-row['rougher.output.tail_au'])
    denominator = row['rougher.input.feed_au'] * \
        (row['rougher.output.concentrate_au']-row['rougher.output.tail_au'])
    recovery_rougher = numerator/denominator*100

    return recovery_rougher


gold_recovery_train_clean['recovery_rougher'] = gold_recovery_train_clean.apply(
    lambda x: recovery_calc(x), axis=1)
gold_recovery_train_clean.head()


# Se calcula el EAM

# In[8]:


eam = (gold_recovery_train_clean['recovery_rougher'] -
       gold_recovery_train_clean['rougher.output.recovery']).abs().mean()
print(eam)


# El valor del Error Absoluto Medio es muy cercano a cero, un EAM tan pequeño sugiere que las predicciones del modelo son casi idénticas a los valores reales.

# ### Carácteristicas no disponibles
#
# Se procede a revisar los valores ausentes de los diferentes parámetros del dataset.

# In[9]:


gold_recovery_full_columns_names = gold_recovery_full.columns.tolist()
gold_recovery_test_columns_names = gold_recovery_test.columns.tolist()

missing_columns = list(set(gold_recovery_full_columns_names) -
                       set(gold_recovery_test_columns_names))

print(
    f'Número de columnas faltantes en el dataset de prueba: {len(missing_columns)} columnas')
print(f'\nNombres de las columnas ausentes:')
sorted(missing_columns)


# Se observa que son 34 las columnas faltantes en el dataset de prueba.
#
# Los nombres de las columnas no disponibles en el conjunto de prueba son de tipo float y corresponden tanto a outputs como a calculos.
#
# Se menciona la precencia de diferentes elementos como "au" (oro), "pb" (plomo), "ag" (plata) en diferentes estapas del procesamiento, así como sustancias "sulfate" y "sol".
#

# ### Preprocesamiento de datos
#
# Se comienza el preprocesamiento de los datos.
#
# Recordando las caracteristicas de los datos, tenemos unicamente una columna tipo objetc, que es la de la fecha, el resto de las columnas son tipo float.
#
#  Data | gold_recovery_train | gold_recovery_test | gold_recovery_full
# --- | --- | --- | ---
# Filas duplicadas | 0 | 0 | 0
# % datos ausentes | 2.07% | 0.76% | 1.85%
# Filas | 16860 | 5856 | 22716
# Columnas | 87 | 53 | 87
# D. float64 | 86 | 52 | 86
# D. object | 1 | 1 | 1
#
# Debido a que no requeriremos hacer ningún tipo de cálculo con la columna de fechas se procede a borrarla.
# Se sabe que los parámetros cercanos en el tiempo suelen ser similares, por lo que procede a rellenar los datos ausentes con el datos inmediato anterior.

# In[10]:


gold_recovery_test = gold_recovery_test.drop(['date'], axis=1)
gold_recovery_train = gold_recovery_train.drop(['date'], axis=1)

print(gold_recovery_test.shape)
print(gold_recovery_train.shape)


# In[11]:


gold_recovery_train = gold_recovery_train.fillna(method='ffill', axis=0)
gold_recovery_test = gold_recovery_test.fillna(method='ffill', axis=0)

print(
    f'Los datos nulos en el dataset de entranmiento es de: {gold_recovery_train.isnull().sum().sum()} datos nulos')
print(
    f'Los datos nulos en el dataset de prueba es de: {gold_recovery_test.isnull().sum().sum()} datos nulos')


# Con esto se concluye el preprocesamiento de datos.

# ## Análisis de datos
#
# En el siguiente histograma, se muestra cómo cambia la concentración de metales (Au, Ag, Pb) en función de la etapa de purificación.

# ### Oro

# In[12]:


data_au = gold_recovery_train[['rougher.input.feed_au', 'rougher.output.concentrate_au',
                               'primary_cleaner.output.concentrate_au', 'final.output.concentrate_au']]

fig, ax = plt.subplots(figsize=(12, 8))

for column in data_au.columns:
    ax.hist(data_au[column], bins=20, alpha=0.6,
            label=column, histtype="stepfilled")

plt.title('Histogram of Gold Concentration per Stage')
plt.xlabel('Gold concentration')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
plt.grid(True)

plt.show()


# Acorde a lo demostrado, la concentración de __oro__ va incrementandose conforme se avanza en el proceso de purificación.

# ### Plata

# In[13]:


data_ag = gold_recovery_train[['rougher.input.feed_ag', 'rougher.output.concentrate_ag',
                               'primary_cleaner.output.concentrate_ag', 'final.output.concentrate_ag']]

fig, ax = plt.subplots(figsize=(12, 8))

for column in data_ag.columns:
    ax.hist(data_ag[column], bins=20, alpha=0.6,
            label=column, histtype="stepfilled")

plt.title('Histogram of Silver Concentration per Stage')
plt.xlabel('Silver concentration')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(True)

plt.show()


# La concentración de __Plata__ es mayor va en decrimento conforme se avanza en el proceso de purificación.

# ### Plomo

# In[14]:


data_pb = gold_recovery_train[['rougher.input.feed_pb', 'rougher.output.concentrate_pb',
                               'primary_cleaner.output.concentrate_pb', 'final.output.concentrate_pb']]

fig, ax = plt.subplots(figsize=(12, 8))

for column in data_pb.columns:
    ax.hist(data_pb[column], bins=20, alpha=0.6,
            label=column, histtype="stepfilled")

plt.title('Histogram of Silver Concentration per Stage')
plt.xlabel('Silver concentration')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(True)

plt.show()


# La concentración de __Plomo__ aumenta tras la salida del primer proceso. Posteriormente aumente muy poco con respecto a lo proceso anterior.

# ### Distribuciones del tamaño de las partículas de la alimentación

# Se comparan las distribuciones del tamaño de las partículas de la alimentación en el conjunto de entrenamiento y en el conjunto de prueba, pues si se tuvieran variaciones la evaluación del modelo no será correcta.

# In[15]:


plt.figure(figsize=(10, 6))

gold_recovery_train['rougher.input.feed_size'].hist(
    alpha=0.6, label='Training Set', bins=50)
gold_recovery_test['rougher.input.feed_size'].hist(
    alpha=0.6, label='Test Set', bins=50)

plt.title('Distribution of feed size in train and test sets')
plt.xlabel('Feed size')
plt.ylabel('Frequency')
plt.legend()

plt.xlim(0, 200)

plt.show()


# In[16]:


train_mean = gold_recovery_train['rougher.input.feed_size'].mean()
test_mean = gold_recovery_test['rougher.input.feed_size'].mean()

print(
    f'Media de rougher.input.feed_size: \n - Conjunto de entrenamiento: {train_mean:.2f} \n - Conjunto de prueba: {test_mean:.2f}')

train_median = gold_recovery_train['rougher.input.feed_size'].median()
test_median = gold_recovery_test['rougher.input.feed_size'].median()

print(
    f'\nMediana de rougher.input.feed_size: \n - Conjunto de entrenamiento: {train_median:.2f} \n - Conjunto de prueba: {test_median:.2f}')

train_std = gold_recovery_train['rougher.input.feed_size'].std()
test_std = gold_recovery_test['rougher.input.feed_size'].std()

print(
    f'\nDesviación estándar de rougher.input.feed_size: \n - Conjunto de entrenamiento: {train_std:.2f} \n - Conjunto de prueba: {test_std:.2f}')


# Las partículas de la alimentación en el conjunto de entrenamiento y en el conjunto de prueba no varían considerablemente, por lo cual los datos sirven para evaluar el modelo correctamente.


# ### Concentración total de substancias
#
# Se revisan las concentraciones totales de todas las sustancias en las diferentes etapas:
#
#     - materia prima
#     - concentrado rougher
#     - concentrado final
#
# Con el fin de visualizar si la distribución es normal o anormal para considerar conservar o eliminar esos valores.

# In[17]:


rougher_input = ['rougher.input.feed_ag', 'rougher.input.feed_pb',
                 'rougher.input.feed_sol', 'rougher.input.feed_au']
rougher_output = ['rougher.output.concentrate_ag', 'rougher.output.concentrate_pb',
                  'rougher.output.concentrate_sol', 'rougher.output.concentrate_au']
final_output = ['final.output.concentrate_ag', 'final.output.concentrate_pb',
                'final.output.concentrate_sol', 'final.output.concentrate_au']

rougher_input_sum = gold_recovery_train.loc[:, rougher_input].sum(axis=1)
rougher_output_sum = gold_recovery_train.loc[:, rougher_output].sum(axis=1)
final_output_sum = gold_recovery_train.loc[:, final_output].sum(axis=1)


# In[18]:


plt.figure(figsize=(12, 8))

plt.hist(rougher_input_sum, bins=50, alpha=0.6, label='Rougher input')
plt.hist(rougher_output_sum, bins=50, alpha=0.6, label='Rougher output')
plt.hist(final_output_sum, bins=50, alpha=0.6, label='Final output')

plt.title('Total concentrations of all substances')
plt.xlabel('Sum of Concentrations')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.show()


# En el histograma se observan valores en 0, estos valores no tendrían porqué existir, dado que siempre hay una entrada de alguna substancia, no debería dar 0. Se procede a filtrar las filas con datos > que 0 para el dataset de entrenamiento.

# In[19]:


condition = (
    (gold_recovery_train[rougher_input].sum(axis=1) > 0) &
    (gold_recovery_train[rougher_output].sum(axis=1) > 0) &
    (gold_recovery_train[final_output].sum(axis=1) > 0)
)

gold_recovery_train = gold_recovery_train[condition]
gold_recovery_train.info()


# ## Contrucción de Modelo de Machine Learning
#
# ### Datos
#
# Para entrenar al modelo se toman en cuenta para los "features" las columnas del conjunto de prueba.
#
# Para el apartado de "target" se toman en cuenta las columnas objetivo 'rougher.output.recovery' y 'final.output.recovery' del mismo conjunto de entrenamiento.

# In[20]:


features = gold_recovery_test.columns.values
features


# In[21]:


targets = ['rougher.output.recovery', 'final.output.recovery']
X_train = gold_recovery_train[features].reset_index(drop=True)
y_train = gold_recovery_train[targets].reset_index(drop=True)
y_train.columns = [0, 1]
y_train


# In[22]:


X_train


# ### Valor sMAPE
#
# Ya con "features" y "target" definido, se procede a realizar la función 'compute_smape' que calcula la métrica de sMAPE (Symmetric Mean Absolute Percentage Erro o Error medio absoluto porcentual simétrico) cuya fórmula es:
#
# $sMAPE = \frac{1}{n} \sum_{t=1}^{n} \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|)/2} \times 100\%$
#
# Y posterior a eso, se usará la función 'smape_ponderado' para calcular con dos conjuntos de datos que están divididos en dos etapas, "rougher" y "final".

# In[23]:


def compute_smape(y, y_pred):
    n = len(y)
    real = abs(y)
    pred = abs(y_pred)
    diff = abs(y - y_pred)
    smape = (1/n)*np.sum(diff / ((real + pred)/2))*100
    return smape


def smape_ponderado(y, y_pred):

    y_rougher = y.iloc[:, 0]
    y_pred_rougher = y_pred[:, 0]

    y_final = y.iloc[:, 1]
    y_pred_final = y_pred[:, 1]

    smape_rougher = compute_smape(y_rougher, y_pred_rougher)
    smape_final = compute_smape(y_final, y_pred_final)

    return (0.25*smape_rougher + 0.75*smape_final)


# In[24]:


smape_scorer = make_scorer(smape_ponderado, greater_is_better=False)


# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
#
# Entendido. Muchas gracias.
#
# </div>

# ### Bosque Aleatorio
#
# Se entrena un modelo de bosque aleatorio con validación cruzada

# In[26]:


best_result = 1000
best_depth = 0
best_model = None

for depth in range(1, 10):
    model = RandomForestRegressor(
        n_estimators=20, random_state=12345, max_depth=depth)
    model_scores = cross_val_score(
        model, X_train, y_train, scoring=smape_scorer, cv=5)

    model_scores_abs = np.abs(model_scores)
    mean_score = np.mean(model_scores_abs)

    print('Max depth =', depth, '-- sMAPE score:', model_scores)

    if mean_score < best_result:
        best_model = model
        best_result = mean_score
        best_depth = depth

print('\nBest max depth =', best_depth, '-- sMAPE score:', best_result)


# ### Regresión Lineal
#
# Se entrena un modelo de regresión lineal con validación cruzada

# In[27]:


lr_model = LinearRegression()
lr_scores = cross_val_score(
    lr_model, X_train, y_train, scoring=smape_scorer, cv=5)

lr_scores_abs = np.abs(lr_scores)
lr_final_score = np.mean(lr_scores_abs)

print('Puntajes sMAPE para cada iteración:', lr_scores)
print('Modelo de Regresión Lineal | sMAPE = {:.6f}'.format(lr_final_score))


# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
#
# Ya revisé lo indicado y entiendo la importancia de haber mandando el valor de los resultados a absoluto para después obtener la media.
#
# </div>

# ## Conclusión
#
# Zyfra nos proporciona los datos en bruto que fueron descargados del almacén de datos:
#
#     - gold_recovery_train
#     - gold_recovery_test
#     - gold_recovery_full
#
# Se hace el escrutinio de los datos, donde no se encuentran filas duplicadas, pero sí un porcentaje de valores ausentes.
#
# Se verifica que el valor del Error Absoluto Medio fuera cercano a cero para verificar que las predicciones del modelo son casi idénticas a los valores reales.
#
# Se verifican cuales son las columnas que hacen falta en el dataset de test, para identificar que en su mayoria son columnas de procesos de salida.
#
# Para los datos ausentes, se utiliza un rellenado de datos con los datos inmediatos siguientes, debido a que se conoce que los datos obtenidos en horarios cercanos suelen ser muy parecidas.
#
# A travez de histogramas se muestra cómo cambia la concentración de metales (Au, Ag, Pb) en función de la etapa de purificación y tambien se muestra la concentracion total de todas las sustancias en las diferentes etapas, donde se observan valores en 0, que se deciden eliminar las filas con datos menores que 0, por ser valores atípicos.
#
# Para el modelo de Machine Learning, se calcula el sMAPE (error medio absoluto porcentual simétrico), y se utiliza tanto Bosque Aleatorio y Regresión Lineal con validación cruzada.
#
# Finalmente el modelo de Bosque Aleatorio con profundidad 5 proporciona el mejor valor de sMAPE que fue de 10.0223, lo que indica que las predicciones de este modelo son más cercanas a los valores reales.
