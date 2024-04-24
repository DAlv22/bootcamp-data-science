# # Proyecto OilyGiant

# ## Introducción
#
# OilyGiant es una compañia de extracción de petróleo que busca seguir expandiendo sus alcances y para ello busca encontrar zonas donde se encuentren los pozos de petroléo que mejor retribución generen.

# ## Objetivo del proyecto
#
# Encontrar los mejores lugares donde abrir 200 pozos nuevos de petróleo con un presupuesto de 100 millones de dolares.

# ### Objetivo específico
#
# Se crea un modelo que ayude a elegir la región con el mayor margen de beneficio y se proporciona un análisis de los beneficios y riesgos potenciales.

# ## Datos
#
# Para trabajar el presente proyecto, OilyGiant proporciona los datos de exploración geológica sobre muestras de crudo de tres regiones con los parámetros de cada pozo petrolero de la región, por lo que se procede a importar esos archivos.

# ### Carga de librerias y de datos
#
# Ser carga la información proporcionada por OilyGiant.
#
# - Se descargan las librerías necesarias para el tratamiento necesarios de los datos


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats as st
import locale


# - Se importan los datos de las regiones pretolíferas

# In[2]:


region_0 = pd.read_csv('/datasets/geo_data_0.csv')
region_1 = pd.read_csv('/datasets/geo_data_1.csv')
region_2 = pd.read_csv('/datasets/geo_data_2.csv')


#

# - Se revisan las características generales de los datos

# Región 0

# In[3]:


print(region_0.shape)
print()
print(region_0.info())
print()
print(region_0.dtypes)
print()
print(region_0.describe())
print()
print(f'Este dataset tiene {region_0.duplicated().sum()} filas duplicadas.')
display(region_0)


# Región 1

# In[4]:


print(region_1.shape)
print()
print(region_1.info())
print()
print(region_1.dtypes)
print()
print(region_1.describe())
print()
print(f'Este dataset tiene {region_1.duplicated().sum()} filas duplicadas.')
display(region_1)


# Región 2

# In[5]:


print(region_2.shape)
print()
print(region_2.info())
print()
print(region_2.dtypes)
print()
print(region_2.describe())
print()
print(f'Este dataset tiene {region_2.duplicated().sum()} filas duplicadas.')
display(region_2)


# ### Revisión de los datos
#
# Recordando que los datasets muestran las muestras de crudo de tres regiones con los parámetros de cada pozo petrolero de la región.
#
# Los 3 datasets cuentan con 1000 registros cada uno, conformado por 5 columnas, sin datos nulos ni filas duplicadas.
# Las 5 columnas se conforman de la siguiente forma:
#
#     •	id — identificador único de pozo de petróleo
#     •	f0, f1, f2 — tres características de los puntos (Calidad de crudo)
#     •	product — volumen de reservas en el pozo de petróleo (miles de barriles)
#
# Los datos se describen de la siguiente forma:
#
#     - columna ID: tipo objetc
#     - columnas F0, F1, F2 y Product: tipo flotante
#     - En la región 0 y región 2, las medias y medianas de todas las columnas se encuentran muy cercanas una con la otra en cada región
#     - En la región 1 las medias y medianas de la columna f0, f1, f2 se encuentran muy cercanas una con la otra, excepto en la columna de producto, donde esto nos es así y se tenemos al media mas alta que la mediana
#
# Comparación los mínimos de cada columna por región
#
# Nombre de columna | Región 0 | Región 1 | Región 2
# --- | --- | --- | ---
# F0 | -1.408605 | -31.609576 | -8.760004
# F1| -0.848218 | -26.358598 | -7.084020
# F2 | -12.088328 | -0.018144 | -11.970335
# Producto | 0 | 0 | 0
#
#
# Comparación los máximos de cada columna por región
#
# Nombre de columna | Región 0 | Región 1 | Región 2
# --- | --- | --- | ---
# F0 | 2.362331 | 29.421755 | 7.238362
# F1| 1.343769 | 18.734063 | 7.844801
# F2 | 16.003790 | 5.019721 | 16.739402
# Producto | 185.364347 | 137.945408 | 190.029838
#
# El único tratamiento de datos que daremos para estos datasets, será la eliminación de la columna ID, debido a que no apoartará ningún tipo de información a nuestros futuros análisis.

# In[6]:


region_0 = region_0.drop(columns=['id'])
region_1 = region_1.drop(columns=['id'])
region_2 = region_2.drop(columns=['id'])


# In[7]:


print(region_0.head())
print(region_1.head())
print(region_2.head())


# ## Creación modelo regresión lineal
#
# Se procede a trabajar la creación del modelo de regresión lineal.

# ### Separación de datos Región 0

# In[8]:


features_region_0 = region_0.drop(['product'], axis=1)
target_region_0 = region_0['product']

features_train_region_0, features_valid_region_0, target_train_region_0, target_valid_region_0 = train_test_split(
    features_region_0, target_region_0, test_size=0.25, random_state=12345)


# #### Regresión linear Región 0

# In[9]:


model_region_0_regression = LinearRegression()
model_region_0_regression.fit(features_train_region_0, target_train_region_0)
predictions_valid_region_0_regression = model_region_0_regression.predict(
    features_valid_region_0)

result_region_0_regression = mean_squared_error(
    target_valid_region_0, predictions_valid_region_0_regression) ** 0.5

print("RECM del modelo de regresión lineal en el conjunto de validación para la región 0:",
      result_region_0_regression)


# ### Separación de datos Región 1

# In[10]:


features_region_1 = region_1.drop(['product'], axis=1)
target_region_1 = region_1['product']

features_train_region_1, features_valid_region_1, target_train_region_1, target_valid_region_1 = train_test_split(
    features_region_1, target_region_1, test_size=0.25, random_state=12345)


# #### Regresión linear Región 1

# In[11]:


model_region_1_regression = LinearRegression()
model_region_1_regression.fit(features_train_region_1, target_train_region_1)
predictions_valid_region_1_regression = model_region_1_regression.predict(
    features_valid_region_1)

result_region_1_regression = mean_squared_error(
    target_valid_region_1, predictions_valid_region_1_regression) ** 0.5

print("RECM del modelo de regresión lineal en el conjunto de validación para la región 1:",
      result_region_1_regression)


# ### Separación de datos región 2

# In[12]:


features_region_2 = region_2.drop(['product'], axis=1)
tarjet_region_2 = region_2['product']

features_train_region_2, features_valid_region_2, target_train_region_2, target_valid_region_2 = train_test_split(
    features_region_2, tarjet_region_2, test_size=0.25, random_state=12345)


# #### Bosque Aleatorio Región 2

# In[13]:


model_region_2_regression = LinearRegression()
model_region_2_regression.fit(features_train_region_2, target_train_region_2)
predictions_valid_region_2_regression = model_region_2_regression.predict(
    features_valid_region_2)

result_region_2_regression = mean_squared_error(
    target_valid_region_2, predictions_valid_region_2_regression) ** 0.5

print("RECM del modelo de regresión lineal en el conjunto de validación para la región 2:",
      result_region_2_regression)


# ### Análisis de resultados
#
# Los resultados del análisis indican que los modelos de regresión lineal para las tres regiones muestran variaciones significativas en su rendimiento, medido por el error cuadrático medio (RECM) en el conjunto de validación.
#
#     - Para la Región 0, el modelo presenta un RECM de 37.58 (miles de barriles), sugiriendo una cantidad sustancial de variabilidad en las predicciones
#
#     - La Región 1 destaca con un RECM mínimo de 0.89 (miles de barriles), indicando un rendimiento mucho más preciso en las predicciones. Este resultado sugiere que el modelo en la Región 1 tiene una capacidad predictiva más fuerte en comparación con las otras regiones
#
#     - En cambio, la Región 2 muestra un RECM de 40.03 (miles de barriles), lo que sugiere un mayor grado de error en las predicciones
#
# Una vez con estos resultados se procede a continuar el análisis.


# ## Preparación para el cálculo de ganancias
#
# Se conoce que al explorar la región, se lleva a cabo un estudio de 500 puntos con la selección de los mejores 200 puntos para el cálculo del beneficio.
#
# Para OilyGiant el panorama para el desarrollo de 200 pozos es el siguiente:
#
#     - Presupuesto: 100 millones de dólares
#     - Un barril de materias primas genera 4.5 USD de ingresos
#     - El ingreso de una unidad de producto es de 4500 dólares (el volumen de reservas está expresado en miles de barriles)
#     - De media un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades) *Compara esta cantidad con la cantidad media de reservas en cada región.

# In[14]:


budget = 100000000  # dólares
revenue_per_barrel = 4.5  # Dólares por barril
income_per_unit = 4500  # Dólares por unidad (por 1000 barriles)
units = 200
media_production = 500000  # dólares
min_uni = 111.1


# In[15]:


media_region_0 = region_0['product'].mean()
media_region_1 = region_1['product'].mean()
media_region_2 = region_2['product'].mean()

print(f'Media región 0: {round(media_region_0, 2)} miles de barriles')
print(f'Media región 1: {round(media_region_1, 2)} miles de barriles')
print(f'Media región 2: {round(media_region_2, 2)} miles de barriles')


# In[16]:


total_zero_region_0 = (region_0['product'] == 0).sum()
total_zero_region_1 = (region_1['product'] == 0).sum()
total_zero_region_2 = (region_2['product'] == 0).sum()

print(f"Total de '0' en en la región O es de: {total_zero_region_0}")
print(f"Total de '0' en en la región 1 es de: {total_zero_region_1}")
print(f"Total de '0' en en la región 2 es de: {total_zero_region_2}")


#  Recapitulando se conoce lo siguiente:
#
#  Reservas | Región 0 | Región 1 | Región 2
# --- | --- | --- | ---
# Promedio reservas previstas | 92.56 | 68.73 | 94.97
# RECM | 37.58 | 0.89 | 40.03
# Promedio por región | 92.5 | 68.83 | 95
# Valor máximo | 185.36 | 137.95 | 190.03
# Total en 0 | 1 | 8235 | 1
#
#  Se puede observar en la tabla anterior, que el promedio en las predicciones y la media de las reservas por región, no están muy alejadas unas de las otras.
#  Por otra parte se puede asumir por el RECM que la variabilidad de los datos en la región 0 y región 2 es mayor y que los datos en la región 1 tienen menos variabilidad.
#
#  Teniendo en cuenta también los valores máximos en reservas, en la región 0 y 2 se tienen reservas más grandes que en la región 1.
#
#  Se revisó el numero de 0 que se tienen en los registros por region en el apartado de reservas, arrojando que la región 0 y 2 solo tienen 1, y que la región 1 tienen 8235 registros en 0.
#
# Considerando que un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades), y reflexionando los datos obtenidos hasta ahora, en primera instancia comparando unicamente el promedio de las predicciones realizadas por región, ninguna llegaría al mínimo de 111.1 unidades. Revisando los valores máximos, se puede ver que la regiones 0 y 2 pueden llegar a tener pozos con reservas de grandes cantidades. La región 1, con sus valores máximos también podría sobrepasar lo mínimo requerido.

# ### Conclusiones

# Se debe seguir explirando los datos, ahora con un calculo de beneficio, para conocer cual de las zonas podría proporcionar mayor beneficio a la compañia.

# ## Calculo de ganancias
#
# Con base a las reservas de petróleo previsto para cada region, se calcula el beneficio. Se crea una funcion que tome como parámetros las predicciones de volumen, así como el tamaño de la muestra de pozos petrolíferos de cada región.
#
# Se tomarán en cuenta los 200 pozos más grandes de cada región.

# ### Ganancia potencial de los principales pozos por región
#
# Se realiza una función que nos ayuda a calcular la ganancias en los principales 200 pozos de la región

# In[17]:


def funcion_profit(target_valid, predictions_valid, count):
    target_valid_reset = target_valid.reset_index(drop=True)
    predictions_valid_series = pd.Series(predictions_valid)
    predictions_sorted_selected = predictions_valid_series.sort_values(ascending=False)[
        :count]
    selected_wells = target_valid_reset[predictions_sorted_selected.index]
    potential_profit = (
        (selected_wells.sum() * income_per_unit) - budget) / 1000000
    return round(potential_profit, 2)


predictions = [pd.Series(predictions_valid_region_0_regression), pd.Series(
    predictions_valid_region_1_regression), pd.Series(predictions_valid_region_2_regression)]
targets = [target_valid_region_0.reset_index(drop=True), target_valid_region_1.reset_index(
    drop=True), target_valid_region_2.reset_index(drop=True)]
profits = []

for i in range(len(predictions)):
    profits.append(funcion_profit(targets[i], predictions[i], 200))


for i in range(len(profits)):
    print(
        f'La ganancia potencial en la región {i} es de: {profits[i]} millones de dolares.')


# ### Propuesta
#
# Se propone la Región 0 para la apertura de 200 pozos, debido a que la ganancia potencial reflejada en nuestro estimado es el mas alto.


# ## Calculos de riesgos y ganancias por región

# Se crea una función que informará de los beneficios que podría tener cada región, esto por medio de un bootstrapping que tomará 1000 muestras a partir de 500 pozos. El intervalo de confianza del para esta prueba es del 95% y se proporciona el riesgo de pérdida para cada región.

# ### Bootstrapping
#


def region_sample(targets, predictions, count):

    state = np.random.RandomState(12345)

    values = []

    combined_df = pd.DataFrame()
    combined_df['predictions'] = predictions
    combined_df['targets'] = targets.reset_index(drop=True)

    for i in range(1000):
        target_subsample = combined_df.sample(
            n=500, replace=True, random_state=state).reset_index(drop=True)
        values.append(funcion_profit(
            target_subsample['targets'], target_subsample['predictions'], count))

    values = pd.Series(values)
    mean = values.mean()
    upper = values.quantile(0.975)
    lower = values.quantile(0.025)

    count = 0
    for value in values:
        if value < 0:
            count += 1

    print(
        f'- El beneficio promedio es de: {round(mean, 2)} millones de dolares')
    print(
        f'- El intervalo de confianza del 95% es: {round(lower, 2)} a {round(upper, 2)}')
    print(f"- El riesgo de perdida es del: {count * 100 / len(values)}%")


for i in range(len(predictions)):
    print(f'Datos para la Region {str(i)}:')
    region_sample(targets[i], predictions[i], 200)
    print()


# ### Observaciones
#
# Se puede ver que la región 0 y la región 2 tienen un riesgo de perdida superior a los indicado, que era un máximo de 2.5%.
# Acorde a los criterior indicados por OilyGiant, la región 1 es la sugerida para la realización de los nuevos 200 pozos de petróleo.
#
# Las ganancias promedio arrojadas por el boostrapping podrían apegarse más a lo que realmente se podría llegar a obtener la compañia, tomando en cuenta que es poco probable que los 200 pozos se lleven a cabo especificamente en los sitios con mayor número de reservas petróleras.

# ### Conclusiones
#
# En el presente proyecto se trabajó con los conjuntos de datos proporcionados por OilyGiant que contienen la Calidad de crudo
# y el volumen de reservas en los pozos de petróleros (miles de barriles). Con esa información y después de adecuarla para su uso, se entrenó un modelo de regresión lineal para cada región para calcular y predecir el beneficio potencial que cada región tiene. Para tener una mejor certeza en la propuesta, se utiliza la técnica de bootstrapping con 1000 muestras para encontrar el beneficio promedio, el intervalo de confianza del 95%, y el riesgo de pérdida para cada región. Se concluye con la sugerencia de la Región 1 para la contrucción sus nuevos pozos.
