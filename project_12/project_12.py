# # Introduccion
#
# Rusty Bargain, empresa dedicada a la venta de carros usados, se encuentra desarrollando una app para atraer nuevos clientes, lo que hace atractiva la app es que pueda averiguar rápidamente el valor de mercado del coche del cliente.
#
# # Recursos
#
# La compañia proporciona la siguiente información:
#
#      - especificaciones técnicas
#      - versiones de equipamiento
#      - precios
#
# # Objetivo
#
# Crear un modelo que determine el valor de mercado de los vehiculos de los clientes. El modelo debe complir con los siguientes requerimientos:
#
#     - la calidad de la predicción
#     - la velocidad de la predicción
#     - el tiempo requerido para el entrenamiento

# ## Preparación de datos

# Importación de librerias

# In[1]:


import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as ltb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline

from catboost import CatBoostRegressor, Pool
from time import time


# In[2]:


df = pd.read_csv("/datasets/car_data.csv")


# In[3]:


df.info()


# In[4]:


display(df)


# In[5]:


df.isnull().sum()


# In[6]:


print(
    f'El número total de filas duplicadas en este archivo es de {df.duplicated().sum()} filas.')


# In[7]:


duplicates = df[df.duplicated()]
print(duplicates)


# In[8]:


df.describe()


# ### Descripción de los datos
#
# El dataset proporcionado  cuenta con un total de registros de 354,369 y 16 columnas, de las cuales tenemos la siguiente información:
#
# | Nombre de columna | Característica | Tipo de dato | Datos Nulos |
# |-------------------|----------------|--------------|-------------|
# | DateCrawled | Fecha en la que se descargó el perfil de la base de datos | object | 0 |
# | Price | Precio | int64 | 0 |
# | VehicleType | Tipo de carrocería del vehículo | object | 37,490 |
# | RegistrationYear | Año de matriculación del vehículo | int64 | 0 |
# | Gearbox | Tipo de caja de cambios | object | 19,833 |
# | Power | Potencia (CV) | int64 | 0 |
# | Model | Modelo del vehículo | object | 19,705 |
# | Mileage | Kilometraje (medido en km de acuerdo con las especificidades regionales del conjunto de datos) | int64 | 0 |
# | RegistrationMonth | Mes de matriculación del vehículo | int64 | 0 |
# | FuelType | Tipo de combustible | object | 32,895 |
# | Brand | Marca del vehículo | object | 0 |
# | NotRepaired | Vehículo con o sin reparación | object | 71,154 |
# | DateCreated | Fecha de creación del perfil | object | 0 |
# | NumberOfPictures | Número de fotos del vehículo | int64 | 0 |
# | PostalCode | Código postal del propietario del perfil (usuario) | int64 | 0 |
# | LastSeen | Fecha de la última vez que el usuario estuvo activo | object | 0 |
#
# Las columnas con fechas serán cambiados su tipo de datos:
#
#     - Solo se dejará año
#         DateCrawled
#         DateCreated
#         LastSeen
#
#
# Se tiene este total de valores asentes:
#
#     VehicleType          37490 object 10.58%
#     Gearbox              19833 object 5.6%
#     Model                19705 object 5.6%
#     FuelType             32895 object 9.28%
#     NotRepaired          71154 object 20.08%
#
# Los primeros 4 tipos de datos se pueden intentar llenar con kNN vecinos.
# NotRepaired se rellanará con datos NaN.
#
# Se observa que las columnas 'Price', 'Power' y tienen celdas con valores en 0, lo que podría ser anormal.
#
# Los datos en la columna 'RegistrationYear' tiene un rango de años estraño que va del 1000 al 9999.
# Se desconoce la razón por la cual la columna 'NumberOfPictures' presenta sus datos en 0.
#
# El número total de filas duplicadas en este archivo es de 262 filas, mismas que se procederan a eliminarse, debido a que son muy pocos registro, que hay duplicados en filas que incluyen fecha y hora exacta y que no se puede investigar la razón de su probable duplicidad.

# In[9]:


df = df.drop_duplicates()


# In[10]:


print(
    f'El número total de filas duplicadas en este archivo es de {df.duplicated().sum()} filas.')


# Cambio de nombre de las columnas a snake case!

# In[11]:


def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# In[12]:


df.columns = [to_snake_case(col) for col in df.columns]


# In[13]:


display(df)


# Cambio de tipo de dato a las columnas con fecha

# In[14]:


columns_date = ['date_crawled', 'date_created', 'last_seen']


for columns in columns_date:
    df[columns] = pd.to_datetime(df[columns], infer_datetime_format=True)


# In[16]:


for columns in columns_date:
    df[columns] = pd.DatetimeIndex(df[columns]).year

display(df)


# Ya con los datos duplicados eliminados, y las fechas ya adaptadas se procede a revisar lo antes indicado.
#
# Se revisa profundamente las columnas 'Price', 'Power'y  'RegistrationYear'

# In[17]:


df.loc[:, ['price', 'power', 'registration_year']].describe()


# Se elimina la columna 'number_of_pictures' dado que todos sus valores estan en 0

# In[18]:


df = df.drop('number_of_pictures', axis=1)


# La composición de los datos ausentes es de la siguiente forma:
#
#     NotRepaired          71154 object 20.08%
#     VehicleType          37490 object 10.58%
#     FuelType             32895 object 9.28%
#     Gearbox              19833 object 5.6%
#     Model                19705 object 5.6%
#
# Se buscan las frecuencias relativas de estas columnas:

# In[19]:


def category_distribution(df, column):

    print(f'Distribution for {column}')
    print(df[column].value_counts(normalize=True, dropna=False))
    print()


# In[20]:


cat_cols = ['not_repaired', 'vehicle_type', 'fuel_type', 'gearbox', 'model']
for col in cat_cols:
    category_distribution(df, col)


# Para la columna 'not_repaired' no se tiene información suficiente para realizar imputación de datos, por lo que se opta por colocar en los datos ausentes la palabra 'unknown'.

# In[21]:


df['not_repaired'].fillna('unknown', inplace=True)


# En el caso de 'gearbox' se trabaja una moda condiconal, tomando como referencia la columna 'brand'. Se crea una función que tome como entrada un conjunto de columnas y devuelve el valor más frecuente en cada columna.

# In[22]:


def replace_null(cols):
    return cols.astype(str).mode(dropna=False).iloc[0]


# Se aplica a 'gearbox'

# In[23]:


conditional_gearbox = df.groupby(
    ['brand'])['gearbox']          .transform(replace_null)

df['gearbox'] = df['gearbox'].fillna(conditional_gearbox)


# In[24]:


category_distribution(df, 'gearbox')


# De esta manera se asigna un valor adecuado a los datos ausentes.


# Para 'vehicle_type', 'fuel_type', 'model' revisa más a fondo para rellenar los NaN con datos más especificos. Se revisan sus frecuencias relativas.

# In[25]:


opt_cols = ['registration_year', 'power', 'brand']
for col in opt_cols:
    category_distribution(df, col)


# In[26]:


df['registration_year'].sort_values().unique()


# En 'registration_year' se tienen datos incoherentes y se decide quitar estos datos

# In[27]:


df = df.loc[~((df['registration_year'] <= 1800) |
              (df['registration_year'] >= 2066)), :]

print(df['registration_year'].sort_values().unique())


# In[28]:


df.info()


# La eliminación de esta información no fue significativa para el dataset completo..
#
# Con los datos finales se revisa el tipo de vehículos que se tienen y la cantidad que hay por tipo. El objetivo es llenar los datos ausentes con el tipo de vehículo más significativo por modelo.
#
# Para ello, se buscará los modelos con mayor cantidad de datos ausentes y se les imputará la moda.

# In[29]:


nan_counts = df[df['vehicle_type'].isna()]['model'].value_counts().head(30)
relative_frequency = nan_counts / nan_counts.sum()

relative_frequency


# Se decide trabajar con los primeros 18 registros, que representan el 80% de los datos.

# In[30]:


model_mode = df.groupby('model')['vehicle_type'].agg(pd.Series.mode)
model_mode


# In[31]:


models = ['golf', 'other', 'polo', 'corsa', '3er', 'astra', 'a3', 'twingo',
          'passat', 'fiesta', 'a_klasse', 'focus', 'punto', 'a4', 'vectra',
          'c_klasse', 'transporter', 'clio']


# In[32]:


for v_type in models:
    print(v_type, ':', model_mode[v_type])


# In[33]:


df['vehicle_type'] = np.where(
    (df['vehicle_type'].isna() == True) & (df['model'].isin(['golf', 'other ',
                                                             '3er', 'astra', 'a3', 'a_klasse', 'vectra', 'c_klasse'])), 'sedan',
    np.where((df['vehicle_type'].isna() == True) & (df['model'].isin(['polo', 'corsa', 'twingo', 'fiesta', 'punto', 'clio'])), 'small',
             np.where((df['vehicle_type'].isna() == True) & (df['model'].isin(['passat', 'focus', 'a4'])), 'wagon',
             np.where((df['vehicle_type'].isna() == True) & (df['model'] == 'transporter'), 'bus', df['vehicle_type']))))

df.dropna(subset=['vehicle_type'], inplace=True)


# Se sigue sigue el mismo patrón anterior para los datos ausentes de 'fuel_type'.

# In[34]:


fuel_mode = df.groupby('vehicle_type')['fuel_type'].agg(pd.Series.mode)
print(fuel_mode)


# In[35]:


for vehicle_type, fuel in fuel_mode.items():
    df.loc[df['vehicle_type'] == vehicle_type, 'fuel_type'] = df.loc[df['vehicle_type']
                                                                     == vehicle_type, 'fuel_type'].fillna(fuel)


# In[36]:


df.info()


# Por ultimo, se utiliza la función previamente creada para el calculo de la moda de la columna 'model' para cada combinación única de 'brand' y 'registration_year' y que reemplace los valores faltantes en 'model' con estas modas calculadas.

# In[37]:


conditional_model = df.groupby(['brand', 'registration_year'])[
    'model']          .transform(replace_null)

df['model'] = df['model'].fillna(conditional_model)


# In[38]:


df.info()


# In[39]:


nan_count_model = df['model'].value_counts()['nan']
print("Cantidad de 'nan' en la columna 'model':", nan_count_model)


# Algunos datos que quedaban ausentes, se quedarán como datos desconocidos con el nombr de 'nan'

# Para terminar la preparación de datos, se revisa el tema de los datos mínimos de la columna 'price'y 'power'.

# In[40]:


columns_interest = ['price', 'power']

for col in columns_interest:
    print('\n', col)
    print(df[col].sort_values().unique())


# In[41]:


df_filtered_power = df[(df['power'] >= 0) & (df['power'] <= 100)]

value_frequency_2 = df_filtered_power['power'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(value_frequency_2.index, value_frequency_2.values,
         marker='o', linestyle='-')
plt.xlabel('Power')
plt.ylabel('Frequency')
plt.title('Frequency of registrations')
plt.grid(True)
plt.show()


# In[42]:


df_filtered_price = df[(df['price'] >= 0) & (df['price'] <= 100)]

value_frequency_3 = df_filtered_price['price'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(value_frequency_3.index, value_frequency_3.values,
         marker='o', linestyle='-')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Frequency of registrations')
plt.grid(True)
plt.show()


# Se decide observar el comportamiento de los datos para considerar los datos que se conservarán.
#
#         Para Power a partir de 40
#         Para Price a partir de 50

# In[43]:


df = df.loc[(df['power'] >= 40) & (df['price'] >= 50)]


# In[44]:


df = df.reset_index(drop=True)


# In[45]:


display(df)


# El total de observaciones originales era de 354369 y después de la preparación ya decuación de los datos, se trabajará con un total de 293819 observaciones. La limpieza supuso una pérdida del 17.09% de los datos.
#
#
# Por último, para determinar con qué columnas trabajar, se revisa la correlación de las variables

# In[46]:


plt.figure(figsize=(10, 10), dpi=80)
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", fmt=".2f")
plt.show()


# La matriz de correlación muestra que no hay relación entre el precio y las columnas: 'date_crawled', 'registration_month', 'date_created', 'postal_code' y 'last_seen'.
#
# Por lo cual se dejarán fuera de los modelos. Por lo que la información con la que se trabajará será el siguiente:
#
# Columna objetivo
#
#     price - precio al que se vende el vehículo
#
# Columnas características
#
#     vehicle_type - tipo de carrocería del vehículo
#     registration_year - año de matriculación del vehículo
#     gearbox - tipo de caja de cambios
#     power - potencia (CV)
#     model - modelo
#     mileage - kilometraje (Km)
#     fuel_type - tipo de combustible
#     brand - marca del vehículo
#     not_repaired - vehículo con o sin reparación

# ## Entrenamiento del modelo

# Se entrena diferentes modelos con varios hiperparámetros con el objetivo de comparar la métrica raíz del error cuadrático medio (RECM).
#
# Se ejecutarán las siguientes modelos:
#
#     - Modelo de regresión lineal (como prueba de cordura)
#     - Modelo árbol de decision
#     - Modelo bosque aleatorio
#     - LightGBM

# ### Separación y división de datos
#
# Eliminamos las columnas que ya no se necesitan

# In[47]:


df = df.drop(['date_crawled', 'registration_month',
             'date_created', 'postal_code', 'last_seen'], axis=1)


# Se aplica  la codificación one-hot a las variables categóricas

# In[48]:


df = pd.get_dummies(df, drop_first=True)


# Se separan el conjunto de price 75% entrenamiento, 25% de validación

# In[49]:


target = df['price']
features = df.drop('price', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)


# Se realiza el escalado de los datos númericos

# In[50]:


numeric_features_train = features_train.select_dtypes(include=['number'])
numeric_features_valid = features_valid.select_dtypes(include=['number'])
scaler = StandardScaler()
numeric_features_train_scaled = scaler.fit_transform(numeric_features_train)
numeric_features_valid_scaled = scaler.transform(numeric_features_valid)


categorical_features_train = features_train.select_dtypes(exclude=['number'])
categorical_features_valid = features_valid.select_dtypes(exclude=['number'])


features_train_complete = np.hstack(
    (numeric_features_train_scaled, categorical_features_train))
features_valid_complete = np.hstack(
    (numeric_features_valid_scaled, categorical_features_valid))


# Se crea la función que nos ayudará al calculo de los tiempos de entrenamiento y validación

# In[51]:


def train_and_pred_time(model, features_train, target_train, features_valid):
    start_time1 = time()
    train = model.fit(features_train, target_train)
    end_time1 = time()
    train_time = end_time1 - start_time1

    start_time2 = time()
    predictions = model.predict(features_valid)
    end_time2 = time()
    pred_time = end_time2 - start_time2

    return train_time, pred_time, predictions


# ### Modelos

# *Regresión Lineal*

# In[52]:


model_lr = LinearRegression()
lr_train_time, lr_pred_time, lr_pred = train_and_pred_time(model_lr, features_train_complete, target_train,
                                                           features_valid_complete)


# In[53]:


lr_train_rmse = round(mean_squared_error(
    target_train, model_lr.predict(features_train_complete), squared=False), 2)
lr_valid_rmse = round(mean_squared_error(
    target_valid, lr_pred, squared=False), 2)

print('Valor RMSE para el entrenamiento:', lr_train_rmse)
print('Valor RMSE para la validation:', lr_valid_rmse)
print('Tiempo de entrenamiento:', lr_train_time)
print('Tiempo de predicción:', lr_pred_time)


# *Regresión lineal con uso de Descenso de Gradientes Estocástico*

# In[54]:


model_sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=12345)
sgd_train_time, sgd_pred_time, sgd_pred = train_and_pred_time(model_sgd, features_train_complete, target_train,
                                                              features_valid_complete)


# In[55]:


sgd_train_rmse = round(mean_squared_error(
    target_train, model_sgd.predict(features_train_complete), squared=False), 2)
sgd_valid_rmse = round(mean_squared_error(
    target_valid, sgd_pred, squared=False), 2)

print('Valor RMSE para el entrenamiento:', sgd_train_rmse)
print('Valor RMSE para la validation:', sgd_valid_rmse)
print('Tiempo de entrenamiento:', sgd_train_time)
print('Tiempo de predicción:', sgd_pred_time)


# *Árbol de desición*
#
# Se buscan los mejores hiperparámetros con una grilla, y se implementa el modelo con los mejores hiperparámetros encontrados

# In[56]:


param_grid = {'max_depth': [1, 2, 4, 6, 8, 10, 15, 20]}
tree_reg = DecisionTreeRegressor(random_state=12345)
grid_search = GridSearchCV(
    estimator=tree_reg, param_grid=param_grid, cv=5, verbose=True)
get_ipython().run_line_magic(
    'time', 'grid_search.fit(features_train_complete, target_train)')
best_model_tree_reg = grid_search.best_estimator_

dtr_train_time, dtr_pred_time, dtr_pred = train_and_pred_time(best_model_tree_reg, features_train_complete,
                                                              target_train, features_valid_complete)

dtr_train_rmse = round(mean_squared_error(target_train, best_model_tree_reg.predict(features_train_complete),
                                          squared=False), 2)
dtr_valid_rmse = round(mean_squared_error(
    target_valid, dtr_pred, squared=False), 2)

print('\nValor RMSE para el entrenamiento:', dtr_train_rmse)
print('Valor RMSE para la validación:', dtr_valid_rmse)
print('Tiempo de entrenamiento:', dtr_train_time)
print('Tiempo de predicción:', dtr_pred_time)


# *Random Forest*
#
# Igualmente aplicamos grilla en la busqueda de los mejores hiperparámetros y se implementan en el modelo

# In[57]:


model_rfr = RandomForestRegressor(random_state=12345)
param_grid = {'n_estimators': [1, 2, 3, 4, 6, 8, 10], 'max_features': [
    'sqrt', 'log2'], 'max_depth': [2, 3, 4, 5, 6, 8, 10]}
grid_search_rfr = GridSearchCV(
    estimator=model_rfr, param_grid=param_grid, cv=5)
grid_search_rfr.fit(features_train_complete, target_train)


# In[59]:


best_model_rfr = grid_search_rfr.best_estimator_

rfr_train_time, rfr_pred_time, rfr_pred = train_and_pred_time(
    best_model_rfr, features_train_complete, target_train, features_valid_complete)

rfr_train_rmse = round(mean_squared_error(
    target_train, best_model_rfr.predict(features_train_complete), squared=False), 2)
rfr_valid_rmse = round(mean_squared_error(
    target_valid, rfr_pred, squared=False), 2)

print('Valor RMSE para el entrenamiento:', rfr_train_rmse)
print('Valor RMSE para la validación:', rfr_valid_rmse)
print('Tiempo de entrenamiento:', rfr_train_time)
print('Tiempo de predicción:', rfr_pred_time)


# *LightGBM*

# In[60]:


model_ltb1 = ltb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)

ltb_train_time1, ltb_pred_time1, ltb_pred1 = train_and_pred_time(model_ltb1, features_train_complete, target_train,
                                                                 features_valid_complete)


# In[61]:


ltb1_train_rmse = round(mean_squared_error(
    target_train, model_ltb1.predict(features_train_complete), squared=False), 2)
ltb1_valid_rmse = round(mean_squared_error(
    target_valid, ltb_pred1, squared=False), 2)

print('Valor RMSE para el entrenamiento:', ltb1_train_rmse)
print('Valor RMSE para la validation:', ltb1_valid_rmse)
print('Tiempo de entrenamiento:', ltb_train_time1)
print('Tiempo de predicción:', ltb_pred_time1)


# In[62]:


model_ltb2 = ltb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=20, max_depth=10, learning_rate=0.05, n_estimators=100)

ltb_train_time2, ltb_pred_time2, ltb_pred2 = train_and_pred_time(model_ltb2, features_train_complete, target_train,
                                                                 features_valid_complete)


# In[63]:


ltb2_train_rmse = round(mean_squared_error(
    target_train, model_ltb2.predict(features_train_complete), squared=False), 2)
ltb2_valid_rmse = round(mean_squared_error(
    target_valid, ltb_pred2, squared=False), 2)


print('Valor RMSE para el entrenamiento:', ltb2_train_rmse)
print('Valor RMSE para la validation:', ltb2_valid_rmse)
print('Tiempo de entrenamiento:', ltb_train_time2)
print('Tiempo de predicción:', ltb_pred_time2)

#

# ## Análisis del modelo

# In[64]:


model_names = ['Regresión Lineal', 'Descenso de Gradientes Estocástico',
               'Árbol de desición', 'Bosque Aleatorio', 'LightGBM_1', 'LightGBM_2']

scores = [lr_valid_rmse, sgd_valid_rmse, dtr_valid_rmse,
          rfr_valid_rmse, ltb1_valid_rmse, ltb2_valid_rmse]
training_times = [lr_train_time, sgd_train_time, dtr_train_time,
                  rfr_train_time, ltb_train_time1, ltb_train_time2]
prediction_times = [lr_pred_time, sgd_pred_time,
                    dtr_pred_time, rfr_pred_time, ltb_pred_time1, ltb_pred_time2]


models_df = pd.DataFrame({
    'Model': model_names,
    'RMSE': scores,
    'Training Time': training_times,
    'Prediction Time': prediction_times
})

print(models_df)


# In[68]:


models_sorted_rmse = models_df.sort_values(by='RMSE')
models_sorted_training = models_df.sort_values(by='Training Time')
models_sorted_prediction = models_df.sort_values(by='Prediction Time')

filtered_df_dge = models_df[models_df['Model']
                            != 'Descenso de Gradientes Estocástico']
filtered_sorted_rmse_dge = filtered_df_dge.sort_values(by='RMSE')

fig, axs = plt.subplots(1, 3, figsize=(12, 8))
fig.suptitle('Comparación de algoritmos de aprendizaje automático', fontsize=15)


axs[0].bar(filtered_sorted_rmse_dge['Model'],
           filtered_sorted_rmse_dge['RMSE'], color='g')
axs[0].set_xticks(np.arange(len(filtered_sorted_rmse_dge['Model'])))
axs[0].set_xticklabels(filtered_sorted_rmse_dge['Model'], rotation=90)
axs[0].set_title('RMSE score')


axs[1].bar(models_sorted_training['Model'],
           models_sorted_training['Training Time'], color='b')
axs[1].set_xticks(np.arange(len(models_sorted_training['Model'])))
axs[1].set_xticklabels(models_sorted_training['Model'], rotation=90)
axs[1].set_title('Tiempo de entrenamiento')


axs[2].bar(models_sorted_prediction['Model'],
           models_sorted_prediction['Prediction Time'], color='orange')
axs[2].set_xticks(np.arange(len(models_sorted_prediction['Model'])))
axs[2].set_xticklabels(models_sorted_prediction['Model'], rotation=90)
axs[2].set_title('Tiempo de Predicción')

plt.tight_layout()
plt.show()

# # Concluciones

# Después de la importanción de datos, se procedió a revisar con qué tipo de datos contabamos y a la misma vez, saber que columnas tenian datos nulos. Se siguieron diferentes estrategias para trabajar los datos nulos, desde la eliminación de columnas que no tenian aportación a la inforación de los datos y a la imputación de los datos a travez de modas condiconales.
#
# Una vez con los datos limpios se comienza a trabajar en la implementación de modelos, para ello, se uso la técnica de ONE HOT para datos categoricos y StandarScaler para datos númericos.
#
# Se procede a trabajar con los modelos:
#
#     - Regresión Lineal
#     - Descenso de Gradientes Estocástico
#     - Árbol de desición
#     - Bosque Aleatorio
#     - LightGBM
#
# donde se calculó el Error Cuadrático Medio (RMSE) y los tiempos de entrenamiento y predicción, donde se puede observar los siguiente:
#
# Con base a la observación de resultados el modelo LightGBM_1 parece ser la mejor opción para determinar el valor de mercado de los vehículos de los clientes. Este modelo tiene un RMSE bajo, lo que indica que sus predicciones son precisas, su tiempo de predicción se puede considerar y su  tiempo de entrenamiento es aceptable, considerando la precisión y velocidad del modelo.
#
# Observaciones del resto de modelos:
#
# Regresión Lineal: Este modelo tiene el segundo mejor RMSE, pero su tiempo de predicción es más lento que el de LightGBM_1.
# Descenso de Gradientes Estocástico: Este modelo tiene un RMSE muy alto, lo que lo hace inadecuado para esta tarea.
# Árbol de Decisión: Este modelo tiene un RMSE similar al de LightGBM_1, pero su tiempo de predicción es más lento.
# Bosque Aleatorio: Este modelo tiene un RMSE similar al de LightGBM_1, pero su tiempo de entrenamiento es más lento.
#
# De igual manera, se puede analizar los diferentes resultados y elegir el camino que mejor se adapte a las necesidades del cliete final.
