
# La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y por lo que se procede a revisar las tareas por realizar.
# - Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
# - Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. ¿Puede un modelo de predictivo funcionar mejor que un modelo dummy?
# - Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
# - Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior. Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas.
#

# # Preprocesamiento y exploración de datos
#
# ## Inicialización


import numpy as np
import pandas as pd
import math
import seaborn as sns

import sklearn.linear_model
import sklearn.preprocessing

import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score


import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from IPython.display import display

# Es preferible agrupar los imports siguiendo el siguiente orden:
#
# Imports de la biblioteca estándar de Python.
# Imports de bibliotecas de terceros relacionadas.
# Imports específicos de la aplicación local o biblioteca personalizada.
# Para mejorar la legibilidad del código, también es recomendable dejar una línea en blanco entre cada grupo de imports, pero solo un import por línea.
# Te dejo esta referencia con ejemplos:
# https://pep8.org/#imports
#
# </div>

# ## Carga de datos

# Se carga los datos y se hace una revisión básica para comprobar que no hay problemas obvios.

# In[2]:


df = pd.read_csv('/datasets/insurance_us.csv')


# Renombramos las columnas para que el código se vea más coherente con su estilo.

# In[3]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income',
               'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


#

# In[4]:


df.sample(10)


# In[5]:


df.info()


# In[6]:


print(
    f'El número total de filas duplicadas en este archivo es de {df.duplicated().sum()} filas.')


# In[7]:


duplicates = df[df.duplicated()]
print(duplicates)


# En otras palabras, en lugar de hacer esto:
#
# ```python
# print(mi_dataframe)
# ```
#
# Puedes hacer esto en una celda por separado:
#
# ```python
# mi_dataframe
# ```
#
# Cuando lo haces de esta manera, Jupyter Notebook interpreta y muestra tus DataFrames de una manera más agradable, haciéndolos más fáciles de leer. No es necesario usar `print` en este caso.
#
# Espero que esta sugerencia te ayude a trabajar de manera más eficiente en tus proyectos. ¡Sigue adelante y sigue aprendiendo!
# </div>
#

# Se tienen 153 filas duplicdas, sin embargo no parece haber una tendencia por la cual se haya duplicado la información. Si se pudiera contactar al personal que proporcionó la información, se les contactaría para averiguar esta duplicidad. Sin embargo, se decide no borrar las filas dupicadas.
#
# Se procede a tranformar el tipo de datos de la columna edad (age).

# In[8]:


df['age'] = df['age'].astype(int)
df.info()


# Te recomendaría revisar tu conjunto de datos y considerar si hay oportunidades para optimizar los tipos de datos. ¡Esto puede hacer una gran diferencia en el rendimiento de tus análisis y modelos!
# </div>
#

# In[9]:


df.describe()


# Las estadisticas descriptivas no reflejan anomalías.

# In[ ]:


# ## Análisis exploratorio de datos

# Vamos a comprobar rápidamente si existen determinados grupos de clientes observando el gráfico de pares.

# In[10]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# A simple vista, no se detecgan grupos obvios (clústeres), ya que es difícil combinar diversas variables simultáneamente (para analizar distribuciones multivariadas).

# # Tarea 1. Clientes similares

# En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos.
# Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)- Distancia entre vectores -> Distancia euclidiana
# - Distancia entre vectores -> Distancia Manhattan
#
# Para resolver la tarea, podemos probar diferentes métricas de distancia.

# Escribe una función que devuelva los k vecinos más cercanos para un $n^{th}$ objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.
# Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) o tu propia implementación.
# Pruébalo para cuatro combinaciones de dos casos- Escalado
#   - los datos no están escalados
#   - los datos se escalan con el escalador [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
# - Métricas de distancia
#   - Euclidiana
#   - Manhattan
#
# Responde a estas preguntas:- ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?- ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

# In[11]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[12]:


def get_knn(df, n, k, metric):
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar    :param n: número de objetos para los que se buscan los vecinos más cercanos    :param k: número de vecinos más cercanos a devolver
    :param métrica: nombre de la métrica de distancia    """

    nbrs = NearestNeighbors(n_neighbors=n, metric=metric)
    nbrs.fit(df[feature_names])

    nbrs_distances, nbrs_indices = nbrs.kneighbors(
        [df.iloc[n][feature_names]], k, return_distance=True)

    df_res = pd.concat([
        df.iloc[nbrs_indices[0]],
        pd.DataFrame(nbrs_distances.T,
                     index=nbrs_indices[0], columns=['distance'])
    ], axis=1)

    return df_res


# Escalar datos.

# In[13]:


transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(
    df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(
    df[feature_names].to_numpy())


# In[14]:


df_scaled.sample(5)


# Ahora, vamos a obtener registros similares para uno determinado, para cada combinación

# Combinación 1: Datos no escalados, con distancia Euclidiana.

# In[15]:


get_knn(df, 7, 5, "euclidean")


# Combinación 2: Datos no escalados, con distancia Manhattan.

# In[16]:


get_knn(df, 7, 5, "manhattan")


# Combinación 3: Datos escalado, con distancia Euclidiana.

# In[17]:


get_knn(df_scaled, 7, 5, "euclidean")


# Combinación 4: Datos escalados, con distancia Manhattan.

# In[18]:


get_knn(df_scaled, 7, 5, "manhattan")


# Respuestas a las preguntas

# **¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?**
#
# Al no tener escalados los datos, el algoritmo kNN puede sugerir diferente indices de vecinos más cercanos, en comparación con los indices de los vecinos más cercanos de los datos sí escalados.
#
# Para los datos no escalados y para los escalados, el algoritmo sugiere el mismo índice de vecinos más cercanos, sin importar con qué distancia se corra el algoritmo.

# **¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?**
#
# Los resultados no son similares.

# # Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?

# En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

# Con el valor de `insurance_benefits` superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy.
# Instrucciones:
# - Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) o tu propia implementación.- Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1.
# La probabilidad de pagar cualquier prestación del seguro puede definirse como
# $$
# P\{\text{prestación de seguro recibida}\}=\frac{\text{número de clientes que han recibido alguna prestación de seguro}}{\text{número total de clientes}}.
# $$
#
# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.

# In[19]:


# Calcula el objetivo
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)
df_scaled['insurance_benefits_received'] = (
    df_scaled['insurance_benefits'] > 0).astype(int)


# In[20]:


# comprueba el desequilibrio de clases con value_counts()

df['insurance_benefits_received'].value_counts()/df.shape[0]*100


#     - 88.72% de los clientes nunca han recibido beneficos alguno por parte del seguro.
#     - 11.28% de los clientes sí han recibido en algún momento algún beneficio por parte del seguro.
#
#
# Se procede a seprar las caracteristicas y objetivos de los data frames con datos escalados y no escalados.
#
# No escalados

# In[21]:


target = df['insurance_benefits_received']
features = df.drop(['insurance_benefits_received',
                   'insurance_benefits'], axis=1)


features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.3, random_state=12345)


# Escalados

# In[22]:


target_scaled = df_scaled['insurance_benefits_received']
features_scaled = df_scaled.drop(
    ['insurance_benefits_received', 'insurance_benefits'], axis=1)

features_train_s, features_valid_s, target_train_s, target_valid_s = train_test_split(
    features, target, test_size=0.3, random_state=12345)


# Contrucción del clasificador KNN

# In[23]:


def knn_classifier(features_train, target_train, features_valid, target_valid, metric):

    best_F1 = 0
    best_k = 0

    for k in range(1, 11):
        neigh = KNeighborsClassifier(
            n_neighbors=k, weights='distance', metric=metric, p=2)
        neigh.fit(features_train, target_train)

        predictions = neigh.predict(features_valid)
        score_f1 = f1_score(target_valid, predictions)

        if score_f1 > best_F1:
            best_F1 = score_f1
            best_k = k

    print('Best k = ', best_k, '\nBest F1 Score: ', best_F1)


# Datos no escalados

# In[24]:


knn_classifier(features_train, target_train,
               features_valid, target_valid, 'euclidean')


# Datos Escalados

# In[25]:


knn_classifier(features_train_s, target_train_s,
               features_valid_s, target_valid_s, 'euclidean')


# Acorde a los resultados obtenidos después de la aplaicación del clasificador, se obtienen los mismos mejores resultados para los datos escalados que para los no escalados, tanto para k vecinos como para el F1.

# Ahora se fija el modelo con el mejor valor de K

# In[26]:


neigh = KNeighborsClassifier(
    n_neighbors=2, weights='distance', metric='euclidean', p=2)
neigh.fit(features_train, target_train)

predictions = neigh.predict(features_valid)


# Se ejecuta la función que evalúa el rendimiento del clasificador.

# In[27]:


def eval_classifier(y_true, y_pred):

    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión')
    print(cm)


# In[28]:


eval_classifier(target_valid, predictions)


# El valor para F1 podría ser mejor, pero vale la pena revisar la matriz de confusión para determinar si los resultados son lo suficientemente satisfactorios para lo que se está buscando.
#
# 87.2% se clasificó correctamente como positivo
# 5.6% se clasificó correctamente como negativo
# 1.93% se clasificó incorrectamente como negativo
# 5.27% se clasificó incorrectamente como positivo

# Se genera un modelo Dummy para comprobar que nuestro modelo tiene mejores resutlados que cualquier otro modelo aleatorio, el cual devuelve "1" con cierta probabilidad.
#
# Se prueba el modelo dummy  con 4 valores de probabilidad:
#
#     -  0
#     - probabilidad de pagar cualquier prestación del seguro
#     - 0.5
#     -  1
#
# La probabilidad de pagar cualquier prestación del seguro puede definirse como
#
# $$
# P\{\text{prestación de seguro recibida}\}=\frac{\text{número de clientes que han recibido alguna prestación de seguro}}{\text{número total de clientes}}.
# $$

# In[29]:


def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[30]:


for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'La probabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, df.shape[0], seed=42)

    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)

    print()


#  El valor más grande para F1 en el modelo Dummy fue de F1=.20 y conociendo que valor para F1 del clasificado fue de 0.61, se determina que el clasificador es mejor que el modelo dummy.

# # Tarea 3. Regresión (con regresión lineal)

# Con `insurance_benefits` como objetivo, evalúa cuál sería la RECM de un modelo de regresión lineal.

# Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de LA. Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?
#
# Denotemos- $X$: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades- $y$ — objetivo (un vector)- $\hat{y}$ — objetivo estimado (un vector)- $w$ — vector de pesos
# La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:
# $$
# y = Xw
# $$
#
# El objetivo de entrenamiento es entonces encontrar esa $w$ w que minimice la distancia L2 (ECM) entre $Xw$ y $y$:
#
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
#
# Parece que hay una solución analítica para lo anteriormente expuesto:
# $$
# w = (X^T X)^{-1} X^T y
# $$
#
# La fórmula anterior puede servir para encontrar los pesos $w$ y estos últimos pueden utilizarse para calcular los valores predichos
# $$
# \hat{y} = X_{val}w
# $$

# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

# In[31]:


class MyLinearRegression:

    def __init__(self):

        self.weights = None

    def fit(self, X, y):

        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = (np.linalg.inv(X2.T@X2) @ X2.T) @ y

    def predict(self, X):

        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2.dot(self.weights)

        return y_pred


# In[32]:


def eval_regressor(y_true, y_pred):

    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')

    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')


# In[33]:


def run_linear_regression(x):
    y = df['insurance_benefits'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=12345)

    lr = MyLinearRegression()
    lr.fit(x_train, y_train)
    print('Peso w: ', lr.weights)

    y_test_pred = lr.predict(x_test)
    print('\nPredicciones conjunto de prueba: ', y_test_pred, '\n')

    return eval_regressor(y_test, y_test_pred)


# In[34]:


x = df[['age', 'gender', 'income', 'family_members']].to_numpy()
run_linear_regression(x)


# In[35]:


x = df_scaled[['age', 'gender', 'income', 'family_members']].to_numpy()
run_linear_regression(x)


# En los resultados de la aplicación de la regresión lineal para los datos escalados y no escalados es la siguiente:
#
# Datos | age | gender | income | family_members | valor w
# --- | --- | --- | --- | --- | ---
# **No escalados** | -9.43539012e-01 | 3.57495491e-02 | 1.64272726e-02 | -2.60743659e-07 | -1.16902127e-02
# **Sí escalados** | -0.94353901 | 2.32372069 | 0.01642727 | -0.02059875 | -0.07014128
#
# Cada resultado de la tabla anterior, indica cómo contribuye cada uno al modelo de regresión lineal.
#
#      - Un coeficiente positivo indica que un aumento en el valor de esa característica está asociado con un aumento en la variable objetivo
#      - Un coeficiente negativo indica el decremento en el valor de esa característica asociada con la disminución del la variable objetivo.
#
# No hay diferencia entre el RECM y el R2 de los datos originales y la de los escalados.

# # Tarea 4. Ofuscar datos

# Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz $X$) por una matriz invertible $P$.
#
# $$
# X' = X \times P
# $$
#
# Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que $P$ sea realmente invertible.
#
# Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.

# In[36]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[37]:


X = df_pn.to_numpy()


# Generar una matriz aleatoria $P$ y se comprueba que es invertible.

# In[38]:


def random_matrix(seed):
    rng = np.random.default_rng(seed=seed)
    P = rng.random(size=(X.shape[1], X.shape[1]))
    print('Matriz aleatoria\n', P, '\n')

    # Obtener la inversa de la matriz P
    P_inv = np.linalg.inv(P)
    print('\nInversa de matriz aleatoria\n', P_inv)

    return P, P_inv


# In[39]:


P, P_inv = random_matrix(42)


# ¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

# In[40]:


x_transformed = X @ P
df_transformed = pd.DataFrame(x_transformed, columns=personal_info_column_list)

pd.DataFrame(x_transformed, columns=personal_info_column_list).head()


# No es posible adivinar la edad o los ingresos de los clientes después de la transformación.

# ¿Puedes recuperar los datos originales de $X'$ si conoces $P$? Intenta comprobarlo a través de los cálculos moviendo $P$ del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles

# In[41]:


df_recovered = pd.DataFrame(x_transformed @ P_inv,
                            columns=personal_info_column_list)


# Muestra los tres casos para algunos clientes
#
#     - Datos originales
#     - El que está transformado
#     - El que está invertido (recuperado)

# In[42]:


print("DataFrame Original:")
print(df[personal_info_column_list].head())
print()

print("DataFrame transformado:")
print(df_transformed.head())
print()

print("DataFrame recuperado:")
print(df_recovered.head())


# Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

# La diferencia se debe a los decimales, que se pueden atender facílmente.

# In[43]:


df_recovered = df_recovered.abs().round(1)
df_recovered


# ## Prueba de que la ofuscación de datos puede funcionar con regresión lineal

# En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar _analytically_ que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

# Entonces, los datos están ofuscados y ahora tenemos $X \times P$ en lugar de tener solo $X$. En consecuencia, hay otros pesos $w_P$ como
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
#
# ¿Cómo se relacionarían $w$ y $w_P$ si simplificáramos la fórmula de $w_P$ anterior?
#
# Teniendo presente que la fórmula en cuestión:
#
# $$w_P = [(XP)^T XP]^{-1} (XP)^T y$$
#
#   Dado que $XP = X × P$, podemos reescribir $(XP)^T XP$ como:
#
#   $$(XP)^T XP = (X × P)^T (X × P)$$
#
#   Usando la propiedad de la traspuesta $(AB)^T = B^T A^T$, podemos expandir esto como:
#
#   $$(X × P)^T (X × P) = P^T X^T X P$$
#
#   Entonces, la expresión para wP se convierte en:
#
#   $$wP = [P^T X^T X P]^{-1} P^T X^T y$$
#
# Ahora, para simplificar wP, podemos utilizar la propiedad de inversión de la multiplicación de matrices:
#
#
#   $$(AB)^{-1} = B^{-1} A^{-1}$$
#
#   Aplicando esta propiedad a wP, obtenemos:
#
#   $wP = (P^T X^T X P)^{-1} P^T X^T y$
#
#   $= P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T y$
#
#   $= (X^T X)^{-1} X^T y$
#

# Esto es idéntico a la fórmula original para $w$, lo que demuestra que la regresión lineal no se ve afectada por el proceso de ofuscación en términos de valores predichos.

# ¿Cuáles serían los valores predichos con $w_P$?
#
# Los valores predichos $ \hat{y}_P $ utilizando ${w}_P $ serían los mismos que los valores predichos $\hat{y} $ utilizando $w $, ya que $w_P$ es idéntico a $w $. Por lo tanto, la calidad de la regresión lineal medida mediante la RECM no se vería afectada por el proceso de ofuscación.
#

# ## Prueba de regresión lineal con ofuscación de datos

# Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida.
# Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación.
# Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y $R^2$. ¿Hay alguna diferencia?

# **Procedimiento**
#
# - Crea una matriz cuadrada $P$ de números aleatorios.
# - Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz invertible.
# - Utiliza $XP$ como la nueva matriz de características

# Al ya haberse trabajado una matriz invertible, se toma para este ejercicio.

# In[44]:


P, P_inv = random_matrix(72)


# In[45]:


x_transformed = X @ P


# In[46]:


run_linear_regression(X)


# In[47]:


run_linear_regression(x_transformed)


# Se observa que tanto para los datos orginales como los ofuscados, no hay diferencia entre los valores RMSE y $R^2$.

# # Conclusiones

# Con base a las tareas solicitadas por la compañía, se concluye lo siguiente:
#
#     - Tarea 1: encontrar clientes que sean similares a un cliente determinado
#         - Utilizando el clasificador de KNN, se encuentran diferencias entre los indices de los vecinos más cercanos entre datos escalados y no escalados. Sin embargo, los resultados dados utilizando 2 diferentes calculos de distancias eran iguales para el conjunto de datos escalados, e iguales entre el conjunto de datos no escalados.
#
#     - Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro.
#         - Al correr el modelo basado en Clasificación KNN, se obtuvo el mejor la mejor métrica de f1 score en 0.61 y con base a la información de la matriz de confusión se obtuvo un porcentaje de verdaderos positivos = 6.06%.
#
#     - Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
#         - Se obtuvieron los pesos para la predicción a travez de un modelo de regresión lineal, donde las métricas de evaluación fueron RMSE: 0.34 y R2: 0.66, para los datos sin balanceo y escalados.
#
#     - Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior.
#         - Se explica a travez de una demostración y una aplicación, que la ofuscación de datos, no afectan las métricas del modelo y que siempre se puede regresar a los datos originales, después de una ofuscación.
#
