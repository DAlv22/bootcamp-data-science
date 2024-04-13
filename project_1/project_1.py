
# # Contenido <a id='back'></a>
#
# * [Introducción](#intro)
# * [Etapa 1. Descripción de los datos](#data_review)
#     * [Conclusiones](#data_review_conclusions)
# * [Etapa 2. Preprocesamiento de datos](#data_preprocessing)
#     * [2.1 Estilo del encabezado](#header_style)
#     * [2.2 Valores ausentes](#missing_values)
#     * [2.3 Duplicados](#duplicates)
#     * [2.4 Conclusiones](#data_preprocessing_conclusions)
# * [Etapa 3. Prueba de hipótesis](#hypotheses)
#     * [3.1 Hipótesis 1: actividad de los usuarios y las usuarias en las dos ciudades](#activity)
#     * [3.2 Hipótesis 2: preferencias musicales los lunes y los viernes](#week)
#     * [3.3 Hipótesis 3: preferencias de género en Springfield y Shelbyville](#genre)
# * [Conclusiones](#end)

# ## Introducción <a id='intro'></a>
# Como analista de datos, tu trabajo consiste en analizar datos para extraer información valiosa y tomar decisiones basadas en datos. Esto implica diferentes etapas, como la descripción general de los datos, el preprocesamiento y la prueba de hipótesis.
#
# Siempre que investigamos, necesitamos formular hipótesis que después podamos probar. A veces aceptamos estas hipótesis; otras, las rechazamos. Para tomar las decisiones correctas, una empresa debe ser capaz de entender si está haciendo las suposiciones correctas.
#
# En este proyecto, compararás las preferencias musicales de las ciudades de Springfield y Shelbyville. Estudiarás datos reales de transmisión de música online para probar las hipótesis a continuación y comparar el comportamiento de los usuarios y las usuarias de estas dos ciudades.
#
# ### Objetivo:
# Prueba tres hipótesis:
# 1. La actividad de los usuarios y las usuarias difiere según el día de la semana y dependiendo de la cuidad.
# 2. Los lunes por la mañana, los habitantes de Springfield y Shelbyville escuchan géneros distintos. Lo mismo ocurre con los viernes por la noche.
# 3. Los oyentes de Springfield y Shelbyville tienen preferencias distintas. En Springfield prefieren el pop, mientras que en Shelbyville hay más personas a las que les gusta el rap.
#
# ### Etapas
# Los datos del comportamiento del usuario se almacenan en el archivo `/datasets/music_project_en.csv`. No hay ninguna información sobre la calidad de los datos, así que necesitarás examinarlos antes de probar las hipótesis.
#
# Primero, evaluarás la calidad de los datos y verás si los problemas son significativos. Entonces, durante el preprocesamiento de datos, tomarás en cuenta los problemas más críticos.
#
# Tu proyecto consistirá en tres etapas:
#  1. Descripción de los datos
#  2. Preprocesamiento de datos
#  3. Prueba de hipótesis
#
#
# ### Reto
#
# En este proyecto, preparamos un pequeño reto para ti. Incluimos un nuevo tipo de estructura de datos: las marcas temporales. Las marcas temporales son muy comunes y merecen una atención adicional. Más adelante en el programa, aprenderás mucho sobre ellas. Sin embargo, por ahora las trataremos como simples strings. Necesitamos marcas temporales en este proyecto para poner a prueba una de nuestras hipótesis. No te preocupes, te ayudaremos con esto. Tu nivel de conocimientos actual será suficiente para abordarlo.
#
# Por ejemplo, digamos que tenemos dos marcas temporales: `dt1 = "12:00:00"` y `dt2 = "06:00:00"`. Queremos comparar estas dos marcas temporales y ver cuál es posterior.
#
# Podemos compararlas mediante los operadores de comparación estándar (`<`, `>`, `<=`, `>=`, `==`, `!=`). Ejecuta la siguiente celda de código para comparar dos marcas temporales:
#

# In[68]:


# Comparar los objetos datetime

import pandas as pd
dt1 = "12:00:00"
dt2 = "06:00:00"

if dt1 < dt2:
    print("La marca temporal 2 es posterior")
else:
    print("La marca temporal 1 es posterior")


# ## Etapa 1. Descripción de los datos <a id='data_review'></a>
#
# Abre los datos y examínalos.

# Necesitarás `pandas`, así que impórtalo.

# In[69]:


# importar pandas


# Si estás trabajando en la plataforma, lee el archivo `music_project_en.csv` de la carpeta /datasets/ y guárdalo en la variable `df`. Si estás trabajando localmente, no es necesario especificar carpeta alguna siempre y cuando el archivo `music_project_en.csv` esté en la misma carpeta que este Notebook:


# leer el archivo y almacenarlo en df
df = pd.read_csv('/datasets/music_project_en.csv')


# Muestra las 10 primeras filas de la tabla:

# In[71]:


# obtener las 10 primeras filas de la tabla df
display(df.head(10))


# Obtén la información general sobre la tabla con un comando:

# In[72]:


# obtener información general sobre los datos en df
print(df.info())


# Estas son nuestras observaciones sobre la tabla. Contiene siete columnas. Todas almacenan el mismo tipo de datos: `object` (objeto).
#
# Según la documentación:
# - `'userID'` — identificador del usuario o la usuaria;
# - `'Track'` — título de la canción;
# - `'artist'` — nombre del artista;
# - `'genre'` — género musical;
# - `'City'` — ciudad del usuario o la usuaria;
# - `'time'` — hora exacta en la que se reprodujo la pista;
# - `'Day'` — día de la semana.
#
# Podemos ver tres problemas con el estilo en los encabezados de la tabla:
# 1. Algunos encabezados están en mayúsculas, otros, en minúsculas.
# 2. Hay espacios en algunos encabezados.
# 3. `Faltaria espacio entre entre palabras que deberían ir separadas`.
#
#
#

# ### Tus observaciones <a id='data_review_conclusions'></a>
#
# `Escriba sus observaciones aquí:`
#
# `1.   ¿Qué tipo de datos tenemos a nuestra disposición en las filas? ¿Y cómo podemos entender lo que almacenan las columnas?`R= Todos los datos son objetos, los títulos de las columnas nos indican qué tipo de información podrémos obtener en cada una de ellas.
#
# `2.   ¿Hay suficientes datos para proporcionar respuestas a nuestras tres hipótesis, o necesitamos más información?`R=
#
#     Para la hipótesis 1, tenemos suficiente información contando con información de usuario, día de la semana y ciudad.
#
#     Para la hipótesis 2, tenemos el nombre de la ciudad, el horario de reproducción, día de la semana y género, por lo que sí podríamos tratar de resolverla.
#
#     Para la hipótesis 3, tenemos el nombre de las ciudades y tenemos el género, aunque de primera instancia, se desconoce si el rap viene descrito como tipo de género en la columna 'genre'. Tendríemos que verificarlo prosteriormente.
#
# `3.   ¿Notaste algún problema en los datos, como valores ausentes, duplicados o tipos de datos incorrectos?`R= Tenemos datos nulos en las columnas de 'track', 'artist' y 'genre' siendo 'artist' la que más datos nulos presenta. Aunado a los detalles por arreglar en el formato del título de las columnas y que en la fila 3, el usuario presenta un 'user_id' de 6 caractéres cuando el resto es de 8, desconozco si a futuro esto nos pueda traer algun tipo de problema.

# ## Etapa 2. Preprocesamiento de datos <a id='data_preprocessing'></a>
#
# El objetivo aquí es preparar los datos para que sean analizados.
# El primer paso es resolver cualquier problema con los encabezados. Luego podemos avanzar a los valores ausentes y duplicados. Empecemos.
#
# Corrige el formato en los encabezados de la tabla.
#

# ### Estilo del encabezado <a id='header_style'></a>
# Muestra los encabezados de la tabla:

# In[73]:


# la lista de encabezados para la tabla df
print(df.columns)


# Cambia los encabezados de la tabla de acuerdo con las reglas del buen estilo:
# * todos los caracteres deben ser minúsculas;
# * elimina los espacios;
# * si el nombre tiene varias palabras, utiliza snake_case.

# Pon todos los caracteres en minúsculas e imprime el encabezado de la tabla de nuevo:

# In[74]:


# bucle en los encabezados poniendo todo en minúsculas
df.columns = df.columns.str.lower()

print(df.columns)


# Ahora elimina los espacios al principio y al final de los encabezados y muéstralos:

# In[75]:


# bucle en los encabezados eliminando los espacios
df.columns = df.columns.str.strip()

print(df.columns)


# Aplica snake_case al encabezado userID e imprime el encabezado de la tabla:

# In[76]:


# cambiar el nombre del encabezado "user_id"
df.rename(columns={'userid': 'user_id'}, inplace=True)

print(df.columns)


# Comprueba el resultado. Muestra los encabezados una vez más:

# In[77]:


# comprobar el resultado: la lista de encabezados
print(df.columns)

# ### Valores ausentes <a id='missing_values'></a>
# Primero, encuentra el número de valores ausentes en la tabla. Para ello, utiliza dos métodos `pandas`:

# In[78]:


# calcular el número de valores ausentes
print(df.isna().sum())

print(df.isnull().sum())


# No todos los valores ausentes afectan a la investigación. Por ejemplo, los valores ausentes en `track` y `artist` no son cruciales. Simplemente puedes reemplazarlos con valores predeterminados como el string `'unknown'` (desconocido).
#
# Pero los valores ausentes en `'genre'` pueden afectar la comparación entre las preferencias musicales de Springfield y Shelbyville. En la vida real, sería útil saber las razones por las cuales hay datos ausentes e intentar recuperarlos. Pero no tenemos esa oportunidad en este proyecto. Así que tendrás que:
# * rellenar estos valores ausentes con un valor predeterminado;
# * evaluar cuánto podrían afectar los valores ausentes a tus cómputos;

# Reemplazar los valores ausentes en `'track'`, `'artist'` y `'genre'` con el string `'unknown'`. Para hacer esto, crea la lista `columns_to_replace`, recorre sobre ella con un bucle `for`, y para cada columna reemplaza los valores ausentes en ella:

# In[79]:


# bucle en los encabezados reemplazando los valores ausentes con 'unknown'
columns_to_replace = ['track', 'artist', 'genre']

for column in columns_to_replace:
    df[column].fillna('unknown', inplace=True)


# Asegúrate de que la tabla no contiene más valores ausentes. Cuenta de nuevo los valores ausentes.

# In[80]:


# contando valores ausentes
print(df.isna().sum())

# ### Duplicados <a id='duplicates'></a>
# Encuentra el número de duplicados explícitos en la tabla usando un comando:

# In[81]:


# contar duplicados explícitos
print(df.duplicated().sum())


# Llama al método `pandas` para deshacerte de los duplicados explícitos:

# In[82]:


# eliminar duplicados explícitos
df = df.drop_duplicates().reset_index(drop=True)


# Cuenta los duplicados explícitos una vez más para asegurarte de haberlos eliminado todos:

# In[83]:


# comprobación de duplicados
print(df.duplicated().sum())


# ```python
# df = df.drop_duplicates().reset_index(drop=True)
# ```
# </div>
#


# Ahora queremos deshacernos de los duplicados implícitos en la columna `genre`. Por ejemplo, el nombre de un género se puede escribir de varias formas. Dichos errores también pueden afectar al resultado.

# Para hacerlo, primero imprimamos una lista de nombres de género únicos, ordenados en orden alfabético. Para hacerlo:
# * recupera la columna deseada del dataFrame;
# * llama al método que te devolverá todos los valores de columna únicos;
# * aplica un método de ordenamiento a tu resultado.
#

# In[84]:


# inspeccionar los nombres de género únicos

genre_values = df['genre'].unique()
sorted_values = sorted(genre_values)

print(sorted_values)


# Busca en la lista para encontrar duplicados implícitos del género `hiphop`. Estos pueden ser nombres escritos incorrectamente o nombres alternativos para el mismo género.
#
# Verás los siguientes duplicados implícitos:
# * `hip`
# * `hop`
# * `hip-hop`
#
# Para deshacerte de ellos, declara la función `replace_wrong_genres()` con dos parámetros:
# * `wrong_genres=` — la lista de duplicados;
# * `correct_genre=` — el string con el valor correcto.
#
# La función debería corregir los nombres en la columna `'genre'` de la tabla `df`, es decir, remplaza cada valor de la lista `wrong_genres` con el valor en `correct_genre`. Utiliza un bucle `'for'` para iterar sobre la lista de géneros incorrectos y reemplazarlos con el género correcto en la lista principal.

# In[85]:


# función para reemplazar duplicados implícitos
def replace_wrong_genres(df, column, wrong_genres, correct_genre):
    for wrong_genre in wrong_genres:
        df[column].replace(wrong_genre, correct_genre, inplace=True)
    return df


# Llama a `replace_wrong_genres()` y pásale argumentos para que retire los duplicados implícitos (`hip`, `hop` y `hip-hop`) y los reemplace por `hiphop`:

# In[86]:


# eliminar duplicados implícitos
duplicates = ['hip', 'hop', 'hip-hop']
correct = 'hiphop'

df = replace_wrong_genres(df, 'genre', duplicates, correct)
print(df)


# Asegúrate de que los nombres duplicados se hayan eliminado. Muestra la lista de valores únicos de la columna `'genre'` una vez más:

# In[87]:


# comprobación de duplicados implícitos
genre_value_replaced = df['genre'].unique()
sorted_values_replaced = sorted(genre_value_replaced)

display(sorted_values_replaced)


# ### Tus observaciones <a id='data_preprocessing_conclusions'></a>
#
# `Describa brevemente lo que ha observado al analizar los duplicados, cómo abordó sus eliminaciones y qué resultados logró.`
#
# La primera fase fue el poder limpiar los datos con los cuales íbamos a trabajar. Teníamos un total de 3826 datos duplicados explícitos, dejándolos posteriormente a 0.
#
# Para trabajar con los duplicados implícitos, debíamos imprimir los géneros únicos y ordenarlos alfabéticamente para poder identificar más fácilmente estos duplicados. Aunque debo mencionar que de no ser porque el ejercicio daba por default los géneros repetidos, hacer una validación visual hubiera sido ineficiente. Supongo que hacer estas verificaciones se verán en un futuro. Una vez con la información de los géneros a unificar bajo un solo nombre, se procedió a construir una función que hiciera esta tarea.
#
# Se definió la función que tomaría 4 parámetros, el propio dataframe, la columna a trabajar, los nombres de los géneros escritos de manera diferente y el nombre del género bajo el cual pretendíamos unificar los nombres erróneos.
#
# Con un bucle for, tomamos un iterador que toma el valor de cada elemento en la lista ` “wrong_generes” `, que será donde se indique el nombre de género que se quiere cambiar. Después para no volver a declarar la variable, hacemos uso del “inplace”. En esta nueva línea de código que está dentro del bucle for, estaremos llamando a la columna con la que trabajaremos, en este caso ` ‘genre’ ` del dataframe y con ayuda del método `replace `, pedimos que tome lo que encuentre el iterador y lo cambie por la palabra a unificar, en este caso ` “correct_genre” ` y lo reemplace siempre que `inplace ` nos dé True.
#
# Por último, para ya poder ocupar nuestra función, se crean 2 variables, una que contenga la lista de los datos a cambiar y la otra indicará por qué palabra debe ser cambiada. Ya solo nos tomamos el tiempo para sustituir los parámetros de la función por los argumentos con los que queremos que trabaje.
#
# Lo que nos deja, según mis resultados, un total de 279 géneros músicales.

# ## Etapa 3. Prueba de hipótesis <a id='hypotheses'></a>

# ### Hipótesis 1: comparar el comportamiento del usuario en las dos ciudades <a id='activity'></a>

# La primera hipótesis afirma que existen diferencias en la forma en que los usuarios y las usuarias de Springfield y Shelbyville consumen música. Para comprobar esto, usa los datos de tres días de la semana: lunes, miércoles y viernes.
#
# * Agrupa a los usuarios y las usuarias por ciudad.
# * Compara el número de pistas que cada grupo reprodujo el lunes, el miércoles y el viernes.
#

# Realiza cada cálculo por separado.
#
# El primer paso es evaluar la actividad del usuario en cada ciudad. Agrupa los datos por ciudad y encuentra el número de canciones reproducidas en cada grupo.
#
#

# In[88]:


# contando las pistas reproducidas en cada ciudad

springfield = df[df['city'] == 'Springfield']['city'].count()
shelbyville = df[df['city'] == 'Shelbyville']['city'].count()

print(f'Las pistas reproducidas en Springfield son: {springfield} pistas')
print(f'Las pistas reproducidas en Shelbyville son: {shelbyville} pistas')


# `Comenta tus observaciones aquí`
#
# Podemos observar una diferencia significatiba en los hábitos de reproducción musical entre ambas ciudades. Podríamos indicar que por cada reproducción efectuada por los usuarios de `Shelbyville`, los usuarios de `Springfield` realizan un promedio de 2.3 reproducciones.

# Ahora agrupa los datos por día de la semana y encuentra el número de pistas reproducidas el lunes, el miércoles y el viernes.
#

# In[89]:


# Cálculo de las pistas reproducidas cada día de la semana

grouped_by_city = df.groupby('city')['track'].count()

grouped_by_day = df.groupby('day')['track'].count()

print("Número de tracks reproducidos por ciudad")
print(grouped_by_city)
print()
print("Número de tracks reproducidos por día")
print(grouped_by_day)


# Con base a los datos obtenidos, podemos comentar que el comportamiento entre ciudades verdaderamente es diferente entre sí.
#
# Como observación particular de `Shelbyville`, vemos que el miércoles es el día de la semana dónde más reproducciones se realizan.
#
# Para `Springfield` tanto lunes como viernes son días donde los usuarios hacen más reproducciones, siendo el miércoles el día que tiene una fuerte caída en reproducciones.
#
#

# Ya sabes cómo contar entradas agrupándolas por ciudad o día. Ahora necesitas escribir una función que pueda contar entradas según ambos criterios simultáneamente.
#
# Crea la función `number_tracks()` para calcular el número de canciones reproducidas en un determinado día **y** ciudad. La función debe aceptar dos parámetros:
#
# - `day`: un día de la semana para filtrar. Por ejemplo, `'Monday'`.
# - `city`: ciudad: una ciudad para filtrar. Por ejemplo, `'Springfield'`.
#
# Dentro de la función, aplicarás un filtrado consecutivo con indexación lógica.
#
# Primero filtra los datos por día y luego filtra la tabla resultante por ciudad.
#
# Después de filtrar los datos por dos criterios, cuenta el número de valores de la columna 'user_id' en la tabla resultante. Este recuento representa el número de entradas que estás buscando. Guarda el resultado en una nueva variable y devuélvelo desde la función.

# In[90]:


# <crear la función number_tracks()>

# declararemos la función con dos parámetros: day=, city=.

# deja que la variable track_list almacene las filas df en las que

# el valor del nombre de la columna ‘day’ sea igual al parámetro day= y, al mismo tiempo,

# el valor del nombre de la columna ‘city’ sea igual al parámetro city= (aplica el filtrado consecutivo

# con indexación lógica)

# deja que la variable track_list_count almacene el número de valores de la columna 'user_id' en track_list

# (igual al número de filas en track_list después de filtrar dos veces).
# permite que la función devuelva un número: el valor de track_list_count.

# la función cuenta las pistas reproducidas en un cierto día y ciudad.
# primero recupera las filas del día deseado de la tabla,
# después filtra las filas de la ciudad deseada del resultado,
# luego encuentra el número de pistas en la tabla filtrada,
# y devuelve ese número.
# para ver lo que devuelve, envuelve la llamada de la función en print().


# empieza a escribir tu código aquí

def number_tracks(day, city):
    track_list = df[(df['day'] == day) & (df['city'] == city)]
    track_list_count = track_list['user_id'].count()
    return track_list_count


# Ejemplos de uso de la función
print(number_tracks('Monday', 'Springfield'))
print(number_tracks('Wednesday', 'Shelbyville'))


# Llama a `number_tracks()` seis veces, cambiando los valores de los parámetros, para que recuperes los datos de ambas ciudades para cada uno de los tres días.

# In[91]:


# el número de canciones reproducidas en Springfield el lunes
print(number_tracks('Monday', 'Springfield'))


# In[92]:


# el número de canciones reproducidas en Shelbyville el lunes
print(number_tracks('Monday', 'Shelbyville'))


# In[93]:


# el número de canciones reproducidas en Springfield el miércoles
print(number_tracks('Wednesday', 'Springfield'))


# In[94]:


# el número de canciones reproducidas en Shelbyville el miércoles
print(number_tracks('Wednesday', 'Shelbyville'))


# In[95]:


# el número de canciones reproducidas en Springfield el viernes
print(number_tracks('Friday', 'Springfield'))


# In[96]:


# el número de canciones reproducidas en Shelbyville el viernes
print(number_tracks('Friday', 'Shelbyville'))


# Utiliza `pd.DataFrame` para crear una tabla, donde
# * los encabezados de la tabla son: `['city', 'monday', 'wednesday', 'friday']`
# * Los datos son los resultados que conseguiste de `number_tracks()`

# In[97]:


# tabla con los resultados

cities = ['Springfield', 'Shelbyville']
days = ['Monday', 'Wednesday', 'Friday']

heads = ['city', 'monday', 'wednesday', 'friday']
data = []

for city in cities:
    row = [city]
    for day in days:
        number_of_songs = number_tracks(day, city)
        row.append(number_of_songs)
    data.append(row)

table = pd.DataFrame(data=data, columns=heads)
display(table)


# **Conclusiones**
#
# `Comente si la primera hipótesis es correcta o debe rechazarse. Explicar tu razonamiento`
#
# La primera hípotesis nos dice:
#
#     1.-"La actividad de los usuarios y las usuarias difiere según el día de la semana y dependiendo de la ciudad."
#
# Se considera que la primera hipótesis es correcta. Observamos que los usuarios tienen un comportamiento distinto entre los días de la semana y las ciudades. Siendo que para días de alto número de reproducciones en una ciudad, para la otra ciudad el número de reproducciones cae.
#
# ### Hipótesis 2: música al principio y al final de la semana <a id='week'></a>

# Según la segunda hipótesis, el lunes por la mañana y el viernes por la noche, los ciudadanos de Springfield escuchan géneros que difieren de los que disfrutan los usuarios de Shelbyville.

# Cree dos tablas con los nombres proporcionados en los dos bloques de código a continuación:
# * Para Springfield — `spr_general`
# * Para Shelbyville — `shel_general`

# In[98]:


# crear la tabla spr_general a partir de las filas df
# donde los valores en la columna 'city' es 'Springfield'

spr_general = df[df["city"] == "Springfield"]
display(spr_general)


# In[99]:


# crear la tabla shel_general a partir de las filas df
# donde los valores en la columna 'city' es 'Shelbyville'

shel_general = df[df["city"] == "Shelbyville"]
display(shel_general)
#
# Buen trabajo!

# Escribe la función `genre_weekday()` con cuatro parámetros:
# * Una tabla para los datos (`df`)
# * El día de la semana (`day`)
# * La marca de fecha y hora en formato 'hh:mm:ss' (`time1`)
# * La marca de fecha y hora en formato 'hh:mm:ss' (`time2`)
#
# La función debe devolver los 15 géneros más populares en un día específico dentro del período definido por las dos marcas de tiempo, junto con sus respectivos recuentos de reproducción.
# Aplica la misma lógica de filtrado consecutiva, pero usa cuatro filtros esta vez y luego crea una nueva columna con los recuentos de reproducción respectivos.
# Ordena el resultado de un recuento más grande a uno más pequeño y devuélvelo.

# In[100]:


# 1) Deja que la variable genre_df almacene las filas que cumplen varias condiciones:
#    - el valor de la columna 'day' es igual al valor del argumento day=
#    - el valor de la columna 'time' es mayor que el valor del argumento time1=
#    - el valor en la columna 'time' es menor que el valor del argumento time2=
#    Utiliza un filtrado consecutivo con indexación lógica.

# 2) Agrupa genre_df por la columna 'genre', toma una de sus columnas,
#    y utiliza el método size() para encontrar el número de entradas por cada uno de
#    los géneros representados; almacena los Series resultantes en
#    la variable genre_df_count

# 3) Ordena genre_df_count en orden descendente de frecuencia y guarda el resultado
#    en la variable genre_df_sorted

# 4) Devuelve un objeto Series con los primeros 15 valores de genre_df_sorted - los 15
#    géneros más populares (en un determinado día, en un determinado periodo de tiempo)

# escribe tu función aquí

def genre_weekday(df, day, time1, time2):
    genre_df = df[(df['day'] == day) & (
        df['time'] > time1) & (df['time'] < time2)]
    genre_df_count = genre_df.groupby('genre')['genre'].count()
    genre_df_sorted = genre_df_count.sort_values(ascending=False)
    return genre_df_sorted[:15]

    # filtrado consecutivo
    # Crea la variable genre_df que almacenará solo aquellas filas df donde el día es igual a day=

    # Filtra genre_df nuevamente para almacenar solo las filas donde el tiempo es menor que time2=

    # Filtra genre_df una vez más para almacenar solo las filas donde el tiempo es mayor que time1=

    # Agrupa el DataFrame filtrado por la columna con los nombres de los géneros, selecciona la columna 'genre',
    # y encuentra el número de filas para cada género con el método count()

    # Ordenaremos el resultado en orden descendente (por lo que los géneros más populares aparecerán primero en el objeto Series)

    # Devuelve un objeto de Series con los primeros 15 valores de genre_df_sorted: los 15 géneros más populares (en un día determinado, dentro de un período de timeframe)


# Compara los resultados de la función `genre_weekday()` para Springfield y Shelbyville el lunes por la mañana (de 7 a 11) y el viernes por la tarde (de 17:00 a 23:00). Utiliza el mismo formato de hora de 24 horas que el conjunto de datos (por ejemplo, 05:00 = 17:00:00):

# In[101]:


# llamando a la función para el lunes por la mañana en Springfield (utilizando spr_general en vez de la tabla df)
print(genre_weekday(spr_general, 'Monday', '07:00:00', '11:00:00'))


# In[102]:


# llamando a la función para el lunes por la mañana en Shelbyville (utilizando shel_general en vez de la tabla df)
print(genre_weekday(shel_general, 'Monday', '07:00:00', '11:00:00'))


# In[103]:


# llamando a la función para el viernes por la tarde en Springfield
print(genre_weekday(spr_general, 'Friday', '17:00:00', '23:00:00'))


# In[104]:


# llamando a la función para el viernes por la tarde en Shelbyville
print(genre_weekday(shel_general, 'Monday', '17:00:00', '23:00:00'))


# **Conclusiones**
#
# `Comente si la segunda hipótesis es correcta o debe rechazarse. Explica tu razonamiento.`
#
#     2da hipótesis: Los lunes por la mañana, los habitantes de Springfield y Shelbyville escuchan géneros distintos. Lo mismo ocurre con los viernes por la noche.
#
# R=Con base a los datos obtenidos, podemos determinar que en ambas ciudades, los primeros 15 géneros más populares son muy similares, teniendo una similitud más cercana en el top 4 y alejándose esta similitud conforme descendemos en la lista. Podríamos concluir que esta hipótesis se rechaza, debido a que los géneros escuchados son los mismos, en un orden que no es igual en ambas ciudades exactamente, pero manteniendo una estrecha cercanía.

# ### Hipótesis 3: preferencias de género en Springfield y Shelbyville <a id='genre'></a>
#
# Hipótesis: Shelbyville ama la música rap. A los residentes de Springfield les gusta más el pop.

# Agrupa la tabla `spr_general` por género y encuentra el número de canciones reproducidas de cada género con el método `count()`. Luego ordena el resultado en orden descendente y guárdalo en la variable `spr_genres`.

# In[ ]:


# escribe una línea de código que:
# 1. agrupe la tabla spr_general por la columna 'genre';
# 2. cuente los valores 'genre' con count() en la agrupación;
# 3. ordene el Series resultante en orden descendente y lo guarde en spr_genres.


spr_general_count = spr_general.groupby('genre')['genre'].count()
spr_genres = spr_general_count.sort_values(ascending=False)


# Imprime las 10 primeras filas de `spr_genres`:

# In[ ]:


# muestra los primeros 10 valores de spr_genres
print(spr_genres[:10])


# Ahora haz lo mismo con los datos de Shelbyville.
#
# Agrupa la tabla `shel_general` por género y encuentra el número de canciones reproducidas de cada género. Después, ordena el resultado en orden descendente y guárdalo en la tabla `shel_genres`:
#

# In[ ]:


# escribi una línea de código que:
# 1. agrupe la tabla shel_general por la columna 'genre';
# 2. cuente los valores 'genre' con count() en la agrupación;
# 3. ordene el Series resultante en orden descendente y lo guarde en shel_genres.

shel_general_count = shel_general.groupby('genre')['genre'].count()
shel_genres = shel_general_count.sort_values(ascending=False)


# Imprime las 10 primeras filas de `shel_genres`:

# In[ ]:


# imprimir las 10 primeras filas de shel_genres
print(shel_genres[:10])


# **Conclusión**
#
# `Comente si la tercera hipótesis es correcta o debe rechazarse. Explica tu razonamiento.`
#
#     3.-Los oyentes de Springfield y Shelbyville tienen preferencias distintas. En Springfield prefieren el pop, mientras que en Shelbyville hay más personas a las que les gusta el rap.
#
# R=Con base a los datos obtenidos, podemos indicar que la 3ra hipótesis es parcialmente correcta. Afirmamos que en la Ciudad de Springfield el género más escuchado es el Pop. Sin embargo, se rechaza que en Shelbyville el genero más escuchado sea el rap.
# Podríamos concluir que en ambas ciudades el género Pop es el más reproducido. Podríamos agregar que el top 5 de géneros es el mismo para ambas ciudades y que el género rap no figura dentro del top 10 de ninguna de las dos ciudades.

# # Conclusiones <a id='end'></a>

# `Resuma sus conclusiones sobre cada hipótesis aquí`
#
# Conclusión hipótesis 1: Se acepta. Se observan que los patrones de comportamiento de los usuarios varían entre los días de la semana y las diferentes ciudades.
#
#
# Conclusión hipótesis 2: se rechaza. Se puede observar que en ambas ciudades los géneros más populares son muy similares, disminuyendo su similitud conforma descendemos en la lista.
#
#
# Conclusión hipótesis 3: se acepta parcialemnte. Debidoa que Springfield cumplió con lo estimado, pues el género más escuchado es el Pop, como se habia planteado. Mientras que para Shelbyville, el género mas escuchado No es el rap.
#
# A pesar de que el comportamiento de reproducciones varia segun la ciudad y el día, podemos observar que el gusto músical en ambas ciudades es muy similar.
