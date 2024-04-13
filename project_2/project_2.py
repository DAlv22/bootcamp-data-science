
# ## Diccionario de datos
#
# Hay cinco tablas en el conjunto de datos, y tendrás que usarlas todas para hacer el preprocesamiento de datos y el análisis exploratorio de datos. A continuación se muestra un diccionario de datos que enumera las columnas de cada tabla y describe los datos que contienen.
#
# - `instacart_orders.csv`: cada fila corresponde a un pedido en la aplicación Instacart.
#     - `'order_id'`: número de ID que identifica de manera única cada pedido.
#     - `'user_id'`: número de ID que identifica de manera única la cuenta de cada cliente.
#     - `'order_number'`: el número de veces que este cliente ha hecho un pedido.
#     - `'order_dow'`: día de la semana en que se hizo el pedido (0 si es domingo).
#     - `'order_hour_of_day'`: hora del día en que se hizo el pedido.
#     - `'days_since_prior_order'`: número de días transcurridos desde que este cliente hizo su pedido anterior.
# - `products.csv`: cada fila corresponde a un producto único que pueden comprar los clientes.
#     - `'product_id'`: número ID que identifica de manera única cada producto.
#     - `'product_name'`: nombre del producto.
#     - `'aisle_id'`: número ID que identifica de manera única cada categoría de pasillo de víveres.
#     - `'department_id'`: número ID que identifica de manera única cada departamento de víveres.
# - `order_products.csv`: cada fila corresponde a un artículo pedido en un pedido.
#     - `'order_id'`: número de ID que identifica de manera única cada pedido.
#     - `'product_id'`: número ID que identifica de manera única cada producto.
#     - `'add_to_cart_order'`: el orden secuencial en el que se añadió cada artículo en el carrito.
#     - `'reordered'`: 0 si el cliente nunca ha pedido este producto antes, 1 si lo ha pedido.
# - `aisles.csv`
#     - `'aisle_id'`: número ID que identifica de manera única cada categoría de pasillo de víveres.
#     - `'aisle'`: nombre del pasillo.
# - `departments.csv`
#     - `'department_id'`: número ID que identifica de manera única cada departamento de víveres.
#     - `'department'`: nombre del departamento.

# # Paso 1. Descripción de los datos
#
# Lee los archivos de datos (`/datasets/instacart_orders.csv`, `/datasets/products.csv`, `/datasets/aisles.csv`, `/datasets/departments.csv` y `/datasets/order_products.csv`) con `pd.read_csv()` usando los parámetros adecuados para leer los datos correctamente. Verifica la información para cada DataFrame creado.
#

# ## Plan de solución
#
# Escribe aquí tu plan de solución para el Paso 1. Descripción de los datos.
#
#     Primeramente averiguaré cómo se conforman los datos con los que estoy trabajando. Esto lo lograré con un .info(), además quiero ver el tipo de datos que hay en cada DataFrame por lo que también quiero ver los datos. Lo haré con display.

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


orders = pd.read_csv('/datasets/instacart_orders.csv', sep=';')
products = pd.read_csv('/datasets/products.csv', sep=';')
aisles = pd.read_csv('/datasets/aisles.csv', sep=';')
departments = pd.read_csv('/datasets/departments.csv', sep=';')
order_products = pd.read_csv('/datasets/order_products.csv', sep=';')


# In[3]:


orders.info()
print()
display(orders)


# In[4]:


products.info()
print()
display(products)


# In[5]:


aisles.info()
print()
display(aisles)


# In[6]:


departments.info()
print()
display(departments)


# In[7]:


order_products.info()
print()
print(order_products.isnull().sum())
display(order_products.head(5))


# ## Conclusiones
#
#
#     Nuestros archivos son csv, sin embargo no se encuentraban separados por ',' sino que estaban separados po ';'.
#     En el archivo de 'order_products' no se ven reflejadas las columnas de null, pero lo soluciono llamando a .isnull() con .sum()
#     Falta revisar algunos detalles de los datos, como tranformar algunos a int.
#

# # Paso 2. Preprocesamiento de los datos
#
# Preprocesa los datos de la siguiente manera:
#
# - Verifica y corrige los tipos de datos (por ejemplo, asegúrate de que las columnas de ID sean números enteros).
# - Identifica y completa los valores ausentes.
# - Identifica y elimina los valores duplicados.
#
# Asegúrate de explicar qué tipos de valores ausentes y duplicados encontraste, cómo los completaste o eliminaste y por qué usaste esos métodos. ¿Por qué crees que estos valores ausentes y duplicados pueden haber estado presentes en el conjunto de datos?

# ## Plan de solución
#
# Escribe aquí tu plan para el Paso 2. Preprocesamiento de los datos.
#
#     Encontrar y eliminar los valores duplicados de cada Dataframe.
#     Verificar la existencia de datos ausentes y analizar qué tipo de datos son para determinar cómo podré sustituirlos o en su defecto eliminarlos.
#

# ## Encuentra y elimina los valores duplicados (y describe cómo tomaste tus decisiones).

# ### `orders` data frame

# In[8]:


# Revisa si hay pedidos duplicados
display(orders[orders['order_id'].duplicated()])


# ¿Tienes líneas duplicadas? Si sí, ¿qué tienen en común?
#
#     Se hicieron el día miércoles a las 2 am, todo es idéntico a excepción del indice.

# In[9]:


# Basándote en tus hallazgos,
# Verifica todos los pedidos que se hicieron el miércoles a las 2:00 a.m.
display(orders[(orders['order_hour_of_day'] == 2)
        & (orders['order_dow'] == 3)])


# El resultado sugiere...
#
#     Había pensado que tendríamos algún error con los pedidos en miércoles a las 2 am. Pero obteniendo el total de pedido realizados el día miércoles a las 2, nos percatamos que no todos los pedidos tuvieron error de duplicarse. Podría tener alguna otra razón, quizá los duplicados se realizaron en un día en específico cuando se tuvo algún error en la plataforma de pedidos.

#


# Elimina los pedidos duplicados
orders = orders.drop_duplicates().reset_index(drop=True)


# In[11]:


# Vuelve a verificar si hay filas duplicadas
print("El numero de filas duplicadas es:", orders.duplicated().sum())


# In[12]:


# Vuelve a verificar únicamente si hay IDs duplicados de pedidos
print("El numero de IDs duplicados de pedidos es:",
      orders['order_id'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos
#
#     Se descubrieron que alguno de los registros se encontraba duplicados, probablemente algún tema sufrido por la plataforma generadora de datos, sin embargo, vemos que el problema solo se presentó con un número limitado de registros.

# ### `products` data frame

# In[13]:


# Verifica si hay filas totalmente duplicadas
print("El número de filas totalmente duplicadas es:", products.duplicated().sum())


# In[14]:


# Verifica únicamente si hay IDs duplicadas de productos
print("El númedo de IDs duplciados de producto es:",
      products['product_id'].duplicated().sum())


# In[15]:


# Revisa únicamente si hay nombres duplicados de productos (convierte los nombres a letras mayúsculas para compararlos mejor)
print("El total de nombres duplicados de productos es:",
      products['product_name'].duplicated().sum())
print()
display(products[products['product_name'].duplicated()])
print()
products['product_name'] = products['product_name'].str.upper()


# In[16]:


# Revisa si hay nombres duplicados de productos no faltantes
print(products['product_name'].value_counts(dropna=False))


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     Filas duplicadas no tuvimos en este DataFrame, sin embargo, si teniamos datos duplicados en la columna de 'product_name', un total de 1361 datos duplicados de los cuales 1258 corresponden a datos NaN. Dandonos un total de 103 registros duplicados de otro tipo de nombre de productos.

# ### `departments` data frame

# In[17]:


# Revisa si hay filas totalmente duplicadas
print("El número de filas totalmente duplicadas es:",
      departments.duplicated().sum())


# In[18]:


# Revisa únicamente si hay IDs duplicadas de productos
print("Esta es la cantidad de IDs duplicadas de producto:",
      departments['department_id'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     No contamos con con datos duplicados en este DataFrame

# ### `aisles` data frame

# In[19]:


# Revisa si hay filas totalmente duplicadas
print("El número de filas totalmente duplicadas es:", aisles.duplicated().sum())


# In[20]:


# Revisa únicamente si hay IDs duplicadas de productos
print("El numero de IDs duplicadas de productos es:",
      aisles['aisle_id'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     No tenemos datos duplicados en el DataFrame aisles. Al menos hasta los datos que hemos inspecionado.

# ### `order_products` data frame

# In[21]:


# Revisa si hay filas totalmente duplicadas
print("El número de filas totalmente duplicadas es:",
      order_products.duplicated().sum())


# In[22]:


# Vuelve a verificar si hay cualquier otro duplicado engañoso
print(order_products['order_id'].duplicated().sum())
print(order_products['product_id'].duplicated().sum())
print(order_products['add_to_cart_order'].duplicated().sum())
print(order_products['reordered'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     Se observa que no hay filas duplicadas pero sí tenemos datos duplicados en al parecer todas las columnas de este DataFrame, puede ser porque realmente lo que tenemos aquí son datos que se pueden repetir, el Id de orden, los productos, el orden en el que se agregan y si fueron reordenados o no, no parece ser preocupante el número tan grande de duplicados.

# ## Encuentra y elimina los valores ausentes
#
# Al trabajar con valores duplicados, pudimos observar que también nos falta investigar valores ausentes:
#
# * La columna `'product_name'` de la tabla products.
# * La columna `'days_since_prior_order'` de la tabla orders.
# * La columna `'add_to_cart_order'` de la tabla order_productos.

# ### `products` data frame

# In[23]:


# Encuentra los valores ausentes en la columna 'product_name'
display(products[products['product_name'].isna()])


# Describe brevemente cuáles son tus hallazgos.
#
#
#     Son un total de 1258 registros de valores ausentes en la columna de poduct_name, esto quiere decir que no contamos con el nombre del producto para estos registros.

# In[24]:


#  ¿Todos los nombres de productos ausentes están relacionados con el pasillo con ID 100?
product_name_nan = products[products['product_name'].isna()]
print("Los nombres de productos ausentes se encuentran relacionados con el/los siguiente(s) pasillo(s):",
      product_name_nan['aisle_id'].unique())


# Describe brevemente cuáles son tus hallazgos.
#
#     Después de crear un filtro por aquellas filas donde 'product_name' es NaN, se calcula los valores únicos en la columna de pasillos, para saber si solo aparece el pasillo con ID 100. En efecto,  parece ser que los datos ausentes están relacionado con el pasillo ID 100

# In[25]:


# ¿Todos los nombres de productos ausentes están relacionados con el departamento con ID 21?
product_name_nan = products[products['product_name'].isna()]
print("Todos los nombres de los productos ausentes estan relacionados con el/los siguiente(s) departamento(s):",
      product_name_nan['department_id'].unique())


# Describe brevemente cuáles son tus hallazgos.
#
#     Repitiendo la misma metodología, obtuvimos que el pasillo que se repite en el filtro de 'product_name' por NaN, es el pasillo con ID 21.

# In[26]:


# Usa las tablas department y aisle para revisar los datos del pasillo con ID 100 y el departamento con ID 21.
display(aisles.loc[aisles['aisle_id'] == 100])
print()
display(departments.loc[departments['department_id'] == 21])


# Describe brevemente cuáles son tus hallazgos.
#
#     Hemos encontrado que los registros tanto para el pasillo con ID 100 como para el departamento con ID 21, los datos han sido declarado como faltantes en los otros 2 DataFrames. Por lo tanto, se entiende que tengamos NaN en estas mismas columnas en el DataFrame de Products.

# In[27]:


# Completa los nombres de productos ausentes con 'Unknown'
products['product_name'] = products['product_name'].fillna('Unknown')


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     Al comprobar que los datos ausentes de 'product_name' estaban relacionados con el pasillo ID 100 y con el depto ID 21 y que en ambos el tenian como valor missing, podemos cambiar tranquilamente los datos faltantes de 'product_name'.

# ### `orders` data frame

# In[28]:


# Encuentra los valores ausentes
print(orders.isna().sum())
print()
display(orders[orders['days_since_prior_order'].isna()])


# In[29]:


# ¿Hay algún valor ausente que no sea el primer pedido del cliente?
days_since_prior_order_nan = orders[orders['days_since_prior_order'].isna()]
print("El los valores ausentes, concuerda con que es el pedido #:",
      days_since_prior_order_nan['order_number'].unique())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     El valor unico en 'order_number' es el 1, esto quiere decir que es el primer pedido del cliente, por lo tanto, es normal que en 'days_since_prior_order' aparezca como NaN

# ### `order_products` data frame

# In[30]:


# Encuentra los valores ausentes
print(order_products.isna().sum())


# In[31]:


# ¿Cuáles son los valores mínimos y máximos en esta columna?
print("Los valores mínimos del data frame 'order_products'es de:",
      order_products['add_to_cart_order'].min())
print()
print("Los valores máximos del data frame 'order_products'es de:",
      order_products['add_to_cart_order'].max())


# Describe brevemente cuáles son tus hallazgos.
#
#     Vemos que la columna add_to_cart_order es la que tiene datos ausentes en el DataFrame de order_products. Recordando que esta columna nos da el orden secuencial en el que se añadió cada artículo en el carrito, vemos que el mínimo efectivamente es 1, indicando que este es el producto que se agergó primeramente, mostrandonos un máximo de 64, siendo que en algún momento algun usuario agergó hasta 64 artículos.

# In[32]:


# Guarda todas las IDs de pedidos que tengan un valor ausente en 'add_to_cart_order'
id_order_nan = order_products[order_products['add_to_cart_order'].isna(
)]['order_id']
print(id_order_nan)


# In[33]:


# ¿Todos los pedidos con valores ausentes tienen más de 64 productos?
# Agrupa todos los pedidos con datos ausentes por su ID de pedido.
# Cuenta el número de 'product_id' en cada pedido y revisa el valor mínimo del conteo.

orders_with_nan = order_products[order_products['order_id'].isin(id_order_nan)]
order_id_counts = orders_with_nan['order_id'].value_counts()
print("El valor mínimo de productos agregados a una orden de compra y que su valor en la columna 'add_to_cart_order' es de:", order_id_counts.min())


# Describe brevemente cuáles son tus hallazgos.
#
#     Todos los valores ausentes en la columna add_to_cart_order se debe a que sobrepasan el número 64, que pareciera ser una limitante del sistema. Esto es que si un cliente agrega más de 64 artículos, el registro se transforma a un NaN.

# In[34]:


# Remplaza los valores ausentes en la columna 'add_to_cart? con 999 y convierte la columna al tipo entero.
order_products['add_to_cart_order'] = order_products['add_to_cart_order'].fillna(
    999)


# Describe brevemente tus hallazgos y lo que hiciste con ellos.
#
#     En este dataframe donde encotnraremos un importante número de duplicados, pero es normal por la misma naturaleza de los datos que aloja. Por otra parte encotnrarmos que el sistema que creó estos datos tiene una limitante a un máximo de 64 como orden secuencial en el que se añade cada artículo en el carrito, cuando se supera esta cantidad, se generará un ausente con NaN.

# ## Conclusiones
#
#     En cada 3 de los 5 DataFrames hemos tratado los datos debido a que tenian datos ausentes, pero en cada caso se pudo encontrar una razón lógica para ello. Hemos arreglado los datos que requerian algún tipo de ajuste.
#
#     En departments y aisles fueron DataFrames que no tuvieron tema de duplicados ni datos ausentes.
#     En el DataFrame orders hubo datos ausentes en days_since_prior_order, era debido a que en era el primer pedido del cliente por lo que no había como calcular lo ubicado en esta columna.
#     En el DataFrame products, tuvimos duplicados en en la columna de 'product_name', la gran parte de eran datos NaN, que estaban relacionados con el pasillo ID 100 y el pasillo 21, donde realmente el nombre de producto era missing, eso explicaba los datos NaN, por lo que se sustitullo esto por "Unknown".
#     En el DataFrame order_products teniamos datos ausentes en la columna de add_to_cart_order, debido a que al parecer el programa generador de datos, tiene una limitante a 64, cuando este número se superaba, la respuesta era un NaN. Esto se cambia por un 999.
#

# In[ ]:


# # Paso 3. Análisis de los datos
#
# Una vez los datos estén procesados y listos, haz el siguiente análisis:

# # [A] Fácil (deben completarse todos para aprobar)
#
# 1. Verifica que los valores en las columnas `'order_hour_of_day'` y `'order_dow'` de la tabla `orders` sean sensibles (es decir, `'order_hour_of_day'` va de 0 a 23 y `'order_dow'` va de 0 a 6).
# 2. Crea un gráfico que muestre el número de personas que hacen pedidos dependiendo de la hora del día.
# 3. Crea un gráfico que muestre qué día de la semana la gente hace sus compras.
# 4. Crea un gráfico que muestre el tiempo que la gente espera hasta hacer su siguiente pedido, y comenta sobre los valores mínimos y máximos.

# ### [A1] Verifica que los valores sean sensibles

# In[35]:


hour_of_day = (orders['order_hour_of_day'] >= 0) & (
    orders['order_hour_of_day'] <= 23)
count_true = hour_of_day.sum()
print("La cantidad de datos en esta columna es de", hour_of_day.count(),
      "y el total de datos que están dentro de los parámetros indicados son:", count_true)


# In[36]:


days_week = (orders['order_dow'] >= 0) & (orders['order_dow'] <= 6)
dw_count_true = days_week.sum()
print("La cantidad de datos en esta columna es de", days_week.count(),
      "y el total de datos que están dentro de los parámetros indicados son:", dw_count_true)

#
#     Podemos concluir que los datos son sencibles y estan dentro de los parámentros solicitados. Tanto order_hour_of_day va de 0 a 23 hrs y order_dow va de 0 a 6.

# ### [A2] Para cada hora del día, ¿cuántas personas hacen órdenes?

# In[37]:


# Crea un gráfico que muestre el número de personas que hacen pedidos dependiendo de la hora del día.
hourly_order = orders['order_hour_of_day'].value_counts().sort_index()
hourly_order.plot(x=hourly_order.index,
                  y=hourly_order.values,
                  kind='bar',
                  title='Orders by hours',
                  xlabel='Hours',
                  ylabel='Orders',
                  figsize=(10, 5),
                  grid=True)


plt.show()

#
#     Después de revisar la gráfica de ordenes realizada según el horario, vemos que las hora con mayor número de ordenes realizadas va desde las 10 am hasta las 4 pm, antes y después de estas horas, las ordenes de compra no se encuentran en sus máximos, siendo la madrugada desde las 0 horas hasta las 6 am el periodo donde menos ordenes se realiza.

# ### [A3] ¿Qué día de la semana compran víveres las personas?

# In[38]:


# Crea un gráfico que muestre qué día de la semana la gente hace sus compras
shopping_day = orders['order_dow'].value_counts().sort_index()
days_of_week = ['Sunday', 'Monday', 'Tuesday',
                'Wednesday', 'Thursday', 'Friday', 'Saturday']
shopping_day.index = days_of_week

shopping_day.plot(x=days_of_week,
                  y=shopping_day.values,
                  kind='bar',
                  title='Purchases per day',
                  xlabel='Day',
                  ylabel='Orders',
                  figsize=(10, 5),
                  grid=True)

plt.show()


#
#     Se observa que los días predilectos para hacer el mayor número de órdenes es el domingo, seguido del día lunes. El resto de los días no representan el máximo de órdenes. Siendo los miércoles y jueves los días más bajos.


# ### [A4] ¿Cuánto tiempo esperan las personas hasta hacer otro pedido? Comenta sobre los valores mínimos y máximos.


# Crea un gráfico que muestre el tiempo que la gente espera hasta hacer su siguiente pedido, y comenta sobre los valores mínimos y máximos.
days_for_last_order = orders['days_since_prior_order'].value_counts(
).sort_index()
days_for_last_order.index = days_for_last_order.index.astype(int)
selected_data = days_for_last_order.loc[1:30]

selected_data.plot(x=selected_data.index,
                   y=selected_data.values,
                   kind='bar',
                   title='Days since prior order',
                   xlabel='Days',
                   ylabel='Orders',
                   figsize=(10, 5),
                   grid=True)

plt.show()

print('En un periodo de 30 días, el mínimo de pedidos realizados nuevamente por un cliente desde su pedido anterior fue de', selected_data.min(),
      'pedidos. En el mismo periordo, el máximo de pedidos realizados nuevamente por un cliente desde su pedido anterior fue de', (selected_data.max()), 'pedidos.')


#
#     Acorde a los datos, los clientes suelen volver a realizar un pedido a los 7 días ó 30 días, siendo estos dos datos nuestros puntos más altos. Esto podría indicarnos que los clientes planean sus compras de manera semanal o mensual. Valdría la pena revisar el tipo de productos que se piden dependiendo de este periodo de tiempo. De igual manera vale la pena mencionar que la tenemos una actividad media en los primeros días después de su último pedido, siendo importante los días 2 al 6 después de su orden anterior. Y tenemos menor numero de ordenes entre los días 15 y 29 después de su ultima compra.

# # [B] Intermedio (deben completarse todos para aprobar)
#
# 1. ¿Existe alguna diferencia entre las distribuciones `'order_hour_of_day'` de los miércoles y los sábados? Traza gráficos de barra de `'order_hour_of_day'` para ambos días en la misma figura y describe las diferencias que observes.
# 2. Grafica la distribución para el número de órdenes que hacen los clientes (es decir, cuántos clientes hicieron solo 1 pedido, cuántos hicieron 2, cuántos 3, y así sucesivamente...).
# 3. ¿Cuáles son los 20 principales productos que se piden con más frecuencia (muestra su identificación y nombre)?

# ### [B1] Diferencia entre miércoles y sábados para  `'order_hour_of_day'`. Traza gráficos de barra para los dos días y describe las diferencias que veas.

# In[40]:


wednesday_orders = orders[orders['order_dow'] == 3]
saturday_orders = orders[orders['order_dow'] == 6]


# In[41]:


data_wednesday = wednesday_orders['order_hour_of_day'].value_counts(
).sort_index()
data_saturday = saturday_orders['order_hour_of_day'].value_counts(
).sort_index()


# In[42]:


data_wednesday_saturday = pd.concat([data_wednesday, data_saturday], axis=1)
data_wednesday_saturday.columns = ['wednesday', 'saturday']

display(data_wednesday_saturday)


data_wednesday_saturday.plot(kind='bar',
                             figsize=(10, 5),
                             title='Orders by Time of Day (Wednesday vs. Saturday)',
                             xlabel='Hours',
                             ylabel='Orders',
                             alpha=0.8,
                             grid=True)

plt.legend()
plt.show()


#
#     En las horas más bajas de ordenes que van de las 23 a las 6 hrs, se generan más ordenes los sábados que los miércoles, con excepción de las 6 hrs, donde se hacen más ordenes en miércoles.
#
#     De 6 a las 10 hrs, los miércoles suelen hacerse más ordenes que los sábados.
#     Pero el sábado se realizan más ordenes de 11 a 16 hrs, que el miércoles.
#
#     De 15 a 22 hrs, la cantidad de pedidos entre los dos días es muy similar, siendo un poco más alto para los miércoles que los sábados.
#
#     Para el miércoles la mayor parte de las ordenes se realizan entre las 10 y 11hrs  y entre las 15 y 16 hrs.
#
#     Para el sábado la mayor parte de las ordenes se realizan de las 11 a las 16 hrs.
#

# ### [B2] ¿Cuál es la distribución para el número de pedidos por cliente?

# In[44]:


orders['order_number'].plot(kind='hist', bins=20)
plt.xlabel('Number of Orders per Customer')
plt.title('Distribution of the Number of Orders per Customer')

plt.show()


#
#           La mayor parte de ordenes por cliente se encuentra concentrado entre los primeros 5 a 10 órdenes, teniendo una disminución progresiva conforme el número de órdenes crece, esto podría deberse a distintas razones que no podemos determinar en este gráfico.  Podríamos revisar el tiempo que tiene registrado el cliente en la plataforma y cuantas ordenes ha realizado en ese periodo de tiempo.

# ### [B3] ¿Cuáles son los 20 productos más populares (muestra su ID y nombre)?

# In[45]:


# ¿Cuáles son los 20 principales productos que se piden con más frecuencia (muestra su identificación y nombre)?
product_counts = order_products['product_id'].value_counts()
top_20_products = product_counts.head(20)

products_names = products[['product_id', 'product_name']]
print(products_names)

top_20_products_merge = pd.merge(
    top_20_products, products_names, left_index=True, right_on='product_id')


# In[46]:


top_20_products_name = top_20_products_merge[[
    'product_id', 'product_name']].reset_index(drop=True)
print(top_20_products_name)


#
#     Acorde a la información extraída, los principales productos que se ordenan son perecederos, en su mayoría, orgánicos. Por lo que se podría entender que se pida con mayor frecuencia, siendo perecederos, son productos que no se pueden comprar a granel, pues aún con un muy buen almacenaje, terminarán por ser no consumibles después de un corto tiempo. Por lo que los clientes pedirán estos productos en un periodo de tiempo más corto, por lo que su aparición en la lista de ordenes será más frecuente.
#
#     El hecho de que sean productos orgánicos, podría representar el peso que le dan los consumidores a que se tengan dentro de stock, productos que garanticen un origen más limpio de pesticidas y/o substancias que podrían ser consideradas no deseables por parte de los clientes.
#
#

# # [C] Difícil (deben completarse todos para aprobar)
#
# 1. ¿Cuántos artículos suelen comprar las personas en un pedido? ¿Cómo es la distribución?
# 2. ¿Cuáles son los 20 principales artículos que vuelven a pedirse con mayor frecuencia (muestra sus nombres e IDs de los productos)?
# 3. Para cada producto, ¿cuál es la tasa de repetición del pedido (número de repeticiones de pedido/total de pedidos?
# 4. Para cada cliente, ¿qué proporción de los productos que pidió ya los había pedido? Calcula la tasa de repetición de pedido para cada usuario en lugar de para cada producto.
# 5. ¿Cuáles son los 20 principales artículos que la gente pone primero en sus carritos (muestra las IDs de los productos, sus nombres, y el número de veces en que fueron el primer artículo en añadirse al carrito)?

# ### [C1] ¿Cuántos artículos compran normalmente las personas en un pedido? ¿Cómo es la distribución?

# In[47]:


order_item_sum = order_products.groupby('order_id')['product_id'].count()


# In[48]:


print('El mínimo de artículos que se han solicitado en un pedido es de',
      order_item_sum.min(), 'artículo.')
print()
print('El máximo de artículos que se han solicitado en un pedido es de',
      order_item_sum.max(), 'artículos.')


# In[49]:


order_item_sum.plot(kind='hist',
                    figsize=(10, 5),
                    bins=40,
                    grid=(True))

plt.xlabel('Number of items per order')
plt.title('Distribution of the number of items per order')
plt.ylabel('Number of orders')

plt.show()


#
#     Al observar el histograma, podemos indicar que la gran parte de los pedidos se realizan por “pocos” artículos. La mayoría de ordenes va desde solicitar 1 solo producto hasta aproximadamente 21 artículos. Son pocas las ordenes que solicitan más allá de 40 artículos.

# ### [C2] ¿Cuáles son los 20 principales artículos que vuelven a pedirse con mayor frecuencia (muestra sus nombres e IDs de los productos)?

# In[50]:


product_reordered = order_products[order_products['reordered']
                                   == 1]['product_id'].value_counts()
product_reordered_top_20 = product_reordered.head(20)


# In[51]:


product_reordered_top_20_merge = pd.merge(
    product_reordered_top_20, products_names, left_index=True, right_on='product_id')


# In[52]:


top_20_reordered_products = product_reordered_top_20_merge[[
    'product_id', 'product_name']].reset_index(drop=True)
print(top_20_reordered_products)


#
#   Vemos una similitud muy cercana a la lista de productos más populares, y no es de extrañar que sean los productos que se vuelven a pedir con mayor frecuencia. Esto podría ser debido a que son productos perecederos. Razón que habíamos explicado anteriormente.
#
# 	De igual manera, vemos un ligero cambio de posiciones de algunos productos, pero no es de gran relevancia.
#

# ### [C3] Para cada producto, ¿cuál es la proporción de las veces que se pide y que se vuelve a pedir?


product_info = order_products.groupby('product_id').size().reset_index()
product_info.columns = ['product_id', 'total_orders']

reordered_info = order_products[order_products['reordered'] == 1]
reordered_counts = reordered_info.groupby('product_id').size().reset_index()
reordered_counts.columns = ['product_id', 'reordered']

product_orders = pd.merge(
    product_info, reordered_counts, on='product_id', how='left')

product_orders['reordered'] = product_orders['reordered'].fillna(0)

product_orders['repurchase ratio'] = product_orders['reordered'] / \
    product_orders['total_orders']

product_orders_names = pd.merge(
    product_orders, products_names, on='product_id', how='left')

print(product_orders_names[['product_name', 'repurchase ratio']])


#
#     Sin duda es un listado muy interesante, nos deja ver que mu probablemente el usuario esté contento con ese producto y por esa razón lo vuelve a pedir. Nos da muchas oportunidades de acción, tanto para saber qué es lo que no puede faltar en el stock, como para proponer a las marcas algún descuento para que el usuario pueda conocer otros productos y diversificar sus elecciones.

# ### [C4] Para cada cliente, ¿qué proporción de sus productos ya los había pedido?


user_product_merge = pd.merge(orders, order_products, on='order_id')
user_orders = user_product_merge.groupby(
    'user_id')['product_id'].count().reset_index()
user_orders.columns = ['user_id', 'total_products']

reordered_products = user_product_merge[user_product_merge['reordered'] == 1]
reordered_products_count = reordered_products.groupby(
    'user_id')['product_id'].count().reset_index()
reordered_products_count.columns = ['user_id', 'reordered_products']

user_products_info = pd.merge(
    user_orders, reordered_products_count, on='user_id', how='left')
user_products_info['reordered_products'] = user_products_info['reordered_products'].fillna(
    0)

user_products_info['reorder_ratio'] = user_products_info['reordered_products'] / \
    user_products_info['total_products']

print(user_products_info[['user_id', 'reorder_ratio']])


#     Nos permite conocer el comportamiento de compra en el usuario y tomar alguna estrategia dependiendo del cumplimiento de ciertas pautas. Podríamos complementar el estudio del usuario con otros datos, para así elegir estrategias para la mejor experiencia de usuario y por supuesto mantener o incrementar las ventas.

# ### [C5] ¿Cuáles son los 20 principales artículos que las personas ponen primero en sus carritos?


# ¿Cuáles son los 20 principales artículos que la gente pone primero en sus carritos (muestra las IDs de los productos, sus nombres, y el número de veces en que fueron el primer artículo en añadirse al carrito)?
first_added_products = order_products[order_products['add_to_cart_order'] == 1]

Top_20_products_first_added = first_added_products['product_id'].value_counts(
).reset_index().head(20)
Top_20_products_first_added.columns = ['product_id', 'count']

# how='left')
Top_20_products_first_added_with_names = pd.merge(
    Top_20_products_first_added, products_names, on='product_id', how='left')

print(Top_20_products_first_added_with_names[[
      'product_id', 'product_name', 'count']])


#     Con base a los datos obtenidos, vemos productos ya conocidos, los perecederos, en su mayoría frutas, verduras y leches. Pero en esta lista obtenemos algo diferente frente a los 20 productos más re comprados y de los más populares, tenemos que dentro de los 20 productos que se colocan primero en el carrito de compras ahora también se une a la lista las bebidas gasificadas. Se entiende que los perecederos sean de los primeros al ser colocado al carrito de compra, puesto que podrían ser de los primeros artículos que se le hayan acabado al usuario o hayan terminado con su tiempo de vida, de igual manera podríamos ver si esos artículos fueron ingresados manualmente por el usuario o bien se encuentran en una “lista de favoritos” que resurtirán siempre que el cliente así lo desee. El hecho de encontrar bebidas gasificadas dentro de este listado, es un comportamiento interesante, sin embargo, no podrían darse conclusiones certeras del porqué estos artículos aparecen en este listado con la información que tenemos a la mano.
