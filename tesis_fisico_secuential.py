#!/usr/bin/env python
# coding: utf-8
"""
Exportado desde `tesis_fisico_secuential.ipynb` (jupyter nbconvert --to script).
Las celdas markdown del notebook quedan como comentarios (#).
Rutas relativas al directorio de este archivo al ejecutarlo.
"""

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
import os

os.chdir(_SCRIPT_DIR)

try:
    from IPython.display import display
except ImportError:
    def display(obj):  # noqa: A001
        print(obj)

# # 1. Predecir la trayectoria e intensidad de huracanes aplicando modelos de aprendizaje automático, con enfoque en aquellos que amenacen las Islas de San Andrés y Providencia, Caribe colombiano

# ## 1.1 Objetivo General
# Predecir la trayectoria e intensidad de huracanes aplicando modelos de aprendizaje automático, con enfoque en aquellos que amenacen las Islas de San Andrés y Providencia, Caribe colombiano.
# 

# ## 1.2 Objetivos Específicos 
# 
# * Realizar extracción, transformación y carga (ETL) de datos mediante reglas para limpiar y organizar datos en bruto y prepararlos para el almacenamiento, el análisis de datos e implementación de modelos de ML para la predicción de la trayectoria e intensidad de huracanes.
# 
# * Realizar un Análisis Exploratorio de Datos (EDA) con base en la matriz de datos elaborada, como mecanismo para la detección de patrones importantes, datos faltantes y atípicos a tratar, así como también, el preprocesamiento de cada una de las variables de entrada de los modelos de aprendizaje automático, para la predicción de la trayectoria e intensidad de huracanes. 
# 
# * Implementar y evaluar distintos modelos estadísticos y de aprendizaje automático para la predicción de la trayectoria e intensidad de huracanes, mediante el uso de métricas de evaluación y validación cruzada adecuadas, así como métodos de optimización para la selección de parámetros.
# 
# 
# 
# 

# Librerías

# In[352]:


import pandas as pd

# Mostrar todas las columnas sin truncar
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


# In[353]:


from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import joblib


# # 2. ETL 

# In[354]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from geopy.distance import geodesic


# In[355]:


df = pd.read_csv('hurdat2-1851-2023-051124.txt', sep= ',', names=['Date', 'Hour','RecIdentifier','type_storm','Latitude',
                                                              'Longitude','Max_wind','MinPress','NE34','SE34','SW34','NW34',
                                                              'NE50','SE50','SW50','NW50','NE64','SE64','SW64','NW64','Rad_Max_Wind'])


# In[356]:


df.head(5)


# ## 2.1 organización del dataframe

# In[357]:


# Agregar nuevas columnas para almacenar los valores extraídos
df["Code_storm"] = np.nan
df["Name_storm"] = np.nan
df["Trackets"] = np.nan

# Variables temporales para almacenar los valores
current_code, current_name, current_rec = None, None, None

for index, row in df.iterrows():
    if str(row["Date"]).startswith("AL"):
        current_code, current_name, current_rec = row["Date"], row["Hour"], row["RecIdentifier"]

    df.at[index, "Code_storm"] = current_code
    df.at[index, "Name_storm"] = current_name
    df.at[index, "Trackets"] = current_rec


# Eliminar filas donde la columna "Date" comienza con "AL"
df = df[~df["Date"].astype(str).str.startswith("AL")].reset_index(drop=True)

# Reubicar las columnas code_storm, name_storm y recIdentifier al inicio
column_order = ["Code_storm", "Name_storm", "Trackets"] + [col for col in df.columns if col not in ["Code_storm", "Name_storm", "Trackets"]]
df_procesado = df[column_order]


# In[358]:


df.head(3)


# ## 2.2 ajustando las coordenadas a un formato numérico

# In[359]:


# Extraer el valor numérico y convertirlo a formato numérico
df_procesado['Lat_N'] = pd.to_numeric(df_procesado['Latitude'].str.extract(r'(\d+\.\d+)', expand=False))
df_procesado['Lon_W'] = pd.to_numeric(df_procesado['Longitude'].str.extract(r'(\d+\.\d+)', expand=False))


# Eliminar las columnas originales 'Latitude' y 'Longitude'
df_procesado.drop(['Latitude', 'Longitude'], axis=1, inplace=True)


# ## 2.3 ajuste de los targets de la variable Type_storm y el Codigo de tormenta (Code_storm)

# In[360]:


# Cambia la columna 'nombre_columna' a tipo string
df_procesado['type_storm'] = df_procesado['type_storm'].astype(str) # Para asegurar que sea tipo String

df_procesado['type_storm'] = df_procesado['type_storm'].str.strip().str.upper() # Asegurar que no haya espacios en los targets y todo sean en MAYUS

df_procesado['Code_storm'] = df_procesado['Code_storm'].astype(str)
df_procesado['Code_storm'] = df_procesado['Code_storm'].str.strip().str.upper()


# ## 2.4 Ajuste en los datos de fecha

# In[361]:


df_procesado['Date'] = pd.to_datetime(df_procesado['Date'], format='%Y%m%d', errors='coerce')

#Convertir la columna en Date en datetime
#hrc['Date'] = pd.to_datetime(hrc['Date'])

# Crear nuevas columnas para año, mes y día
df_procesado['year'] = df_procesado['Date'].dt.year
df_procesado['month'] = df_procesado['Date'].dt.month
df_procesado['day'] = df_procesado['Date'].dt.day

# Eliminar la columna original "Date"
df_procesado.drop('Date', axis=1, inplace=True)


# In[362]:


# Reorganizar las columnas: mover 'Año', 'Mes' y 'Día' a las posiciones 3, 4 y 5
columnas = list(df_procesado.columns)
# Insertar las nuevas columnas en las posiciones deseadas
columnas_nuevas = ['year', 'month', 'day']
for i, columna in enumerate(columnas_nuevas):
    columnas.remove(columna)  # Eliminar la columna de su posición original
    columnas.insert(3 + i, columna)  # Insertar en la nueva posición (3, 4 y 5)

# Reordenar el DataFrame
df_procesado = df_procesado[columnas]


# In[363]:


# Conversión y separación de la hora y los minutos
df_procesado['Hour'] = df_procesado['Hour'].astype(int)  # Asegurar enteros
df_procesado['hour'] = df_procesado['Hour'] // 100  # División entera para las horas
#df_procesado['minuto'] = df_procesado['Hour'] % 100  # Resto para los minutos

df_procesado.drop('Hour', axis=1)


# In[364]:


df_procesado.drop('Hour', axis=1, inplace=True) # Eliminar la columna Hour


# In[365]:


# Move 'hora' and 'minuto' columns to positions 7 and 8
cols = list(df_procesado.columns)
cols.insert(7, cols.pop(cols.index('hour')))
#cols.insert(8, cols.pop(cols.index('minuto')))
df_procesado = df_procesado[cols]

df_procesado.head(3)


# In[366]:


df_procesado['fecha'] = pd.to_datetime(
    df_procesado[['year', 'month', 'day', 'hour']]
    )


# In[367]:


df_procesado


# ## Se verifican datos duplicados y se reemplaza -99 por NaN

# In[368]:


duplicados = df_procesado.duplicated()
print(duplicados.sum())


# In[369]:


# Reemplazar -999 por NaN en todo el DataFrame
df_procesado.replace(-999, np.nan, inplace=True)
df_procesado.replace(-99, np.nan, inplace=True)


# ## Eliminación de la variable Rad_Max_wind
# 
# Los valores de esta variable solo han sido rastreados de manera precisa a partir de 2021. Por ende, es la variable que cuenta con menor cantidad de datos, teniendo 1331 registros o filas, lo cual representa el 2.46% del set de datos total. Por este motivo, se decidió borrar esta variable del conjunto de datos.  
# 
# Esta variable corresponde a los Radio de Vientos Máximos: Es la distancia desde el centro de un ciclón tropical hasta la ubicación de los vientos máximos del ciclón. En huracanes bien desarrollados, el radio de vientos máximos generalmente se encuentra en el borde interior de la pared del ojo (https://www.nhc.noaa.gov/aboutgloss.shtml#a)
# 

# In[370]:


# Eliminar la columna 'Rad_Max_Wind'
df_procesado.drop('Rad_Max_Wind', axis=1, inplace=True)


# RecIdentifier y Trackets son variables descriptivas asociadas a la toma de información, número de tracks registrados y ...

# In[371]:


df_procesado.drop(columns=['RecIdentifier', 'Trackets'], inplace=True)


# In[372]:


df_ETL = df_procesado


# # 3. EDA #1

# ## 3.1 Datos faltantes

# In[373]:


df_ETL.head(3)


# In[374]:


# Procederemos a ver Missing values a través de msno.matrix
import missingno as msno
msno.matrix(df_ETL)


# In[375]:


df_ETL.isnull().sum()


# Los datos faltantes corresponden a lo siguiente:
# 
# * "Minpress", "solo se incluían valores si había una observación específica que pudiera utilizarse explicitamente". A partir de 1979, se analizaba e incluía toda medición incluso si no fuese una medición in situ especifica.
# 
# 
# 
# * NE34 to NW64: Estos valores están disponibles a partir de 2004 con una resolución de hasta 5 millas nauticas. Las columnas de NE34 a NW64 hace referencia al Wind Radii o *Radios de viento*. "Se refiere a la distancia desde el centro de un huracán hasta donde los vientos alcanzan ciertas velocidades específicas, medidas en diferentes cuadrantes (noreste, sureste, suroeste y noroeste). Se utilizan para describir el tamaño del huracán y la extensión de los vientos peligrosos en diferentes direcciones". La distribución de los vientos alrededor de un huracán no es simétrica, entre mas lejos del ojo del huracán los vientos disminuyen y los vientos del lado derecho de la tormenta son mas fuertes en relación al lado izquierdo (De acuerdo a la dirección) (George & Gray, 1976).         
# *NOTA*:Si bien es cierto que se observaron patrones importantes y que coinciden con la literatura de que son características importantes en cada estado de tormenta, la realidad es que la medición y obtención de estos radios en tiempo real no es práctica lo que entorpece el funcionamiento de un modelo operativo en tiempo real, además dentro del set de datos de HURDAT2 solo se tienen datos desde el 2004 hasta el año presente, y aunque se exploró la estrategia de imputar datos, esta no fue finalmente realizada ni incluida en el trabajo, ya que la naturaleza de la variable se presta para un trabajo adicional enfocado solamente en los radios del viento de los huracanes, por ello, se recomienda tenerla en cuenta para futuros trabajos o incluso explorar la predicción de esta variable importante en los huracanes
# 
# 
# 
# 
# 
# 
# Gray, W. M., & Shea, D. J. (1976). Data summary of NOAA's hurricane inner-core radial leg flight penetrations 1957-1967, and 1969. Department of Atmospheric Sciences, Colorado State University.

# ## 3.2 Exploración de los Radios del viento

# In[376]:


def find_negative_wind_radii(df):

  negative_values = {}
  for col in df.columns:
    if 'NW' in col or 'NE' in col or 'SE' in col or 'SW' in col:
      negative_rows = df[df[col] < 0]
      if not negative_rows.empty:
        negative_values[col] = negative_rows.index.tolist()
  return negative_values


negative_wind_radii = find_negative_wind_radii(df_ETL)

if negative_wind_radii:
  print("Valores negativos en las columnas NW, NE, SE y SW y sus posiciones:")
  for col, rows in negative_wind_radii.items():
    print(f"Columna: {col}")
    print(f"Filas con valores negativos: {rows}")
else:
  print("No se encontraron valores negativos en las columnas NW, NE, SE y SW.")


# In[377]:


def boxplots_windrad(dataframe):
    """
    Genera gráficos de caja (boxplot) de NW, NE, SE y SW organizados en una cuadrícula de 3x3
    para cada tipo de tormenta en el dataframe, incluyendo los tres campos de viento (64, 50, 34).
    Los outliers tienen el mismo color que las cajas de cada campo de viento y no tienen borde negro.

    :param dataframe: DataFrame con los datos de tormentas.
    """
    counter = 0

    # Definir colores personalizados para cada velocidad de viento
    colores = {'64': 'blue', '50': 'green', '34': 'red'}

    for storm_type in dataframe['type_storm'].unique():
        if counter % 9 == 0:
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            fig_index = 0

        row, col = divmod(fig_index, 3)

        subset = dataframe[dataframe['type_storm'] == storm_type]

        # Transformar los datos usando melt
        data_melted = subset.melt(value_vars=[f'NW64', f'NE64', f'SE64', f'SW64',
                                              f'NW50', f'NE50', f'SE50', f'SW50',
                                              f'NW34', f'NE34', f'SE34', f'SW34'],
                                  var_name='Sector_Viento', value_name='Valor')

        # Extraer información de la columna Sector_Viento
        data_melted['Sector'] = data_melted['Sector_Viento'].str[:2]  # NW, NE, SE, SW
        data_melted['Viento'] = data_melted['Sector_Viento'].str[2:]  # 64, 50, 34

        # Crear el boxplot con outliers personalizados
        boxplot = sns.boxplot(
            x='Sector',
            y='Valor',
            hue='Viento',
            data=data_melted,
            ax=axes[row, col],
            flierprops={'marker': 'o', 'markersize': 5, 'linestyle': 'none'}  # Configurar outliers
        )

        # Personalizar los colores de outliers según su grupo de viento
        for artist, label in zip(boxplot.artists, sorted(colores.keys())):
            color = colores[label]
            artist.set_facecolor(color)  # Color de la caja
            artist.set_edgecolor(color)  # Color del borde de la caja

            # Personalizar los outliers
            for line in boxplot.lines:
                if 'fliers' in line.get_label():
                    line.set_markerfacecolor(color)  # Misma tonalidad de la caja
                    line.set_markeredgecolor('none')  # Sin borde negro

        axes[row, col].set_title(f'Boxplot para {storm_type}')
        axes[row, col].set_xlabel('Sector')
        axes[row, col].set_ylabel('Valores')

        fig_index += 1
        counter += 1

        if fig_index == 9 or counter == len(dataframe['type_storm'].unique()):
            plt.tight_layout()
            plt.show()



sns.reset_defaults()
boxplots_windrad(df_ETL)


# Estas cajas de bigote muestran la distribución de los datos en cada cuadrante de la tormenta (NE, NW, SE, SW). Cada cuadrante puede presentar hasta 3 radios de vientos, en azul (64kt), naranja (50kt) y verde (34kt). Las gráficas de boxplot fueron realizadas para cada estado de la tormenta (WV, SD, SS, DB, LO, TD, EX, TS, HU)
# 
# * DB – Perturbación (de cualquier intensidad)
# * WV – Onda tropical (de cualquier intensidad)
# * LO – Baja no ciclónica (de cualquier intensidad)
# * TD – Depresión tropical (< 34 nudos)
# * SD – Depresión subtropical (< 34 nudos)
# * TS – Tormenta tropical (34-63 nudos)
# * SS – Tormenta subtropical (> 34 nudos)
# * EX – Ciclón extratropical (de cualquier intensidad
# * HU – Huracán (> 64 nudos)
# 
# Se aprecian valores extremos en todas las cateegorias de Huracanes, además hay presencia de Radios del viento en LO, DB, y WV, lo cual puede parecer atípico en estos estados de tormenta, ya que son tormentas muy debiles como para tener una formación ciclonica y por ende radios de vientos, pero no son erroneos ya que un Huracan puede disminuir a un estado de tormenta muy debil como estos y de esta manera quedan remanentes de estos radios.
# 
# También, se puede observar que las bandas de 34, 50 y 64 nudos, se presentan especificamente en ciertas categorías de la tormenta, estando presente los 3 casos en la categoría de huracán (HU) y con la mayor distribución de valores (Km)
# 
# Se destaca también que la categoría Tormenta tropical (TS) registra valores de distancia para las bandas de 50 y 34 nudos solamente. Según (McKenzie, 2017) el radio del viento 64 está reservado para los ciclones tropicales con fuerza de huracán.
# 
# 
# 
# McKenzie III, T. B. (2017). A climatology of tropical cyclone size in the western North Pacific using an alternative metric (Master's thesis, The Florida State University).

# In[378]:


#Guardamos una copia de seguridad
df_EDA = df_ETL.copy()


# In[379]:


#Eliminar las columas NW, NE, SE, SW
df_EDA = df_EDA.drop(columns=['NW64', 'NE64', 'SE64', 'SW64', 'NW50', 'NE50', 'SE50', 'SW50', 'NW34', 'NE34', 'SE34', 'SW34'])
df_EDA.head()
df_EDA.info()


# ## 3.3 Agregar Categorias de Huracan & "type_storm" numérico

# * DB – Perturbación (de cualquier intensidad)
# * WV – Onda tropical (de cualquier intensidad)
# * LO – Baja no ciclónica (de cualquier intensidad)
# * TD – Depresión tropical (< 34 nudos)
# * SD – Depresión subtropical (< 34 nudos)
# * TS – Tormenta tropical (34-63 nudos)
# * SS – Tormenta subtropical (> 34 nudos)
# * EX – Ciclón extratropical (de cualquier intensidad
# * HU – Huracán (> 64 nudos)

# In[380]:


def clasificacion_huracan(df):
    '''
    64-82       Cat 1
    83-95       Cat 2
    96-113      Cat 3
    114-135     Cat 4
    Mayor 135 Cat 5
    '''
    condiciones = [
        (df['Max_wind'] >= 64) & (df['Max_wind'] <= 82),
        (df['Max_wind'] >= 83) & (df['Max_wind'] <= 95),
        (df['Max_wind'] >= 96) & (df['Max_wind'] <= 113),
        (df['Max_wind'] >= 114) & (df['Max_wind'] <= 135),
        (df['Max_wind'] > 135)
    ]

    categorias = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']

    # Crear una nueva columna en el DataFrame con las categorías de huracanes
    df['categoria_huracan'] = np.select(condiciones, categorias,default='Sin categoria')

    return df


# In[381]:


df_EDA = clasificacion_huracan(df_EDA)


# In[382]:


# Reemplazar 'HU' en 'type_storm' por el valor de 'categoria_huracan' de la misma fila
df_EDA.loc[df_EDA['type_storm'] == 'HU', 'type_storm'] = df_EDA.loc[df_EDA['type_storm'] == 'HU', 'categoria_huracan']

df_EDA.drop('categoria_huracan', axis=1, inplace=True)


# Las filas 25198 y 25199 presentaban categoría "HU" o sea Huracán. Sin embargo, la velocidad de viento máxima en estos dos registros fue de 60 nudos, lo cual está por debajo de los límites de la escala Saffir-Simpson que categoriza los estados Huracán de 1 a 5. Siendo 5 el mas fuerte. 
# 
# 
# Esto pudo deberse a un error al ingresar los datos por parte del NHC (Centro nacional de Huracanes de EE.UU) o que los vientos del viento de huracán se debilitaron pero se conservó el estado de la tormenta en "HU".
# 
# En cualquier caso, se decide eliminar estas dos filas para no generar ruido en los modelos de clasificación.

# In[383]:


df_EDA = df_EDA.drop([25198, 25199])


# A continuación procedemos a convertir las categorias a númericas de manera manual, quiere decir, asignarle valores especificos para cada estado de tormenta. 
# 
# Se usa el siguiente Mapeo, empezando el 1 con "TD" hasta 7 con "HU-Cat5" porque se usarán estas categorías para los modelos de clasificación de categorías. 
# 
# Mientras que las restantes, se les coloca números restantes en orden, pero estas categorías no serán usadas para entrenar los modelos. Porque lo ideal es que la operatividad de los modelos empiece desde el estado de Depresión Tropical (TD). Además de simplificar el problema a menos categorías. 

# In[384]:


# Mapeo manual
mapeo_target = {'TD': 1, 'TS': 2, 'Cat 1': 3, 'Cat 2': 4, 'Cat 3': 5, 'Cat 4': 6, 'Cat 5': 7,
                'DB': 13, 'WV': 12, 'LO': 11, 'SD': 10, 'SS': 9, 'EX': 8,}

# Aplicar el mapeo a la columna 'type_storm'
df_EDA['type_storm_n'] = df_EDA['type_storm'].map(mapeo_target)


# In[385]:


df_EDA.head()


# ## 3.4 Imputación de Presión Mínima "Minpress"

# In[386]:


# Eliminar filas con NaN
hrc_press = df_EDA.copy()
press_clean = hrc_press.dropna(subset=['MinPress', 'type_storm'])

# Gráficos de distribución por tipo de tormenta
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histogramas por tipo de tormenta
for storm_type in press_clean['type_storm'].unique():
    data = press_clean[press_clean['type_storm'] == storm_type]['MinPress']
    axes[0, 0].hist(data, alpha=0.7, label=storm_type, bins=20)

axes[0, 0].set_title('Distribución de MinPress por Tipo de Tormenta')
axes[0, 0].set_xlabel('MinPress (mb)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].legend()

# 2. Box plots
press_clean.boxplot(column='MinPress', by='type_storm', ax=axes[0, 1])
axes[0, 1].set_title('Box Plot de MinPress por Tipo de Tormenta')
axes[0, 1].set_xlabel('Tipo de Tormenta')
axes[0, 1].set_ylabel('MinPress (mb)')

# 3. Q-Q plots para normalidad
storm_types = press_clean['type_storm'].unique()
for i, storm_type in enumerate(storm_types[:4]):  # Solo primeros 4 tipos
    data = press_clean[press_clean['type_storm'] == storm_type]['MinPress']
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q Plot - {storm_type}')

# 4. Test de Kolmogorov-Smirnov por tipo
results = []
for storm_type in storm_types:
    data = press_clean[press_clean['type_storm'] == storm_type]['MinPress']
    if len(data) > 50:
        # Normalizar los datos
        data_norm = (data - data.mean()) / data.std()
        stat, p_value = stats.kstest(data_norm, 'norm')
        results.append({
            'Tipo': storm_type,
            'Estadístico': stat,
            'p-valor': p_value,
            'Normal': p_value > 0.05
        })

results_df = pd.DataFrame(results)
print("Test de Kolmogorov-Smirnov por tipo de tormenta:")
print(results_df)

# Gráfico de p-valores
axes[1, 1].bar(results_df['Tipo'], results_df['p-valor'])
axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
axes[1, 1].set_title('P-valores del Test K-S')
axes[1, 1].set_xlabel('Tipo de Tormenta')
axes[1, 1].set_ylabel('P-valor')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Resumen estadístico por tipo
print("\nEstadísticas descriptivas por tipo de tormenta:")
print(press_clean.groupby('type_storm')['MinPress'].describe())


# Gráificos solo para Categoría 5

# In[387]:


# Filtrar solo Cat 5 y eliminar NaN en MinPress
cat5_minpress = hrc_press[(hrc_press['type_storm'] == 'Cat 5') & (~hrc_press['MinPress'].isnull())]

# Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(cat5_minpress['MinPress'], dist="norm", plot=plt)
plt.title('Q-Q Plot de MinPress para Cat 5')
plt.xlabel('Cuantiles teóricos')
plt.ylabel('Cuantiles de la muestra')
plt.grid(alpha=0.3)
plt.show()


# In[388]:


# Filtrar solo Cat 5 y eliminar NaN en MinPress
cat5_minpress = hrc_press[(hrc_press['type_storm'] == 'Cat 5') & (~hrc_press['MinPress'].isnull())]

# Histograma
plt.hist(cat5_minpress['MinPress'], bins=10, color='salmon', edgecolor='black', alpha=0.8)
plt.title('Histograma de MinPress para Cat 5')
plt.xlabel('MinPress (mb)')
plt.ylabel('Frecuencia')
plt.grid(alpha=0.3)
plt.show()


# Sabiendo entonces que la distribución de los datos de MinPress no sigue una distribucion en normal en niguna de las categorias. Procedemos a usar la MEDIANA de cada categoria para imputar en los valores faltantes de MinPress. 
# 
# La categoria 5, sigue una distribución normal según el valor P del test K-S 

# In[389]:


# Ver valores faltantes antes de imputar
print("Valores faltantes antes de imputar:")
print(df_EDA['MinPress'].isnull().sum())

# Diccionario con las medianas por tipo de tormenta
medianas_por_tipo = {
    'Cat 1': 983.0,
    'Cat 2': 968.0,
    'Cat 3': 955.0,
    'Cat 4': 941.0,
    'Cat 5': 918.1, # Para esta categoria se usa el promedio y no la mediana.
    'DB': 1010.0,
    'EX': 995.0,
    'LO': 1009.0,
    'SD': 1008.0,
    'SS': 998.0,
    'TD': 1008.0,
    'TS': 1000.0,
    'WV': 1009.0
}

# Imputar directamente en la variable MinPress
df_EDA['MinPress'] = df_EDA['MinPress'].fillna(df_EDA['type_storm'].map(medianas_por_tipo))

# Verificar valores faltantes después de imputar
print(f"\nValores faltantes después de imputar: {df_EDA['MinPress'].isnull().sum()}")

# Mostrar filas que aún tienen valores faltantes
filas_con_nan = df_EDA[df_EDA['MinPress'].isnull()]
print(f"\nFilas que quedaron con valores faltantes después de imputar:")
print(f"Total de filas con NaN: {len(filas_con_nan)}")

if len(filas_con_nan) > 0:
    print("\nDetalle de las filas con valores faltantes:")
    print(filas_con_nan[['type_storm', 'MinPress']].head(20))
else:
    print("No quedaron valores faltantes después de la imputación.")


# In[390]:


df_EDA.head(3)


# In[391]:


msno.matrix(df_EDA)
plt.show()


# # 4. IMPUTACIONES

# ## 4.0 Longitud negativa

# In[392]:


df_EDA


# In[393]:


df_EDA["Lon_W"] = -df_EDA["Lon_W"]


# ## 4.1 Cálculo de vector de longitud

# In[394]:


# Crear columnas para vectores
df_EDA["vec_x"] = np.nan
df_EDA["vec_y"] = np.nan

# Diccionario para almacenar vectores
vecs = {"vec_x": [], "vec_y": [], "id": []}

# Procesar cada tormenta por separado
for storm_id in df_EDA["Code_storm"].unique():
    # Filtrar datos de la tormenta y ordenar por fecha
    storm_data = df_EDA[df_EDA["Code_storm"] == storm_id].copy()
    storm_data = storm_data.sort_values(['year', 'month', 'day', 'hour']).reset_index()
    
    # Inicializar variables
    last_x = None
    last_y = None
    
    # Calcular vectores para cada punto de la tormenta
    for i, row in storm_data.iterrows():
        current_x = row["Lon_W"]
        current_y = row["Lat_N"]
        
        # Si no es el primer punto, calcular vector
        if last_x is not None and last_y is not None:
            vec_x = current_x - last_x
            vec_y = current_y - last_y
            
            # Guardar vector y índice original
            vecs["vec_x"].append(vec_x)
            vecs["vec_y"].append(vec_y)
            vecs["id"].append(row["index"])  # Índice original en df
        
        # Actualizar coordenadas para siguiente iteración
        last_x = current_x
        last_y = current_y

# Asignar vectores al DataFrame original
df_EDA.loc[vecs["id"], "vec_x"] = vecs["vec_x"]
df_EDA.loc[vecs["id"], "vec_y"] = vecs["vec_y"]


# Manejado la ecuación de calculo de un vector a partir de grados decimales. 
# 
# * Vector_Longitud = ((vec_x)"2 + (vec_y)"2)"-2

# In[395]:


df_EDA["vec_len"] = np.sqrt((df_EDA["vec_x"]**2)+(df_EDA["vec_y"]**2))


# Hubo valores de 354 a 356 de vec_len que no permitía graficar un buen histograma, así que eliminamos la ficha que corresponde a ese valor de 356. Sucedió porque se registró una tormenta en dos punto muy distantes y en las etapas finales de la tormenta cuando sale de la zona de los trópicos y gana mas velocidad

# In[396]:


# Ver cuántos se eliminarán
print(f"Filas a eliminar: {(df_EDA['vec_len'] >= 354).sum()}")

# Eliminar
df_EDA = df_EDA[df_EDA["vec_len"] < 354]


# In[397]:


# Histograma de valores mayores a 5.5
plt.figure(figsize=(10, 6))
plt.hist(df_EDA['vec_len'], bins=20, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribución de Valores en vec_len')
plt.xlabel('Valores de vec_len')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)
plt.show()


# ## 4.2 Calculo de la dirección de la tormenta

# La ecuación usada para el cálculo de la dirección es:
# 
# * Dirección = Arctg( vector_x / vector_y)

# In[398]:


def calculate_direction(vec_x, vec_y):
    # atan2 devuelve el ángulo en radianes en el rango [-π, π]
    # (x, y) para que 0 rad apunte al norte
    #return np.arctan2(vec_y, -vec_x)
    
    # (x, y) para que 0 rad apunte al este
    return np.arctan2(vec_x, vec_y)


# Aplicar al DataFrame
df_EDA["vec_direction"] = df_EDA.apply(lambda x: calculate_direction(x.vec_x, x.vec_y), axis=1)


# In[399]:


# Histograma de vec_direction
plt.figure(figsize=(10, 6))
plt.hist(df_EDA["vec_direction"], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
plt.title("Distribución de vec_direction")
plt.xlabel("vec_direction (radianes)")
plt.ylabel("Frecuencia")
plt.grid(True, alpha=0.3)
plt.show()


# 
# Para cada tormenta (Code_storm) vamos a crear 3 lags, quiere decir, los 3 pasos anteriores

# In[400]:


# Para cada tormenta (Code_storm) vamos a crear 3 columnas con desplazamiento
for k in range(1, 4):
    df_EDA[f"prev_len_{k}"] = df_EDA.groupby("Code_storm")["vec_len"].shift(k)
    df_EDA[f"prev_direction_{k}"] = df_EDA.groupby("Code_storm")["vec_direction"].shift(k)


# Se crea una columna que almacenará un paso adelante del vector longitud y vector dirección. Llevará el nombre next_len y next_dir y serán los targets en cada modelo.

# In[401]:


df_EDA["next_len"] = df_EDA.groupby("Code_storm")["vec_len"].shift(-1)
df_EDA["next_direction"] = df_EDA.groupby("Code_storm")["vec_direction"].shift(-1)


# In[402]:


# Finalmente Se eliminan todas las filas que tengan valores nulos, ya que no se puede usar para entrenar los modelos
df_EDA = df_EDA.dropna().reset_index(drop=True)


# In[403]:


df_EDA


# ## 4.3 UNIR DATOS SHIPS (CONDICIONES FÍSICAS DEL AMBIENTE) A LOS DATOS HURDAT (TRACKS DE HURACANES)

# In[404]:


#MIN_TRACKS = 10

#conteos = df_EDA.groupby('Code_storm').transform('size')
#df_EDA = df_EDA[conteos > MIN_TRACKS].copy()

#print(f"Original: {len(conteos)} filas en df_EDA")
#print(f"Filtrados: {len(df_EDA)} filas (Code_storm con más de {MIN_TRACKS} registros)")


# In[405]:


#df_ships = pd.read_csv("/Users/mauriciohurtado/modelo_clasificacion/datos_ships/datos_ships.csv")

df_ships = pd.read_csv("/Users/mauriciohurtado/modelsTesis/real_tesis/datos/ships_filter.csv", sep= ';')


# In[406]:


print("Columnas numéricas:")
print(df_ships.select_dtypes(include=['float64', 'int64']).columns.tolist())
print("\nColumnas de texto:")
print(df_ships.select_dtypes(include=['object']).columns.tolist())


# In[407]:


print("Columnas numéricas:")
print(df_EDA.select_dtypes(include=['float64', 'int64']).columns.tolist())
print("\nColumnas de texto:")
print(df_EDA.select_dtypes(include=['object']).columns.tolist())


# In[408]:


# filtrar el df_ships con las columnas a usar
ships_filter = df_ships[['storm_code', 'shear_kt', 'sst_c', 'rh_pct', 'heat_content', 'storm_bearing_deg', 'storm_motion_kt', 'lat_hurdat2', 'lon_hurdat2', 'lat_s', 'lon_s']]


# In[409]:


print("Columnas numéricas:")
print(ships_filter.select_dtypes(include=['float64', 'int64']).columns.tolist())
print("\nColumnas de texto:")
print(ships_filter.select_dtypes(include=['object']).columns.tolist())


# In[410]:


# ===============================
# 1. Copias de seguridad
# ===============================
df1 = df_EDA.copy()         # df principal
df2 = ships_filter.copy()   # df con variables adicionales


# ===============================
# 2. Renombrar columnas equivalentes
# ===============================
df2 = df2.rename(columns={
    'storm_code': 'Code_storm',
    'lat_hurdat2': 'Lat_N',
    'lon_hurdat2': 'Lon_W'
})


# ===============================
# 3. Columnas que sirven como llave
# ===============================
columnas_llave = ['Code_storm', 'Lat_N', 'Lon_W']


# ===============================
# 4. Columnas numéricas y de texto
#    que quieres traer desde ships_filter
# ===============================

columnas_ships_numericas = [
    'shear_kt', 
    'sst_c', 
    'rh_pct', 
    'heat_content', 
    'storm_bearing_deg',
    'storm_motion_kt',
    'lat_s',
    'lon_s'
]

columnas_ships_texto = [
    # storm_code ya se renombró → Code_storm
    # agrega otras columnas de texto si existen
]

columnas_ships_agregar = columnas_ships_numericas + columnas_ships_texto


# ===============================
# 5. Unir los dataframes
# ===============================
df_combinado = df1.merge(
    df2[columnas_llave + columnas_ships_agregar],
    on=columnas_llave,
    how='left'
)


# ===============================
# 6. Mostrar información final
# ===============================
print("\nColumnas nuevas incorporadas:")
for col in columnas_ships_agregar:
    print(" -", col)

print(f"\nFilas de df_EDA: {len(df_EDA)}")
print(f"Filas de resultado: {len(df_combinado)}")

# Validar coincidencias
coincidencias = df_combinado['shear_kt'].notnull().sum()
print(f"\nCoincidencias encontradas: {coincidencias}")


# ===============================
# 7. df final listo
# ===============================
df_combinado


# In[411]:


#Eliminar las filas que tengan valores nulos en sst_c
df_combinado = df_combinado[df_combinado['sst_c'].notna()]



# In[412]:


df_EDA = df_combinado.copy()


# In[413]:


df_EDA.head(3)


# ## Unir valores de NAO

# In[414]:


import pandas as pd

# Saltar título (línea 1) y fila "Jan Feb ... Dec" (línea 2); leer solo números
df_nao = pd.read_csv(
    "norm_nao_monthly_b5001_current_ascii.txt",
    sep=r"\s+",
    skiprows=2,
    header=None,
    names=[
        "year", "m1", "m2", "m3", "m4", "m5", "m6",
        "m7", "m8", "m9", "m10", "m11", "m12",
    ],
)

df_nao_long = df_nao.melt(
    id_vars=["year"],
    value_vars=[f"m{k}" for k in range(1, 13)],
    var_name="_mk",
    value_name="nao",
)
df_nao_long["month"] = df_nao_long["_mk"].str.replace("m", "", regex=False).astype(int)
df_nao_long = df_nao_long.drop(columns=["_mk"])

df_nao_long["nao"] = pd.to_numeric(df_nao_long["nao"], errors="coerce")
df_nao_long.loc[df_nao_long["nao"] <= -900, "nao"] = pd.NA

df_EDA["year"] = df_EDA["year"].astype(int)
df_EDA["month"] = df_EDA["month"].astype(int)

df_EDA = df_EDA.merge(
    df_nao_long[["year", "month", "nao"]],
    on=["year", "month"],
    how="left",
)


# In[415]:


df_EDA


# In[416]:


df_EDA_fil = df_EDA[['Code_storm', 'Lon_W', 'Lat_N', 'Max_wind', 'MinPress', 'type_storm_n', 'sst_c', 'rh_pct', 'shear_kt', 'storm_bearing_deg', 'storm_motion_kt', 'heat_content', 'nao', 'vec_direction', 'vec_len']].copy()


# In[417]:


#df_EDA_fil.to_csv("df_EDA_fisicos.txt", index=False)


# In[418]:


df_EDA_fil.head(3)


# # ## funciones generales para los modelos, métricas y gráficas

# In[419]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def metricas_regresion_robustas(y_true, y_pred, eps=1e-6, umbral_mape=1.0):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE filtrado (evita explosión por valores cercanos a 0)
    mask = np.abs(y_true) >= umbral_mape
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100 if mask.any() else np.nan
    # sMAPE y WAPE (más estables)
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps) * 100
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_%_filtrado": mape,
        "sMAPE_%": smape,
        "WAPE_%": wape,
    }


# In[420]:


import matplotlib.pyplot as plt
import numpy as np


def plot_one_sequence(y_true, y_pred, idx=0, titulo="Target"):
    # y_true, y_pred: shape (n_samples, horizon)
    h = y_true.shape[1]
    x = np.arange(1, h + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y_true[idx], marker="o", label="Real")
    plt.plot(x, y_pred[idx], marker="o", linestyle="--", label="Pred")
    plt.title(f"{titulo} - muestra {idx}")
    plt.xlabel("Horizonte (t+1 ... t+h)")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[421]:


def plot_residuals(y_true, y_pred, model_name="Modelo", conjunto="Test", save_path=None):
    """Crea gráficos de análisis de residuos."""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Análisis de Residuos - {model_name} ({conjunto})",
        fontsize=16, fontweight="bold", y=0.995
    )

    # 1) Residuos vs predichos
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20, color="#2E86AB")
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    z = np.polyfit(y_pred, residuals, 1)
    axes[0, 0].plot(y_pred, np.poly1d(z)(y_pred), "r--", alpha=0.8, linewidth=1.5)
    axes[0, 0].set_xlabel("Valores Predichos")
    axes[0, 0].set_ylabel("Residuos")
    axes[0, 0].set_title("Residuos vs Valores Predichos")
    axes[0, 0].grid(True, alpha=0.3, linestyle="--")

    # 2) Histograma
    n, bins, _ = axes[0, 1].hist(residuals, bins=30, color="#A23B72", alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = len(residuals) * (bins[1] - bins[0]) * stats.norm.pdf(x_norm, mu, sigma)
    axes[0, 1].plot(x_norm, y_norm, "r-", linewidth=2, label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})")
    axes[0, 1].set_xlabel("Residuos")
    axes[0, 1].set_ylabel("Frecuencia")
    axes[0, 1].set_title("Distribución de Residuos")
    axes[0, 1].grid(True, alpha=0.3, linestyle="--", axis="y")
    axes[0, 1].legend()

    # 3) Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normalidad de Residuos)")
    axes[1, 0].grid(True, alpha=0.3, linestyle="--")

    # 4) Residuos vs índice
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5, s=20, color="#F18F01")
    axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Índice de Observación")
    axes[1, 1].set_ylabel("Residuos")
    axes[1, 1].set_title("Residuos vs Índice")
    axes[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", format="jpeg")
        plt.close()
    else:
        plt.show()





def analizar_residuos_test_tres_targets(
    nombre_modelo,
    model_dir,
    model_len,
    model_max_wind,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_max_wind_flat,
    y_test_dir,
    y_test_len,
    y_test_max_wind,
    output_dir=None,  # ej: "plots_modelos/rfr"
):
    """
    Calcula predicciones en test y genera análisis de residuos para
    Dirección, Length y Max_wind usando la función plot_residuals existente.
    """
    y_pred_dir = model_dir.predict(X_test_dir_flat)
    y_pred_len = model_len.predict(X_test_len_flat)
    y_pred_mw = model_max_wind.predict(X_test_max_wind_flat)

    # Dirección
    save_dir = None if output_dir is None else f"{output_dir}/residuals_direction.jpg"
    plot_residuals(
        y_test_dir.ravel(),
        y_pred_dir.ravel(),
        model_name=f"{nombre_modelo} - Direccion",
        conjunto="Test",
        save_path=save_dir,
    )

    # Length
    save_len = None if output_dir is None else f"{output_dir}/residuals_length.jpg"
    plot_residuals(
        y_test_len.ravel(),
        y_pred_len.ravel(),
        model_name=f"{nombre_modelo} - Length",
        conjunto="Test",
        save_path=save_len,
    )

    # Max_wind
    save_mw = None if output_dir is None else f"{output_dir}/residuals_max_wind.jpg"
    plot_residuals(
        y_test_max_wind.ravel(),
        y_pred_mw.ravel(),
        model_name=f"{nombre_modelo} - Max_wind",
        conjunto="Test",
        save_path=save_mw,
    )

    return y_pred_dir, y_pred_len, y_pred_mw


# In[422]:


import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def resumir_fi_flat_a_no_flat(df_fi, col_importance="importance"):
    """
    Convierte FI flat (feature_t-k) a vistas no-flat:
    1) por variable
    2) por lag (t-k)
    """
    df = df_fi.copy()

    # Detectar columna de importancia
    if col_importance not in df.columns:
        # fallback automático
        posibles = [c for c in ["importance", "importance_promedio_seq", "importance_local"] if c in df.columns]
        if not posibles:
            raise ValueError("No encuentro columna de importancia en df_fi")
        col_importance = posibles[0]

    if "feature" not in df.columns:
        raise ValueError("df_fi debe tener columna 'feature' con formato tipo 'MinPress_t-0'")

    # Parsear nombre y lag
    # Soporta variables con underscore: ej. max_wind_t-3
    parsed = df["feature"].str.extract(r"^(?P<var>.+)_t-(?P<lag>\d+)$")
    if parsed["var"].isna().any():
        raise ValueError("Algunas features no tienen formato '<variable>_t-<lag>'")

    df["variable"] = parsed["var"]
    df["lag"] = parsed["lag"].astype(int)

    # Resumen no-flat por variable (global)
    fi_por_variable = (
        df.groupby("variable", as_index=False)[col_importance]
        .sum()
        .sort_values(col_importance, ascending=False)
        .reset_index(drop=True)
    )

    # Resumen no-flat por lag temporal (global)
    fi_por_lag = (
        df.groupby("lag", as_index=False)[col_importance]
        .sum()
        .sort_values("lag", ascending=False)   # t-11 ... t-0
        .reset_index(drop=True)
    )
    fi_por_lag["etiqueta_lag"] = fi_por_lag["lag"].apply(lambda x: f"t-{x}")

    # Matriz variable x lag (heatmap o tabla)
    fi_matriz_var_lag = (
        df.pivot_table(
            index="variable",
            columns="lag",
            values=col_importance,
            aggfunc="sum",
            fill_value=0.0
        )
        .sort_index(axis=1, ascending=False)  # columnas: t-11 ... t-0
    )

    return fi_por_variable, fi_por_lag, fi_matriz_var_lag


def plot_fi_no_flat(fi_por_variable, fi_por_lag, top_vars=15, titulo="FI no-flat"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Por variable ---
    dfv = fi_por_variable.head(top_vars).copy()
    axes[0].barh(range(len(dfv)), dfv.iloc[:, 1], color=plt.cm.viridis(np.linspace(0, 1, len(dfv))))
    axes[0].set_yticks(range(len(dfv)))
    axes[0].set_yticklabels(dfv["variable"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Importancia agregada")
    axes[0].set_title(f"{titulo} - Por variable")
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")

    # --- Por lag ---
    axes[1].bar(fi_por_lag["etiqueta_lag"], fi_por_lag.iloc[:, 1], color="#2E86AB")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Importancia agregada")
    axes[1].set_title(f"{titulo} - Por lag temporal")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# In[423]:


def _final_step_pipeline(estimator):
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps[list(estimator.named_steps.keys())[-1]]
    return estimator


# In[424]:


from typing import Any, Collection, List, Optional, Union


# In[425]:


def fi_promedio_todas_las_secuencias(
    model,
    X_train_flat,
    timesteps: int,
    base_features,
    agg_horizontes: str = "mean_abs",
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Promedia importancias de características entre salidas del MultiOutputRegressor
    (un árbol/XGB/RF por horizonte). Requiere `feature_importances_` en el último paso del pipeline.
    """
    if not hasattr(model, "estimators_"):
        raise ValueError(
            "Se espera un MultiOutputRegressor ajustado (atributo estimators_)."
        )

    X = np.asarray(X_train_flat)
    feature_names = [
        f"{feat}_t-{t}"
        for t in range(timesteps - 1, -1, -1)
        for feat in base_features
    ]
    n_exp = timesteps * len(base_features)
    if X.shape[1] != n_exp:
        raise ValueError(
            f"X_train_flat tiene {X.shape[1]} columnas; se esperaban {n_exp} "
            f"(timesteps={timesteps} × len(base_features)={len(base_features)})."
        )

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    rows = []
    for est in model.estimators_:
        final = _final_step_pipeline(est)
        if not hasattr(final, "feature_importances_"):
            raise AttributeError(
                f"El estimador {type(final).__name__} no expone feature_importances_; "
                "esta función aplica a RandomForest, XGBoost, árbol, etc."
            )
        rows.append(final.feature_importances_)
    M = np.vstack(rows)

    if agg_horizontes == "mean_abs":
        agg = np.mean(np.abs(M), axis=0)
    elif agg_horizontes == "mean":
        agg = np.mean(M, axis=0)
    else:
        raise ValueError(
            f"agg_horizontes debe ser 'mean_abs' o 'mean', recibido: {agg_horizontes!r}"
        )

    out = pd.DataFrame({"feature": feature_names, "importance_promedio_seq": agg})
    for h in range(M.shape[0]):
        out[f"importance_h{h}"] = M[h]
    return out


# # 5. Split Train_val_test
# 
# Para el entrenamiento de los modelos K-Nearest Neighbors (KNN), Decision Tree, Random Forest, XGBoost y Support Vector Regression (SVR), se realiza un split de los datos en conjuntos de entrenamiento, validación y prueba.
# 
# Este proceso utiliza como variables de entrada (X features) la presión mínima (minpress), los vientos máximos sostenidos (max_wind), el vector de dirección (vec_direction), el vector de longitud (vec_len) y el mes (month).
# 
# Las variables objetivo (targets) corresponden al vector de dirección en el instante futuro (t+1), denominado next_direction, y al vector de longitud en t+1, denominado next_len.

# In[426]:


# export df_EDA

#df_EDA.to_csv("df_EDA_30_03_2026.txt", sep="\t", index=False)


# In[427]:


# SE REALIZA UN SPLIT EN FUNCION DE LOS CODIGOS DE TORMENTA.
# SE CREA UNA VARIABLE QUE CONTIENE LOS CODIGOS DE TORMENTA PARA CADA TORMENTA. PARA SER USADO EN EL SPLIT DE GRIDSEARCH


cantidad_unicos = df_EDA["Code_storm"].nunique()
print("Cantidad de códigos únicos:", cantidad_unicos)

n_test = max(1, int(round(cantidad_unicos * 0.10)))
n_test = min(n_test, cantidad_unicos)  # seguridad

ids_ordenados = df_EDA["Code_storm"].dropna().unique()

# Test: últimos IDs
test_ids = ids_ordenados[-n_test:]
resto_ids = ids_ordenados[:-n_test]
train_ids = resto_ids


train = df_EDA[df_EDA["Code_storm"].isin(train_ids)]
test = df_EDA[df_EDA["Code_storm"].isin(test_ids)]

print("Train:", train["Code_storm"].nunique(), "Test:", test["Code_storm"].nunique())


# In[428]:


# ========================================
# FUNCIÓN PARA CREAR SECUENCIAS (DESPUÉS DE SPLIT)
# ========================================
def create_sequences_separated(df, timesteps=12, horizon=12, features=["MinPress","Max_wind",'vec_direction',"vec_len", "Lat_N", "Lon_W",'nao','shear_kt', 'sst_c', 'rh_pct', 'heat_content','storm_bearing_deg','storm_motion_kt']):
    """
    Crea secuencias LSTM separadas para dirección, longitud y max_wind.
    Cada secuencia X excluye su variable objetivo correspondiente.
    Retorna: X_dir, y_dir, X_len, y_len, X_minP, y_minP, grupos
    """
    X_sequences_dir = []
    y_sequences_dir = []
    X_sequences_len = []
    y_sequences_len = []
    X_sequences_minP = []
    y_sequences_minP = []
    grupos_sequences = []
    
    # Definir features para cada modelo (excluyendo la variable objetivo)
    features_dir = [f for f in features if f != 'vec_direction' and f != 'Max_wind']
    features_len = [f for f in features if f != 'vec_len' and f != 'Max_wind']
    features_minP = [f for f in features if f != 'MinPress']
    
    # Ordenar solo por Code_storm
    df_sorted = df.sort_values(['Code_storm']).reset_index(drop=True)
    
    # Por cada huracán (Code_storm) en este split
    for storm_id in df_sorted['Code_storm'].unique():
        storm_data = df_sorted[df_sorted['Code_storm'] == storm_id].copy().reset_index(drop=True)
        
        if len(storm_data) < timesteps + horizon:
            continue
        
        for i in range(len(storm_data) - timesteps - horizon + 1):
            X_seq_dir = storm_data[features_dir].iloc[i:i+timesteps].values
            X_seq_len = storm_data[features_len].iloc[i:i+timesteps].values
            X_seq_minP = storm_data[features_minP].iloc[i:i+timesteps].values
            
            y_dir = storm_data['vec_direction'].iloc[i+timesteps:i+timesteps+horizon].values
            y_len = storm_data['vec_len'].iloc[i+timesteps:i+timesteps+horizon].values
            y_minP = storm_data['MinPress'].iloc[i+timesteps:i+timesteps+horizon].values

            X_sequences_dir.append(X_seq_dir)
            y_sequences_dir.append(y_dir)
            X_sequences_len.append(X_seq_len)
            y_sequences_len.append(y_len)
            X_sequences_minP.append(X_seq_minP)
            y_sequences_minP.append(y_minP)
            grupos_sequences.append(storm_id)
    
    X_dir = np.array(X_sequences_dir) if X_sequences_dir else np.array([]).reshape(0, timesteps, len(features_dir))
    y_dir = np.array(y_sequences_dir) if y_sequences_dir else np.array([]).reshape(0, horizon)
    X_len = np.array(X_sequences_len) if X_sequences_len else np.array([]).reshape(0, timesteps, len(features_len))
    y_len = np.array(y_sequences_len) if y_sequences_len else np.array([]).reshape(0, horizon)
    X_minP = np.array(X_sequences_minP) if X_sequences_minP else np.array([]).reshape(0, timesteps, len(features_minP))
    y_minP = np.array(y_sequences_minP) if y_sequences_minP else np.array([]).reshape(0, horizon)
    grupos = np.array(grupos_sequences)
    
    return X_dir, y_dir, X_len, y_len, X_minP, y_minP, grupos


# ========================================
# CREAR SECUENCIAS SOBRE TUS CONJUNTOS YA DIVIDIDOS
# ========================================
timesteps = 12
horizon = 12
features = ["MinPress","Max_wind","vec_len", 'vec_direction', "Lat_N", "Lon_W",'shear_kt', 
            'sst_c', 'rh_pct', 'heat_content','storm_bearing_deg','storm_motion_kt', 'nao']


# TRAIN
print("Creando secuencias para TRAIN...")
X_train_dir, y_train_dir, X_train_len, y_train_len, X_train_minP, y_train_minP, grupos_train = create_sequences_separated(
    train, timesteps=timesteps, horizon=horizon, features=features
)
print(f"Train DIR: X shape = {X_train_dir.shape}, y shape = {y_train_dir.shape}, grupos: {len(grupos_train)}")
print(f"Train LEN: X shape = {X_train_len.shape}, y shape = {y_train_len.shape}")
print(f"Train MAX_WIND: X shape = {X_train_minP.shape}, y shape = {y_train_minP.shape}")


# TEST
print("\nCreando secuencias para TEST...")
X_test_dir, y_test_dir, X_test_len, y_test_len, X_test_minP, y_test_minP, grupos_test = create_sequences_separated(
    test, timesteps=timesteps, horizon=horizon, features=features
)


# In[429]:


# APLANAR LAS SECUENCIAS PARA KNN (y otros modelos tradicionales)
# Convertir (n_samples, timesteps=3, n_features=4) → (n_samples, 12)

def flatten_sequences(X):
    """
    Aplana secuencias 3D a 2D para modelos tradicionales.
    (n_samples, timesteps, n_features) → (n_samples, timesteps * n_features)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)

# Aplanar secuencias de train
X_train_dir_flat = flatten_sequences(X_train_dir)
X_train_len_flat = flatten_sequences(X_train_len)
X_train_minP_flat = flatten_sequences(X_train_minP)


X_test_dir_flat = flatten_sequences(X_test_dir)
X_test_len_flat = flatten_sequences(X_test_len)
X_test_minP_flat = flatten_sequences(X_test_minP)
print(f"Train DIR - Original: {X_train_dir.shape}, Aplanado: {X_train_dir_flat.shape}")
print(f"Train LEN - Original: {X_train_len.shape}, Aplanado: {X_train_len_flat.shape}")
print(f"Train MAX_WIND - Original: {X_train_minP.shape}, Aplanado: {X_train_minP_flat.shape}")


# In[430]:


from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import  GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler


# # 5.1 Modelo KNN

# In[431]:


#from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

# Pipeline base
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])

# Envolver con MultiOutputRegressor
multi_pipe = MultiOutputRegressor(pipe)


param_grid = {
    "estimator__knn__n_neighbors": [10, 15, 20],
    "estimator__knn__weights": ["uniform", "distance"],
    "estimator__knn__metric": ["euclidean", "manhattan", "minkowski"],
    "estimator__knn__p": [1, 2]
}


cvdir, cvlen, cvminp = GroupKFold(n_splits=7), GroupKFold(n_splits=7), GroupKFold(n_splits=7)

grid_dir = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvdir,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_len = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvlen,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_minP = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvminp,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_dir.fit(X_train_dir_flat, y_train_dir,  groups=grupos_train)

grid_len.fit(X_train_len_flat, y_train_len,  groups=grupos_train)

grid_minP.fit(X_train_minP_flat, y_train_minP,  groups=grupos_train)

print("Mejores parámetros:", grid_dir.best_params_)
print("Mejor score:", grid_dir.best_score_)

print("Mejores parámetros:", grid_len.best_params_)
print("Mejor score:", grid_len.best_score_)

print("Mejores parámetros:", grid_minP.best_params_)
print("Mejor score:", grid_minP.best_score_)

# Modelo final
model_d_knn = grid_dir.best_estimator_
model_l_knn = grid_len.best_estimator_
model_minP_knn = grid_minP.best_estimator_


# In[432]:


#guardar el modelo
import joblib

joblib.dump(model_d_knn, 'knnr_physicfeatures_secuence_dir.pkl')
joblib.dump(model_l_knn, 'knnr_physicfeatures_secuence_len.pkl')
joblib.dump(model_minP_knn, 'knnr_physicfeatures_secuence_minp.pkl')


# ### plots

# In[433]:


# Predicciones (omite este bloque si ya tienes y_pred_*)
y_pred_test_dir = model_d_knn.predict(X_test_dir_flat)
y_pred_test_len = model_l_knn.predict(X_test_len_flat)
y_pred_test_minP = model_minP_knn.predict(X_test_minP_flat)

idx = 15

plot_one_sequence(y_test_minP, y_pred_test_minP, idx=idx, titulo="MinPress")
plot_one_sequence(y_test_len, y_pred_test_len, idx=idx, titulo="Length")
plot_one_sequence(y_test_dir, y_pred_test_dir, idx=idx, titulo="Direction")


# In[434]:


_ = analizar_residuos_test_tres_targets(
    "Knn",
    model_d_knn,
    model_l_knn,
    model_minP_knn,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_minP_flat,
    y_test_dir,
    y_test_len,
    y_test_minP,
    output_dir=None,
)


# In[435]:


# 1) Predicciones
y_pred_dir = model_d_knn.predict(X_test_dir_flat)
y_pred_len = model_l_knn.predict(X_test_len_flat)
y_pred_mw  = model_minP_knn.predict(X_test_minP_flat)
# 2) Métricas
m_dir = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=1.0)
m_len = metricas_regresion_robustas(y_test_len, y_pred_len, umbral_mape=1.0)
m_mw  = metricas_regresion_robustas(y_test_minP, y_pred_mw, umbral_mape=1.0)
# 3) Tabla final
df_metricas = pd.DataFrame([
  {"Target": "Dirección", **m_dir},
  {"Target": "Length", **m_len},
  {"Target": "MinPress", **m_mw},
])
display(df_metricas)
print(df_metricas.to_string(index=False))


# # 5.2 Modelo Decission tree

# In[436]:


from sklearn.tree import DecisionTreeRegressor
#from pyspark.ml.regression import DecisionTreeRegressor

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("dtr", DecisionTreeRegressor(random_state=42))
])

multi_pipe = MultiOutputRegressor(pipe)


param_grid = {
    "estimator__dtr__max_depth": [None, 5, 10, 30],
    "estimator__dtr__min_samples_split": [5, 10],
    # "estimator__dtr__min_samples_leaf": [1, 2, 4]
}

cvdir, cvlen, cvminp = GroupKFold(n_splits=7), GroupKFold(n_splits=7), GroupKFold(n_splits=7)

grid_dir = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvdir,
    scoring='r2',#'neg_mean_squared_error',
    n_jobs=2)

grid_len = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvlen,
    scoring='r2',#'neg_mean_squared_error',
    n_jobs=2)

grid_minP = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvminp,
    scoring='r2',#'neg_mean_squared_error',
    n_jobs=2)


grid_dir.fit(X_train_dir_flat, y_train_dir,  groups=grupos_train)

grid_len.fit(X_train_len_flat, y_train_len,  groups=grupos_train)

grid_minP.fit(X_train_minP_flat, y_train_minP,  groups=grupos_train)

print("Mejores parámetros:", grid_dir.best_params_)
print("Mejor score:", grid_dir.best_score_)

print("Mejores parámetros:", grid_len.best_params_)
print("Mejor score:", grid_len.best_score_)

print("Mejores parámetros:", grid_minP.best_params_)
print("Mejor score:", grid_minP.best_score_)

# Modelo final
model_d_dt = grid_dir.best_estimator_
model_l_dt = grid_len.best_estimator_
model_minP_dt = grid_minP.best_estimator_


# In[437]:


# guardamos el modelo
joblib.dump(model_d_dt, 'dtree_physicfeatures_secuence_dir.pkl')
joblib.dump(model_l_dt, 'dtree_physicfeatures_secuence_len.pkl')
joblib.dump(model_minP_dt, 'dtree_physicfeatures_secuence_max_wind.pkl')


# ### plot

# In[438]:


# Predicciones (omite este bloque si ya tienes y_pred_*)
y_pred_test_dir = model_d_dt.predict(X_test_dir_flat)
y_pred_test_len = model_l_dt.predict(X_test_len_flat)
y_pred_test_minP = model_minP_dt.predict(X_test_minP_flat)

idx = 10

plot_one_sequence(y_test_minP, y_pred_test_minP, idx=idx, titulo="Min Press")
plot_one_sequence(y_test_len, y_pred_test_len, idx=idx, titulo="Length")
plot_one_sequence(y_test_dir, y_pred_test_dir, idx=idx, titulo="Direction")


# In[439]:


_ = analizar_residuos_test_tres_targets(
    "Dtree",
    model_d_dt,
    model_l_dt,
    model_minP_dt,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_minP_flat,
    y_test_dir,
    y_test_len,
    y_test_minP,
    output_dir=None,
)


# In[440]:


# Config por target
cfg = {
    "Dirección": {
        "model": model_d_dt,
        "X": X_train_dir_flat,
        "base_features": ["MinPress","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
    "Length": {
        "model": model_l_dt,
        "X": X_train_len_flat,
        "base_features": ["MinPress", "Lat_N", 'vec_direction',
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
    "Minpress": {
        "model": model_minP_dt,
        "X": X_train_minP_flat,
        "base_features": ["Max_wind","vec_len", 'vec_direction', "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
}

resultados = {}
for nombre, d in cfg.items():
    fi_flat = fi_promedio_todas_las_secuencias(
        model=d["model"],
        X_train_flat=d["X"],
        timesteps=12,
        base_features=d["base_features"],
        agg_horizontes="mean_abs",
        max_samples=100
    )
    fi_var, fi_lag, fi_mat = resumir_fi_flat_a_no_flat(
        fi_flat, col_importance="importance_promedio_seq"
    )

    resultados[nombre] = {"flat": fi_flat, "var": fi_var, "lag": fi_lag, "mat": fi_mat}

    print(f"\n=== {nombre} | Top variables ===")
    print(fi_var.head(10).to_string(index=False))
    plot_fi_no_flat(fi_var, fi_lag, top_vars=10, titulo=f"DTREE {nombre}")


# In[441]:


# 1) Predicciones
y_pred_dir = model_d_dt.predict(X_test_dir_flat)
y_pred_len = model_l_dt.predict(X_test_len_flat)
y_pred_mw  = model_minP_dt.predict(X_test_minP_flat)
# 2) Métricas
m_dir = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=1.0)
m_len = metricas_regresion_robustas(y_test_len, y_pred_len, umbral_mape=1.0)
m_mw  = metricas_regresion_robustas(y_test_minP, y_pred_mw, umbral_mape=1.0)
# 3) Tabla final
df_metricas = pd.DataFrame([
  {"Target": "Dirección", **m_dir},
  {"Target": "Length", **m_len},
  {"Target": "MinPress", **m_mw},
])
display(df_metricas)
print(df_metricas.to_string(index=False))


# # 5.3 Random Forest

# In[442]:


#from pyspark.ml.regression import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rfr", RandomForestRegressor(random_state=42))
])

multi_pipe = MultiOutputRegressor(pipe)

param_grid = {
    'estimator__rfr__n_estimators': [50, 100, 200],
    'estimator__rfr__max_depth': [None, 5, 10],
    'estimator__rfr__min_samples_leaf': [1, 2]
}

cvdir, cvlen, cvminp = GroupKFold(n_splits=7), GroupKFold(n_splits=7), GroupKFold(n_splits=7)

grid_dir = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvdir,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_len = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvlen,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_minP = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvminp,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_dir.fit(X_train_dir_flat, y_train_dir,  groups=grupos_train)

grid_len.fit(X_train_len_flat, y_train_len,  groups=grupos_train)

grid_minP.fit(X_train_minP_flat, y_train_minP,  groups=grupos_train)

print("Mejores parámetros:", grid_dir.best_params_)
print("Mejor score:", grid_dir.best_score_)

print("Mejores parámetros:", grid_len.best_params_)
print("Mejor score:", grid_len.best_score_)

print("Mejores parámetros:", grid_minP.best_params_)
print("Mejor score:", grid_minP.best_score_)

# Modelo final
model_d_rf = grid_dir.best_estimator_
model_l_rf = grid_len.best_estimator_
model_minP_rf = grid_minP.best_estimator_


# In[443]:


# Guardar los modelos entrenados
joblib.dump(model_d_rf, 'rfr_multioutput_secuential_dir.pkl')
joblib.dump(model_l_rf, 'rfr_multioutput_secuential_len.pkl')
joblib.dump(model_minP_rf, 'rfr_multioutput_secuential_minP.pkl')


# ### plot

# In[444]:


# Predicciones (omite este bloque si ya tienes y_pred_*)
y_pred_test_dir = model_d_rf.predict(X_test_dir_flat)
y_pred_test_len = model_l_rf.predict(X_test_len_flat)
y_pred_test_minP = model_minP_rf.predict(X_test_minP_flat)

idx = 15

plot_one_sequence(y_test_minP, y_pred_test_minP, idx=idx, titulo="MinPress")
plot_one_sequence(y_test_len, y_pred_test_len, idx=idx, titulo="Length")
plot_one_sequence(y_test_dir, y_pred_test_dir, idx=idx, titulo="Direction")


# In[445]:


_ = analizar_residuos_test_tres_targets(
    "Rf",
    model_d_rf,
    model_l_rf,
    model_minP_rf,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_minP_flat,
    y_test_dir,
    y_test_len,
    y_test_minP,
    output_dir=None,
)


# In[447]:


# Config por target
cfg = {
    "Dirección": {
        "model": model_d_rf,
        "X": X_train_dir_flat,
        "base_features": ["MinPress","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
    "Length": {
        "model": model_l_rf,
        "X": X_train_len_flat,
        "base_features": ["MinPress","vec_direction", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
    "Minpress": {
        "model": model_minP_rf,
        "X": X_train_minP_flat,
        "base_features": ["Max_wind","vec_len", 'vec_direction', "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
}

resultados = {}
for nombre, d in cfg.items():
    fi_flat = fi_promedio_todas_las_secuencias(
        model=d["model"],
        X_train_flat=d["X"],
        timesteps=12,              # ajusta si aplica
        base_features=d["base_features"],
        agg_horizontes="mean_abs",
        max_samples=100
    )
    fi_var, fi_lag, fi_mat = resumir_fi_flat_a_no_flat(
        fi_flat, col_importance="importance_promedio_seq"
    )

    resultados[nombre] = {"flat": fi_flat, "var": fi_var, "lag": fi_lag, "mat": fi_mat}

    print(f"\n=== {nombre} | Top variables ===")
    print(fi_var.head(10).to_string(index=False))
    plot_fi_no_flat(fi_var, fi_lag, top_vars=10, titulo=f"KNN {nombre}")


# In[448]:


# 1) Predicciones
y_pred_dir = model_d_rf.predict(X_test_dir_flat)
y_pred_len = model_l_rf.predict(X_test_len_flat)
y_pred_mw  = model_minP_rf.predict(X_test_minP_flat)
# 2) Métricas
m_dir = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=1.0)
m_len = metricas_regresion_robustas(y_test_len, y_pred_len, umbral_mape=1.0)
m_mw  = metricas_regresion_robustas(y_test_minP, y_pred_mw, umbral_mape=1.0)
# 3) Tabla final
df_metricas = pd.DataFrame([
  {"Target": "Dirección", **m_dir},
  {"Target": "Length", **m_len},
  {"Target": "MinPress", **m_mw},
])
display(df_metricas)
print(df_metricas.to_string(index=False))


# ## 5.3.1 Random Forest — Dirección circular (sin/cos)
#
# Enfoque alternativo para vec_direction: entrenar sin/cos y reconstruir con arctan2.
# Ver circular_direction.py y rf_direction_circular_test.py para la comparación completa.

from circular_direction import (
    direction_to_sincos,
    metricas_circulares,
    sincos_to_direction,
)

# --- ANTES: métricas lineales sobre predicción directa en radianes (ya calculado arriba) ---
m_dir_antes = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=0.5)
m_dir_antes.update(metricas_circulares(y_test_dir, y_pred_dir))

# --- DESPUÉS: entrenar RF sobre sin/cos ---
y_train_dir_sincos = direction_to_sincos(y_train_dir)

pipe_dir_sincos = Pipeline([
    ("scaler", StandardScaler()),
    ("rfr", RandomForestRegressor(random_state=42)),
])
multi_pipe_dir_sincos = MultiOutputRegressor(pipe_dir_sincos)

param_grid_dir_sincos = {
    "estimator__rfr__n_estimators": [50, 100, 200],
    "estimator__rfr__max_depth": [None, 5, 10],
    "estimator__rfr__min_samples_leaf": [1, 2],
}

grid_dir_sincos = GridSearchCV(
    multi_pipe_dir_sincos,
    param_grid_dir_sincos,
    cv=GroupKFold(n_splits=7),
    scoring="neg_mean_squared_error",
    n_jobs=2,
)
grid_dir_sincos.fit(X_train_dir_flat, y_train_dir_sincos, groups=grupos_train)
model_d_rf_sincos = grid_dir_sincos.best_estimator_

y_pred_dir_sincos = model_d_rf_sincos.predict(X_test_dir_flat)
y_pred_dir_circular = sincos_to_direction(y_pred_dir_sincos)

m_dir_despues = metricas_regresion_robustas(y_test_dir, y_pred_dir_circular, umbral_mape=0.5)
m_dir_despues.update(metricas_circulares(y_test_dir, y_pred_dir_circular))

df_comp_dir_rf = pd.DataFrame([
    {"Enfoque": "Antes (rad directo)", **m_dir_antes},
    {"Enfoque": "Después (sin/cos)", **m_dir_despues},
])
print("\n=== RF Dirección: comparación antes vs sin/cos ===")
display(df_comp_dir_rf)
print(df_comp_dir_rf.to_string(index=False))

joblib.dump(model_d_rf_sincos, "rfr_multioutput_secuential_dir_sincos.pkl")


# # 5.4 Modelo XGBoost

# In[ ]:


from xgboost import XGBRegressor
#from xgboost.spark import SparkXGBRegressor


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgbr", XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
        tree_method="hist"
        #early_stopping_rounds=50
    ))
])

multi_pipe = MultiOutputRegressor(pipe)


param_grid = {
    'estimator__xgbr__n_estimators': [100, 200, 300],
    'estimator__xgbr__max_depth': [3, 4, 5, 6, 7],
    'estimator__xgbr__learning_rate': [0.1, 0.2, 0.3],
    #'estimator__xgbr__subsample': [0.8, 0.9, 1.0],
    #'estimator__xgbr__colsample_bytree': [0.8, 0.9, 1.0],
    'estimator__xgbr__reg_lambda': [0.0, 1.0],
    'estimator__xgbr__reg_alpha': [0.0, 1.0]    
}


cvdir, cvlen, cvminp = GroupKFold(n_splits=5), GroupKFold(n_splits=5), GroupKFold(n_splits=5)

grid_dir = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvdir,
    scoring='neg_mean_squared_error',
    n_jobs=1)

grid_len = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvlen,
    scoring='neg_mean_squared_error',
    n_jobs=1)

grid_minP = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvminp,
    scoring='neg_mean_squared_error',
    n_jobs=1)

grid_dir.fit(X_train_dir_flat, y_train_dir,  groups=grupos_train)

grid_len.fit(X_train_len_flat, y_train_len,  groups=grupos_train)

grid_minP.fit(X_train_minP_flat, y_train_minP,  groups=grupos_train)

print("Mejores parámetros:", grid_dir.best_params_)
print("Mejor score:", grid_dir.best_score_)

print("Mejores parámetros:", grid_len.best_params_)
print("Mejor score:", grid_len.best_score_)

print("Mejores parámetros:", grid_minP.best_params_)
print("Mejor score:", grid_minP.best_score_)

# Modelo final
model_d_xgb = grid_dir.best_estimator_
model_l_xgb = grid_len.best_estimator_
model_minP_xgb = grid_minP.best_estimator_


# In[ ]:


#guardamos el modelo
joblib.dump(model_d_xgb, 'xgb_physicfeatures_secuence_dir.pkl')
joblib.dump(model_l_xgb, 'xgb_physicfeatures_secuence_len.pkl')
joblib.dump(model_minP_xgb, 'xgb_physicfeatures_secuence_minP.pkl')
# 5.4 Modelo XGBoost


# ### plots

# In[ ]:


# Predicciones (omite este bloque si ya tienes y_pred_*)
y_pred_test_dir = model_d_xgb.predict(X_test_dir_flat)
y_pred_test_len = model_l_xgb.predict(X_test_len_flat)
y_pred_test_minP = model_minP_xgb.predict(X_test_minP_flat)

idx = 15

plot_one_sequence(y_test_minP, y_pred_test_minP, idx=idx, titulo="MinPress")
plot_one_sequence(y_test_len, y_pred_test_len, idx=idx, titulo="Length")
plot_one_sequence(y_test_dir, y_pred_test_dir, idx=idx, titulo="Direction")


# In[ ]:


_ = analizar_residuos_test_tres_targets(
    "Xgb",
    model_d_xgb,
    model_l_xgb,
    model_minP_xgb,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_minP_flat,
    y_test_dir,
    y_test_len,
    y_test_minP,
    output_dir=None,
)


# In[ ]:


# Config por target
# Config por target
cfg = {
    "Dirección": {
        "model": model_d_xgb,
        "X": X_train_dir_flat,
        "base_features": ["MinPress","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
    "Length": {
        "model": model_l_xgb,
        "X": X_train_len_flat,
        "base_features": ["MinPress","vec_direction", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
    "Minpress": {
        "model": model_minP_xgb,
        "X": X_train_minP_flat,
        "base_features": ["Max_wind","vec_len", 'vec_direction', "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt', 'nao'],
    },
}

resultados = {}
for nombre, d in cfg.items():
    fi_flat = fi_promedio_todas_las_secuencias(
        model=d["model"],
        X_train_flat=d["X"],
        timesteps=12,              # ajusta si aplica
        base_features=d["base_features"],
        agg_horizontes="mean_abs",
        max_samples=100
    )
    fi_var, fi_lag, fi_mat = resumir_fi_flat_a_no_flat(
        fi_flat, col_importance="importance_promedio_seq"
    )

    resultados[nombre] = {"flat": fi_flat, "var": fi_var, "lag": fi_lag, "mat": fi_mat}

    print(f"\n=== {nombre} | Top variables ===")
    print(fi_var.head(10).to_string(index=False))
    plot_fi_no_flat(fi_var, fi_lag, top_vars=10, titulo=f"KNN {nombre}")


# In[ ]:


# 1) Predicciones
y_pred_dir = model_d_xgb.predict(X_test_dir_flat)
y_pred_len = model_l_xgb.predict(X_test_len_flat)
y_pred_mw  = model_minP_xgb.predict(X_test_minP_flat)
# 2) Métricas
m_dir = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=1.0)
m_len = metricas_regresion_robustas(y_test_len, y_pred_len, umbral_mape=1.0)
m_mw  = metricas_regresion_robustas(y_test_minP, y_pred_mw, umbral_mape=1.0)
# 3) Tabla final
df_metricas = pd.DataFrame([
  {"Target": "Dirección", **m_dir},
  {"Target": "Length", **m_len},
  {"Target": "MinPress", **m_mw},
])
display(df_metricas)
print(df_metricas.to_string(index=False))


# # 5.5 Modelo SVR 

# In[ ]:


from sklearn.svm import SVR
#from pyspark.ml.regression import LinearSVR


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel='linear'))
])

multi_pipe = MultiOutputRegressor(pipe)

param_grid = {
    'estimator__svr__C': [0.1, 1, 10],
    'estimator__svr__epsilon': [0.01, 0.1],
    #'estimator__svr__gamma': ['scale']
}


cvdir, cvlen, cvminp = GroupKFold(n_splits=7), GroupKFold(n_splits=7), GroupKFold(n_splits=7)

grid_dir = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvdir,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_len = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvlen,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_minP = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvminp,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_dir.fit(X_train_dir_flat, y_train_dir,  groups=grupos_train)

grid_len.fit(X_train_len_flat, y_train_len,  groups=grupos_train)

grid_minP.fit(X_train_minP_flat, y_train_minP,  groups=grupos_train)

print("Mejores parámetros:", grid_dir.best_params_)
print("Mejor score:", grid_dir.best_score_)

print("Mejores parámetros:", grid_len.best_params_)
print("Mejor score:", grid_len.best_score_)

print("Mejores parámetros:", grid_minP.best_params_)
print("Mejor score:", grid_minP.best_score_)

# Modelo final
model_d_svr = grid_dir.best_estimator_
model_l_svr = grid_len.best_estimator_
model_minP_svr = grid_minP.best_estimator_


# In[ ]:


#guardamos el modelo
joblib.dump(model_d_svr, 'svr_physicfeatures_secuence_dir.pkl')
joblib.dump(model_l_svr, 'svr_physicfeatures_secuence_len.pkl')
joblib.dump(model_minP_svr, 'svr_physicfeatures_secuence_minP.pkl')
# 5.4 Modelo XGBoost


# ### plots

# In[ ]:


# Predicciones (omite este bloque si ya tienes y_pred_*)
y_pred_test_dir = model_d_svr.predict(X_test_dir_flat)
y_pred_test_len = model_l_svr.predict(X_test_len_flat)
y_pred_test_minP = model_minP_svr.predict(X_test_minP_flat)

idx = 15

plot_one_sequence(y_test_minP, y_pred_test_minP, idx=idx, titulo="MinPress")
plot_one_sequence(y_test_len, y_pred_test_len, idx=idx, titulo="Length")
plot_one_sequence(y_test_dir, y_pred_test_dir, idx=idx, titulo="Direction")


# In[ ]:


_ = analizar_residuos_test_tres_targets(
    "Svr",
    model_d_svr,
    model_l_svr,
    model_minP_svr,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_minP_flat,
    y_test_dir,
    y_test_len,
    y_test_minP,
    output_dir=None,
)


# In[ ]:


# Config por target
cfg = {
    "Dirección": {
        "model": model_d_svr,
        "X": X_train_dir_flat,
        "base_features": ["MinPress","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt'],
    },
    "Length": {
        "model": model_l_svr,
        "X": X_train_len_flat,
        "base_features": ["MinPress","vec_direction", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt'],
    },
    "Minpress": {
        "model": model_minP_svr,
        "X": X_train_minP_flat,
        "base_features": ["Max_wind","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt'],
    },
}

resultados = {}
for nombre, d in cfg.items():
    fi_flat = fi_promedio_todas_las_secuencias(
        model=d["model"],
        X_train_flat=d["X"],
        timesteps=12,              # ajusta si aplica
        base_features=d["base_features"],
        agg_horizontes="mean_abs",
        max_samples=100
    )
    fi_var, fi_lag, fi_mat = resumir_fi_flat_a_no_flat(
        fi_flat, col_importance="importance_promedio_seq"
    )

    resultados[nombre] = {"flat": fi_flat, "var": fi_var, "lag": fi_lag, "mat": fi_mat}

    print(f"\n=== {nombre} | Top variables ===")
    print(fi_var.head(10).to_string(index=False))
    plot_fi_no_flat(fi_var, fi_lag, top_vars=10, titulo=f"SVR {nombre}")


# In[ ]:


# 1) Predicciones
y_pred_dir = model_d_svr.predict(X_test_dir_flat)
y_pred_len = model_l_svr.predict(X_test_len_flat)
y_pred_mw  = model_minP_svr.predict(X_test_minP_flat)
# 2) Métricas
m_dir = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=1.0)
m_len = metricas_regresion_robustas(y_test_len, y_pred_len, umbral_mape=1.0)
m_mw  = metricas_regresion_robustas(y_test_minP, y_pred_mw, umbral_mape=1.0)
# 3) Tabla final
df_metricas = pd.DataFrame([
  {"Target": "Dirección", **m_dir},
  {"Target": "Length", **m_len},
  {"Target": "MinPress", **m_mw},
])
display(df_metricas)
print(df_metricas.to_string(index=False))


# # 6.1 Modelo MLP

# In[ ]:


from sklearn.neural_network import MLPRegressor


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(random_state=42))
])

multi_pipe = MultiOutputRegressor(pipe)

param_grid = {
    'estimator__mlp__hidden_layer_sizes': [(10, 10), (10,)],
    'estimator__mlp__activation': ['relu', 'tanh'],
    'estimator__mlp__alpha': [0.0001, 0.001, 0.01],
    'estimator__mlp__max_iter': [400, 600],
    'estimator__mlp__solver': ['adam', 'lbfgs'],
} 

cvdir, cvlen, cvminp = GroupKFold(n_splits=7), GroupKFold(n_splits=7), GroupKFold(n_splits=7)

grid_dir = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvdir,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_len = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvlen,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_minP = GridSearchCV(
    multi_pipe, 
    param_grid,
    cv=cvminp,
    scoring='neg_mean_squared_error',
    n_jobs=2)

grid_dir.fit(X_train_dir_flat, y_train_dir,  groups=grupos_train)

grid_len.fit(X_train_len_flat, y_train_len,  groups=grupos_train)

grid_minP.fit(X_train_minP_flat, y_train_minP,  groups=grupos_train)

print("Mejores parámetros:", grid_dir.best_params_)
print("Mejor score:", grid_dir.best_score_)

print("Mejores parámetros:", grid_len.best_params_)
print("Mejor score:", grid_len.best_score_)

print("Mejores parámetros:", grid_minP.best_params_)
print("Mejor score:", grid_minP.best_score_)

# Modelo final
model_d_mlp = grid_dir.best_estimator_
model_l_mlp = grid_len.best_estimator_
model_minP_mlp = grid_minP.best_estimator_


# In[ ]:


#guardamos el modelo
joblib.dump(model_d_mlp, 'mlp_physicfeatures_secuence_dir.pkl')
joblib.dump(model_l_mlp, 'mlp_physicfeatures_secuence_len.pkl')
joblib.dump(model_minP_mlp, 'mlp_physicfeatures_secuence_minP.pkl')
# 5.4 Modelo XGBoost


# ### plots
# 

# In[ ]:


# Predicciones (omite este bloque si ya tienes y_pred_*)
y_pred_test_dir = model_d_mlp.predict(X_test_dir_flat)
y_pred_test_len = model_l_mlp.predict(X_test_len_flat)
y_pred_test_minP = model_minP_mlp.predict(X_test_minP_flat)

idx = 15

plot_one_sequence(y_test_minP, y_pred_test_minP, idx=idx, titulo="MinPress")
plot_one_sequence(y_test_len, y_pred_test_len, idx=idx, titulo="Length")
plot_one_sequence(y_test_dir, y_pred_test_dir, idx=idx, titulo="Direction")


# In[ ]:


_ = analizar_residuos_test_tres_targets(
    "Mlp",
    model_d_mlp,
    model_l_mlp,
    model_minP_mlp,
    X_test_dir_flat,
    X_test_len_flat,
    X_test_minP_flat,
    y_test_dir,
    y_test_len,
    y_test_minP,
    output_dir=None,
)


# In[ ]:


# Config por target
cfg = {
    "Dirección": {
        "model": model_d_mlp,
        "X": X_train_dir_flat,
        "base_features": ["MinPress","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt'],
    },
    "Length": {
        "model": model_l_mlp,
        "X": X_train_len_flat,
        "base_features": ["MinPress","vec_direction", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt'],
    },
    "Minpress": {
        "model": model_minP_mlp,
        "X": X_train_minP_flat,
        "base_features": ["Max_wind","vec_len", "Lat_N", 
                            "Lon_W",'shear_kt', 'sst_c', 'rh_pct', 
                            'heat_content','storm_bearing_deg','storm_motion_kt'],
    },
}

resultados = {}
for nombre, d in cfg.items():
    fi_flat = fi_promedio_todas_las_secuencias(
        model=d["model"],
        X_train_flat=d["X"],
        timesteps=12,              # ajusta si aplica
        base_features=d["base_features"],
        agg_horizontes="mean_abs",
        max_samples=100
    )
    fi_var, fi_lag, fi_mat = resumir_fi_flat_a_no_flat(
        fi_flat, col_importance="importance_promedio_seq"
    )

    resultados[nombre] = {"flat": fi_flat, "var": fi_var, "lag": fi_lag, "mat": fi_mat}

    print(f"\n=== {nombre} | Top variables ===")
    print(fi_var.head(10).to_string(index=False))
    plot_fi_no_flat(fi_var, fi_lag, top_vars=10, titulo=f"KNN {nombre}")


# In[ ]:


# 1) Predicciones
y_pred_dir = model_d_mlp.predict(X_test_dir_flat)
y_pred_len = model_l_mlp.predict(X_test_len_flat)
y_pred_mw  = model_minP_mlp.predict(X_test_minP_flat)
# 2) Métricas
m_dir = metricas_regresion_robustas(y_test_dir, y_pred_dir, umbral_mape=1.0)
m_len = metricas_regresion_robustas(y_test_len, y_pred_len, umbral_mape=1.0)
m_mw  = metricas_regresion_robustas(y_test_minP, y_pred_mw, umbral_mape=1.0)
# 3) Tabla final
df_metricas = pd.DataFrame([
  {"Target": "Dirección", **m_dir},
  {"Target": "Length", **m_len},
  {"Target": "MinPress", **m_mw},
])
display(df_metricas)
print(df_metricas.to_string(index=False))


# # 6.2 Modelo de RNN
