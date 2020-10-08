#!/usr/bin/env python
# coding: utf-8

# In[1]:
# COMPUTACIÓN BLANDA - Sistemas y Computación

# -----------------------------------------------------------------
# AJUSTES POLINOMIALES
# -----------------------------------------------------------------
# Lección 06 - COVID 19 - PEREIRA
#
#   ** Se importan los archivos de trabajo
#   ** Se crean las variables
#   ** Se generan los modelos
#   ** Se grafican las funciones
#
# -----------------------------------------------------------------

# Se importa la librería del Sistema Operativo
# Igualmente, la librería utils y numpy
# -----------------------------------------------------------------
import os

# Directorios: chart y data en el directorio de trabajo
# DATA_DIR es el directorio de los datos
# CHART_DIR es el directorio de los gráficos generados
# -----------------------------------------------------------------
from utils import DATA_DIR, CHART_DIR
import numpy as np

# Se eliminan las advertencias por el uso de funciones que
# en el futuro cambiarán
# -----------------------------------------------------------------
np.seterr(all='ignore')

# Se importa la librería scipy y matplotlib
# -----------------------------------------------------------------
import scipy as sp
import matplotlib.pyplot as plt

# Datos de trabajo
# -----------------------------------------------------------------
data = np.genfromtxt(os.path.join(DATA_DIR, "tgg.tsv"), 
                     delimiter="\t")

# Se establece el tipo de dato
data = np.array(data, dtype=np.float64)
print(data[:10])
print(data.shape)

# Se definen los colores
# g = green, k = black, b = blue, m = magenta, r = red
# g = verde, k = negro, b = azul, m = magenta, r = rojo
colors = ['g', 'k', 'b', 'm', 'r']

# Se definen los tipos de líneas
# los cuales serán utilizados en las gráficas
linestyles = ['-', '-.', '--', ':', '-']    

# Se crea el vector x, correspondiente a la primera columna de data
# Se crea el vercot y, correspondiente a la segunda columna de data
x = data[:, 0]
y = data[:, 1]

# la función isnan(vector) devuelve un vector en el cual los TRUE
# son valores de tipo nan, y los valores FALSE son valores diferentes
# a nan. Con esta información, este vector permite realizar 
# transformaciones a otros vectores (o al mismo vector), y realizar
# operaciones como sumar el número de posiciones TRUE, con lo
# cual se calcula el total de valores tipo nan
print("Número de entradas incorrectas:", np.sum(np.isnan(y)))

# Se eliminan los datos incorrectos
# -----------------------------------------------------------------

# Los valores nan en el vector y deben eliminarse
# Para ello se crea un vector TRUE y FALSE basado en isnan
# Al negar dichos valores (~), los valores que son FALSE se vuelven
# TRUE, y se corresponden con aquellos valores que NO son nan
# Si el vector x, que contiene los valores en el eje x, se afectan
# a partir de dicho valores lógicos, se genera un nuevo vector en
# el que solos se toman aquellos que son TRUE. Por tanto, se crea
# un nuevo vector x, en el cual han desaparecido los correspondientes
# valores de y que son nan

# Esto mismo se aplica, pero sobre el vector y, lo cual hace que tanto
# x como y queden completamente sincronizados: sin valores nan
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

# CON ESTA FUNCIÓN SE DEFINE UN MODELO,  EL CUAL CONTIENE 
# el comportamiento de un ajuste con base en un grado pol    nomial
# elegido
# -----------------------------------------------------------------
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' dibujar datos de entrada '''

    # Crea una nueva figura, o activa una existente.
    # num = identificador, figsize: anchura, altura
    plt.figure(num=None, figsize=(25, 8))
    
    # Borra el espacio de la figura
    plt.clf()
    
    # Un gráfico de dispersión de y frente a x con diferentes tamaños 
    # y colores de marcador (tamaño = 10)
    plt.scatter(x, y, s=70)
    
    # Títulos de la figura
    # Título superior
    plt.title("Numero de contagios diarios COVID-19 Pereira-Risaralda")
    
    # Título en la base
    plt.xlabel("Tiempo")
    
    # Título lateral
    plt.ylabel("Casos/Dia")
    
    # Obtiene o establece las ubicaciones de las marcas 
    # actuales y las etiquetas del eje x.
    
    # Los primeros corchetes ([]) se refieren a las marcas en x
    # Los siguientes corchetes ([]) se refieren a las etiquetas
    
    # En el primer corchete se tiene: 1*7+ 2*7+ ..., hasta
    # completar el total de puntos en el eje horizontal, según
    # el tamaño del vector x
    
    # Además, se aprovecha para calcular los valores de w, los
    # cuales se agrupan en paquetes de w*7. Esto permite
    # determinar los valores de w desde 1 hasta 5, indicando
    # con ello que se tiene un poco más de 30 semanas
    
    # Estos valores se utilizan en el segundo corchete para
    # escribir las etiquetas basadas en estos valores de w
    
    # Por tanto, se escriben etiquetas para w desde 1 hasta
    # 45, lo cual constituye las semanas analizadas
    plt.xticks(
        [w * 7  for w in range(45)], 
        ['S %i' % w for w in range(45)])

    # Aquí se evalúa el tipo de modelo recibido
    # Si no se envía ninguno, no se dibuja ninguna curva de ajuste
    if models:
        
        # Si no se define ningún valor para mx (revisar el 
        # código más adelante), el valor de mx será
        # calculado con la función linspace

        # NOTA: linspace devuelve números espaciados uniformemente 
        # durante un intervalo especificado. En este caso, sobre
        # el conjunto de valores x establecido
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)
        
        # La función zip () toma elementos iterables 
        # (puede ser cero o más), los agrega en una tupla y los devuelve
        
        # Aquí se realiza un ciclo, la funcion zip nos permite unir diferentes
        # listas en una sola lista asi zip(models, linestyles, color) nos devolveria
        # una solo array con todo el contenido de los argumentos dados, asi podremos con
        # un solo ciclo for recorrer todas las listas necesarios para el graficamiento

        # asi para cada modelo que haya dentro de la lista se graficara segun su color 
        #y estilo con la ayuda de ptl.plot quien realiza dicha graficacion 
        
        for model, style, color in zip(models, linestyles, colors):
            # print "Modelo:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        # crea una leyenda en la parte mas superior izquierda mostrando de forma visual la
        # dimension de cada modelo segun su estilo de linea
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    # Activa o desactiva el ajuste de escala automático
    plt.autoscale(tight=True)
    # se definen los limites que tendran las funciones en la grafica, 
    # esto con el fin de poder extrapolar los resultados o 
    # el comportamiento de las funciones a futuro por defecto 
    # en caso de no ser definidas, se tomara como limites nulos
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    # se guarda la grafica generada segun el nombre de archivo asignado 
    plt.savefig(fname)

# Primera mirada a los datos
# -----------------------------------------------------------------
# grafica unicamente los datos dados sin modelo, sin limites,
# y asigna la ubicaion y el nombre para el archivo de la grafica 
plot_models(x, y, None, os.path.join(CHART_DIR, "1400_01_01.png"))

# Crea y dibuja los modelos de datos
# -----------------------------------------------------------------
# fp1=coeficientes polinomiales, res1= la suma de los residuos al cuadrado 
# del ajuste de minimos cuadrados, rank1= rango de la matriz,
# sv1= valores singulares
# todo estos valores son devuletos por la funcion polyfit quien hace dichos 
# calculos a partir de los datos y la dimencion del modelo
fp1, res1, rank1, sv1, rcond1 = np.polyfit(x, y, 1, full=True)
print("Parámetros del modelo fp1: %s" % fp1)
print("Error del modelo fp1:", res1)

# poly1d construye una clase polinomial unidimensional a partir de los 
# coeficientes polinomiales calculados
f1 = sp.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = np.polyfit(x, y, 2, full=True)
print("Parámetros del modelo fp2: %s" % fp2)
print("Error del modelo fp2:", res2)
f2 = sp.poly1d(fp2)

f3 = sp.poly1d(np.polyfit(x, y, 3))
f10 = sp.poly1d(np.polyfit(x, y, 10))

# A pesar de que se admina un grado de polinomio 'n' la funcion hace un ajuste
# polinomial si el grado del polinomio esta mal condicionado, es decir,
# que el grado del polinomio es demaciado alto, por lo que este se ajusta
# al maximo posile "en este caso este solo devolvera hasta el grado 53"
f100 = sp.poly1d(np.polyfit(x, y, 100))

# Se grafican los modelos
# -----------------------------------------------------------------
# se grafica el modelo de grado 1 sin limites(proyeccion)
# en la ruta y con el nombre indicado
plot_models(x, y, [f1], os.path.join(CHART_DIR, "1400_01_02.png"))

# se grafican simultaneamente los modelos de grado 1 y 2 
# sin limites(proyeccion) en la ruta y con el nombre indicado
plot_models(x, y, [f1, f2], os.path.join(CHART_DIR, "1400_01_03.png"))

# se grafican simultaneamente los modelos de grado 1,2,3,10
# sin limites((proyeccion)), en la ruta y con el nombre indicado
plot_models(x, y, [f1, f2, f3, f10, f100], os.path.join(CHART_DIR, "1400_01_04.png"))

# Ajusta y dibuja un modelo utilizando el conocimiento del punto
# de inflexión
# -----------------------------------------------------------------
#calcula el punto de inflexion en la semana 15
inflexion = 15 * 7 

#toma solo los datos que hay hasta el punto de inflexion
xa = x[:int(inflexion)]
ya = y[:int(inflexion)]

# toma los datos desde el punto de inflexion en adelante
xb = x[int(inflexion):]
yb = y[int(inflexion):]

# Se grafican dos líneas rectas
# -----------------------------------------------------------------
# se construye el modelo de una dimension para los datos que hay hasta el 
# punto de inflexion
fa = sp.poly1d(np.polyfit(xa, ya, 1))

# se construye el modelo de una dimesion para los datos que hay desde 
# el punto de inflexion en adelante 
fb = sp.poly1d(np.polyfit(xb, yb, 1))

# Se presenta el modelo basado en el punto de inflexión
# -----------------------------------------------------------------
# se grafican los modelos de inflexion construidos anteriormente 
# sin limites(sin proyeccion), en la ruta y con el nombre indicado
plot_models(x, y, [fa, fb], os.path.join(CHART_DIR, "1400_01_05.png"))
# Función de error
# -----------------------------------------------------------------
# calcula la suma de las diferencias entre el punto y el modelo elevadas al 2
def error(f, x, y):
    return np.sum((f(x) - y) ** 2)

# Se imprimen los errores
# -----------------------------------------------------------------
print("Errores para el conjunto completo de datos:")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

# solo se toman los datos que hay despues del punto de 
# inflexion para calcular el error
print("Errores solamente después del punto de inflexión")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))
#calcula el error unicamente de los modelos construidos a partir del punto de inflexion
print("Error de inflexión=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))

# Se extrapola de modo que se proyecten respuestas en el futuro
# -----------------------------------------------------------------
# se grafican todos los modelos simultaneamente con 
# limites(con proyeccion a futuro) especificamente hasta la semana 40
# en la ruta y con el nombre indicado para su almacenamiento 
plot_models(
    x, y, [f1, f2, f3, f10, f100],
    os.path.join(CHART_DIR, "1400_01_06.png"),
    mx=np.linspace(0 * 7, 40 * 7, 100),
    ymax=250, xmin=0 * 7)

print("Entrenamiento de datos únicamente despúes del punto de inflexión")

# se construyen nuevamente los modelos de 1,2,3,10,100 dimensiones pero
# a partir de unicamente los datos que hay desde el punto de inflexion en adelante
fb1 = fb
fb2 = sp.poly1d(np.polyfit(xb, yb, 2))
fb3 = sp.poly1d(np.polyfit(xb, yb, 3))
fb10 = sp.poly1d(np.polyfit(xb, yb, 10))
fb100 = sp.poly1d(np.polyfit(xb, yb, 100))

#se calcula el error de estos nuevos modelos construidos desde el punto de inflexion
print("Errores después del punto de inflexión")
for f in [fb1, fb2, fb3, fb10, fb100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

# Gráficas después del punto de inflexión
# -----------------------------------------------------------------
# se grafican los modelos construidos solo con los datos que se encuentran
# desde el punto de inflexion en adelante, con limites(proyeccion a futuro)
# especificamente hasta la semana 40, en la ruta y con el nombre de archivo indicado
plot_models(
    x, y, [fb1, fb2, fb3, fb10, fb100],
    os.path.join(CHART_DIR, "1400_01_07.png"),
    mx=np.linspace(0 * 7 , 40 * 7, 100),
    ymax=250, xmin=0 * 7)

# Separa el entrenamiento de los datos de prueba
# -----------------------------------------------------------------

#calcula el 30% de la cantidad de datos despues del punto de inflexion
frac = 0.3
split_idx = int(frac * len(xb))

# se crea un rango de posiciones desde 0 hasta 
# la logngitud de xb(datos despues del punto de inflexion)
# random.permutation permite mezclar todos lo elementos de una lista al azar
shuffled = sp.random.permutation(list(range(len(xb))))

#se toma el primer 30% de las posiciones y se ordenan 
test = sorted(shuffled[:split_idx]) 
# se toma el % restante  y de igual forma se ordenan dichas posiciones
train = sorted(shuffled[split_idx:])

# se crean los nuevos modelos
# a partir de los datos que hay despues del punto de inflexion 
# solo tomando las posiciones calculadas en la lista train
fbt1 = sp.poly1d(np.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(np.polyfit(xb[train], yb[train], 2))

#se muestran el polinomio generado a traves de poly1d
print("fbt2(x)= \n%s" % fbt2)
print("fbt2(x)-100,000= \n%s" % (fbt2-100000))
fbt3 = sp.poly1d(np.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(np.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(np.polyfit(xb[train], yb[train], 100))

# se calcula el error de los nuevos modelos creados 
print("Prueba de error para después del punto de inflexión")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

# se grafican los nuevos modelos a partir del punto de inflexion y las
# posiciones de entrenamiento, con limites (proyeccion a futuro),
# precisamente hasta la semana 40,con la ruta y el nombre de archivo indicado  
plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100],
    os.path.join(CHART_DIR, "1400_01_08.png"),
    mx=np.linspace(0 * 7 , 40 * 7, 100),
    ymax=250, xmin=0 * 7)


from scipy.optimize import fsolve
#imprime el modelo de grado 2
print(fbt2)
#imprime el modelo de grado 2 - 100000
print(fbt2 - 0)
# se hace uso de la funcion fsolve la cual a partir de el modelo dado(al cual se
# le resta el numero del cual deseamos pronosticar su tiempo de llegada)
# nota=x0 estimacion inicial, o punto de partida de la posible respuesta
# nota2= se divide el resultado obtenido en los dias de la semana (7)
# para obtener el resultado dado en dias
alcanzado_max = fsolve(fbt2 - 0, x0=240) / (7)
print("\n0 casos por dia de COVID-19 esperados en la semana %f" % 
      alcanzado_max[0])

