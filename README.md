# Breed Dogs Recognition

Created: Jun 11, 2020 8:17 PM
Related to Projects (Page): https://www.notion.so/Algoritmo-que-detecte-la-raza-de-un-perro-en-base-a-la-foto-5351b02e683047b2abe790323ecea6fa

# Tabla de Contenidos

# Descripción

Este proyecto busca poder identificar la raza de un perro que se encuentre en una imagen, así mismo se busca que se detecte la existencia o no de un perro. Inicialmente es para perros, pero luego se podria incorporar la detección de gatos, no debiera ser necesario identificar la raza del gato. Entonces las salidas posibles deberían ser :  Raza de Perro, Gato, No hay perro ni gato.

Este proyecto puede complementarse en un futuro al proyecto buscomiperro.cl, para asi que el algoritmo detecte automaticamente la raza del perro, y asi aumentar las probabilidades de detección de coincidencias.  Así mismo se evitara que se suban imágenes que no correspondan a un perro o a un gato.

# Procedimiento

Inicialmente, creo que el procedimiento será el siguiente (probablemente sufra cambios en el camino):

1. Encontrar Base de Datos de entrenamiento.  ✅
    1. Imágenes  de Perros y sus Razas ✅
    2. Debe estar la raza mestizo ✅ (En realidad se debe clasificar como la )
    3. En un futuro agregar el label Gato ✅
2. Análisis exploratorio y Feature Ingeeniering ✅
    1. Una pequeña observación de las imágenes (Ej: Tamaño) ✅
    2. Poder Concatenar más datos, como gatos o mestizos, quizá  cambiar algunas razas a español, etc. ✅
3. Modelo Propio 
4. Búsqueda de modelos pre entrenados para poder hacer Transferlearning ✅
5. Implementación de los modelos ✅
6. Validación
    1. Crear Carpeta con imágenes  de validación
7. App Deployement
    1. Template (Html, CSS)
    2. Incorporar el modelo con Flask
    3. Subirlo a algún servidor  (GCP, AWS)

# 1. Base de Datos

[datasets](https://www.notion.so/8055a68808744e79b824be68886ea5eb)

Quizá sea mejor no agregar la raza mestiza, ya que muchos de estos se parecen a una raza, de hecho, la raza mestiza es una union de razas, entonces mejor que el algoritmo lo clasifique como la raza con mayor influencia. 

Se debe agregar las imagenes de gatos al dataset junto con el label "Gato"

Para detectar que no hay ni perros ni gatos, agregaré también imágenes sin perros ni gatos y le pondré un label tipo "Vacio". Googlié y esto es una buena practica, ya que por otro lado, simplemente podría hacer uso de un threshold para cuando las probabilidades estén por debajo, marcar estos como "Vacio", pero esto puede generar falsos positivo. Para mas info : [https://stackoverflow.com/questions/37713674/machine-learning-classifying-algorithm-with-unknown-class](https://stackoverflow.com/questions/37713674/machine-learning-classifying-algorithm-with-unknown-class)

Conclusiones:
-  Usar Stanford Dataset Para razas
-  Añadir a este dataset la imagen de gatos con label "Gato"
-  Añadir imágenes aleatorias sin perros ni gatos con label "Vacio"
-  Dejar que los mestizos sean clasificados con la raza predominante

### Dataset Gatos

Una vez descargado el dataset de gatos, debo poder crear un csv o dataframe del estilo nombre_archivo |    Gato.   Para esto crearé un script que lo haga, así no tengo que hacerlo manualmente.

Cada raza de perro tiene un promedio de 85 +- 13  imagenes , por lo que para los gatos tomaré solo 100 fotos, y para el data set vacío debo buscar 80-100 fotos

### Dataset Vacío

Para esto sacaré imágenes de las siguientes cosas comunes que podrian subir:

- 20 personas
- 10 plazas
- 10 calles
- 10 interior casas
- 20 otros animales (osos, aves, etc.)
- 5 comidas
- 10 random
- 10 objetos
- Total = 95

Una vez descargadas las fotos necesarias para esto, utilice el script realizado anteriormente para poder agregarles un label y obtener un csv. Ahora podre finalmente unir todos los datos, y así tener el csv Final "labels.csv"

Cambié el nombre "Vacio" por "No detectado"

### To do Base de datos

- [x]  Usar Stanford Dataset Para razas
- [x]  Añadir a este dataset la imagen de gatos con label "Gato"
- [x]  Añadir imágenes aleatorias sin perros ni gatos con label "Vacio"
- [x]  Dejar que los mestizos sean clasificados con la raza predominante
- [x]  Unir labels e imágenes

# 2. Análisis Exploratorio y Preparación de Datos

```python
BMP_AnalisisExploratorio.ipynb
```

Antes de comenzar con el Análisis exploratorio, cambié la extension jpeg a jpg de las imagenes de la carpeta "vacio" o "No detectado".    

### To do Análisis Exploratorio

- [x]  Cargar Imagenes
- [x]  Convertirlos a array
- [x]  Explorar imagenes
- [x]  Igualar las dimensiones

Dimensiones de las imágenes  de Entrenamiento:

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled.png)

Debido a que la dimensión que más se repite es (375, 500), se intentará adaptar todas las imágenes  a esta dimension.

Este trabajo se ha realizado exitosamente en el archivo correspondiente a esta sección.

Tomaria mucho trabajo manual el convertir las razas a español, ya que normalmente algunas también se utilizan en inglés, otra punto que se podría dejar para un futuro es el quitarle la palabra "dog" o incluso eliminar algunas razas, como "giant_schnauzer", y dejarla solo como "schnauzer"

A futuro: Limpiar los nombres de las razas, traducir algunas, quitarles ciertas palabras o incluso eliminar algunas razas muy poco comunes.

**Dim Final : (375,500)**

### Organizar archivos con imágenes

En los pasos posteriores, utilizaré la libreria keras y su función Image Data Generator para poder recorrer todo el dataset, sin tener que convertir primero las imagenes a matrices. Al hacerlo con ImageDataGenerator estaré ahorrando un montón de memoria, disminuyendo los costos computacionales.

Existen diversas formas para acomodar los datos a como ImageDataGenerator los necesita, yo escogí la siguiente, ya que pude ver que era una de las mas utilizadas y organizadas. Los archivos deberían quedar de la forma:

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%201.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%201.png)

Para esto, hare un pequeño script, actualmente lo que tengo es :

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%202.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%202.png)

La única  carpeta que importa de este ultimo es features_resized, que si recordamos es donde se logró juntar todas las imágenes  (razas, gatos, vacio(otros)).  En resumen, tenemos solo una carpeta con un montón de imágenes, la cual tenemos que convertirla a algo del estilo de la primera imagen mostrada. Para esto, lo que haré en el script será : 

1. Separar labels entre train y validation set. Debo fijarme en la división sea equitativa, o que tenga todas las clases.
2. Tanto en train como en validation
    1. For clase:
        1. Crear carpeta con el nombre "clase"
    2. For train_label:
        1. obtener la clase
        2. crear una copia de la imagen en la carpeta con nombre de la clase

Script creado : organize_folders.py

[https://gist.github.com/diegulio/8de972e8896c3a620c84c1bbf0d15d1e](https://gist.github.com/diegulio/8de972e8896c3a620c84c1bbf0d15d1e)

Finalmente, luego de aplicar el script, el directorio a quedado de la forma:

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%203.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%203.png)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%204.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%204.png)

Por lo que ahora podemos utilizar Image Data Generator para poder alimentar las imagenes al modelo, además de facilitar el uso de Data Augmentation.

IMPORTANTE: Hasta ahora han habido dos tediosos pasos que he realizado:
1. Cambiar tamaño de las imagenes para que todas queden del mismo tamaño
2. Dividir entre train y validation(test)
3. Crear los directorios de una forma especial
Estos dos procesos podian hacerse de manera mucho mas fácil con Data Image Generator, utilizando los parámetros 
1. resize_image
2. validation_split  
3. la función flow from dataframe .
Esto no lo hice derechamente porque desconocía estos parámetros y funciones , pero ahora se añade a cosas aprendidas

# 3. Convolutional Neural Networks

Con motivos de aprendizaje, crearé una red neuronal por mi mismo, probablemente el accuracy sea desastroso en contraste con aquellas redes con parámetros pre entrenados, pero servirá para practicar y aprender!

# 4. Búsqueda modelos para Transfer Learning

A continuación se recaudarán modelos para poder realizar transferlearning y no estimar los parámetros desde 0.

Probaré los siguientes 3 modelos top para la clasificación de imágenes, estos son:

- VGG-16
- ResNet50
- Inception V3

### VGG-16

- Convolutional Layers : 13
- Pooling Layers : 5
- Dense Layers : 3

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%205.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%205.png)

***Arquitectura***

1. **Input**: Image of dimensions (224, 224, 3).
2. **Convolution Layer Conv1:**
    - Conv1-1:
        - 64 filters
        - padding = Same
        - kerne_size = 3x3
        - Activation: relu

        padding = same es para que la salida de este layer sea del mismo tamaño que la entrada, osea (224,224,3) 
        kernel_size: significa que las dimensiones de el filtro son de 3x3

        Output :  (224,224,64)

    - Conv1-2:
        - Mismos parámetros que Conv1-1

        Output :  (224,224,64)

    - Max Pooling:
        - pool_size : 2x2
        - stride : 2,2

        pool_size: Genera un 'filtro' de 2x2 que se queda unicamente con el máximo de los 4 valores
        stride: recordar que stride es un 'salto' del filtro, en este caso se salta 2 pixeles hacia al lado y hacia abajo

    Output :  (112,112,64)

3. **Convolution layer Conv2:**  Se incrementan los filtros a 128
    - Input Image dimensions: (112,112,64)
    - Conv2-1:
        - 128 filters
        - kernel_size = 3x3
        - padding = same
        - activation = relu

        Output : (112,112,128)

    - Conv2-2:
        - Mismos parámetros que Conv2-1

        Output : (112,112,128)

    - Max Pooling:
        - pool_size : 2x2
        - stride : 2,2

        Output : (56,56,128)

4. **Convolution Layer Conv3:** Se aumentan los filtros y se agrega un layer convolucional
    - Input Image dimensions: (56,56)
    - Conv3-1:
        - 256 filters
        - kernel size 3x3
        - padding same
        - activation relu

        Output : (56,56,256)

    - Conv3-2:
        - Mismos parámetros que 3-1

        Output : (56,56,256)

    - Conv3-3:
        - Mismos parámetros que 3-1

        Output : (56,56,256)

    - Max Pooling :
        - poolsize 2x2
        - strides 2x2

        Output : (28,28,256)

5. **Convolution Layer Conv4:** 512 filtros
    - Conv4-1:
        - 512 filters
        - kernel size 3x3
        - padding same
        - activation relu

        Output : (28,28,512)

    - Conv4-2:
        - Mismos params que 4-1

        Output : (28,28,512)

    - Conv4-3:
        - Mismos params que 4-1

        Output : (28,28,512)

    - Max Pooling:
        - poolsize 2x2
        - strides 2x2

        Output : (14,14,512)

6. **Convolution Layer Conv5:**  Lo mismo que Conv4
    - Conv5-1:
        - 512 filters
        - kernel size 3x3
        - padding same
        - activation relu

        Output : (14,14,512)

    - Conv5-2:
        - Mismos params que 5-1

        Output : (14,14,512)

    - Conv5-3:
        - Mismos params que 5-1

        Output : (14,14,512)

    - Max Pooling:
        - poolsize 2x2
        - strides 2x2

        Output : (7,7,512)

7. **Flatten Layer**

    Flatten es para aplastar el output anterior y dejarlo en un vector, por lo que resultaria en un vector de dimension 7*7*512 = 25088

    Output = (1,25088)

    ![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%206.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%206.png)

8. **Fully Connected/Dense FC1**: 
    - 4096 nodes
    - Output = (1, 4096)
9. **Fully ConnectedDense FC2**: 
    - 4096 nodes
    - Output = (1, 4096)
10. **Fully Connected /Dense FC3**:  
    - 1000 nodes (Depende del numero de clases que se esté clasificando)
    - Output = (1, 1000)
11. Softmax
    - Acá resulta en un vector de 1000 clases con las probabilidades de cada clase

Para nuestro problema, existen 122 clases : 120 razas + gato + no_detectado

### Inception

Este tipo de redes contienen un módulo llamado Inception Module (de acá su nombre), el cual consta de concatenar distintos módulos  convolucionales (diferentes filtros) incluyendo maxpoolings, y asi poder crear una red más profunda o compleja reduciendo el número de parámetros.

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%207.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%207.png)

Esta red al ser mucho más grande no se expondrá paso a paso, simplemente se mostrará una representación gráfica.

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%208.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%208.png)

más info: [https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/?utm_source=blog&utm_medium=top4_pre-trained_image_classification_models](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/?utm_source=blog&utm_medium=top4_pre-trained_image_classification_models)

### EfficientNet

Este modelo ha causado mucho revuelo, elaborado por Google, en donde proponen un método llamado Coumpound Scaling. Normalmente los modelos convencionales escalan las dimensiones arbitrariamente y suman y suman capas, esto para poder conseguir mejores resultados, es por esto que, por ejemplo, luego de VGG-16 se creó VG-19, con mejores resultados pero sacrificando costo computacional al agregar 3 capas o layers, en cambio, el modelo en EfficientNet propone escalar las dimensiones de manera uniforme con una cantidad fija, lo que llevaría, según lo propone el paper, a un redimiendo mejor. Esto quiere decir, que en vez de centrarse en una sola variable que aumentaria la eficiencia del modelo, es mejor ir mejorando en menor medida todas las variables que pueden traer mejores resultados, como lo son: Ancho, profundidad y la resolución de la imagen.

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%209.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%209.png)

Mientras que la mayoria de los modelos escalan como en (b) y en (d) enfocandose sólo en uno de los factores widht,deep o resolution, EfficientNet propone Compound Scaling, el cual uniformemente escala las distintas variables involucradas.

**Arquitectura**

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2010.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2010.png)

más info: [https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2011.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2011.png)

En la imagen se pueden observar las aplicaciones de algunos modelos  pre entrenados en  ImageNet en base a la cantidad de parámetros  y precisión. 

# 5. Aplicación de los Modelos

En esta sección se implementarán los modelos utilizados, exponiendo sus parámetros y sus correspondientes indicadores de precisión.

BMP_models.py

En BMP_models se puede encontrar el proceso entero, debido a que no se contaba con un train set de considerable tamaño, es que se optó por aumentar los datos ¿Cómo? A cada batch de imagenes se les aplicó una serie de procesos como: rotación, desplazamiento, zoom, flips, etc. Esto además de mejorar la precisión, logra que no se produzca mayor overfitting.

Rápida implementación: Si bien esto no se mostrará a continuación, es importante destacar que en un comienzo, primero se aplicaron todos estos modelos a secas, sin preocuparse por aumentar los datos, mejorar los hiperparámetros, elegir el método de optimización,etc.  si no que simplemente se buscó ver la actuación de los modelos de manera rápida, para poder tener una base concreta, y en base a ellos comenzar a tunear el modelo para obtener las puntuaciones que serán mostradas a continuación. Además se debe tener en cuenta que no se intentó todo, por lo que es posible que estos resultados puedan mejorarse, y algunas ideas para llevar a cabo esto se expondrán al final de la sección.

[Implementación](https://www.notion.so/7eaa7dd739184bf2a67916ce00b1db44)

PD: Los espacios vacíos son debido a pérdidas de información debido a problemas con el Notebook de google colab.

En la tabla anterior se pueden ver los indicadores obtenidos con las distintas estructuras y pesos pre entrenados propuestos en un inicio. Podemos darnos cuenta que VGG16 y EfficientNet7 lo hacen muy mal, esto tiene que ver simplemente debido a la estructura que involucra cada uno. El hecho de que Inception lo haya hecho mucho mejor, me llevó a ir variando de a poco su estructura  y así conseguir distintos resultados. Finalmente he guardado 4 modelos, todos Inceptionv3, si bien son 6 modelos de inception, por problemas con el notebook en Colab, se borró la información de uno de ellos (Inception 1 y 2) por lo que no pude quedarme con el modelo, pero lo dejaré pasar ya que luego tuve unos resultados mejores.

El flujo que seguí, en cuanto al variar la infraestructura, algunos parámetros  y aumento de la data, además de algunas conclusiones se pueden encontrar en:

BMP_models.py

Ahora debemos pasar a la validación, para esto debemos nuevamente organizar imágenes para evaluar la implementación de los 5 modelos extraidos en esta sección. Esto pude haberlo hecho antes, en el paso de extracción y preparación de datos. 

# 6. Validación

Debo crear un set de validación, lo ideal de esto es que se parezca lo más posible a las imágenes que pueda enfrentar el algoritmo en producción. Una opción es elaborar un set de validación, pero esto me tomaría mucho tiempo. Debido a que el dataset inicial de las razas lo obtuve de una competencia en kaggle, lo que haré será hacer un submission del test con mi modelo. El problema de esto es que no podré evaluar los casos "Gato" y "No detectado" ya que estos no estaban en el dataset inicial. Sin embargo podré tener una buena aproximación de la efectividad de los distintos modelos. Más tarde, podré probar mis propias imágenes con el mejor modelo resultante y poder obtener algunos insight para mejorar.

### Kaggle

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2012.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2012.png)

[Validación](https://www.notion.so/9f27e2d685da481cab40cc56d19a20bd)

Recordar que esta evaluación fue realizada con el dataset de evaluación de donde se obtuvieron los datos de entrenamiento (Kaggle). Se debe recordar que nosotros creamos 2 clases más (Gato y No detectado), por lo que el score que obtuvimos en kaggle está siendo subestimado por un poquito debido a que cada vector de predicción no sumará precisamente 1, ya que Gato y No detectado aportaban a esa suma.

Usamos esta validación ya que tiene muchos datos **(10357 imágenes)**, estaban ya con label y nos ahorramos el tiempo de colección de los datos. El lado negativo es que no testeamos como funciona el modelo contra gatos y No detectado.

Estoy un poco asombrado de que el modelo 3 (Inception 3) haya sido el ganador, ya que sus indicadores de accuracy de train y testeo no erán mejores que los de el modelo 4 y 5.

Finalmente, escogemos entonces el modelo 3 ya que es aquel con menor Loss (Multi Class Log Loss) y lo que haremos esa probar con imágenes nuestras, para esto, pediré fotos a mis conocidos y sacaré otras de grupos de facebook de Chile, para poder ver que estén las razas Chilenas, además ver que pasa con los kiltros o mestizos, etc.

Inception_3 Modelo Escogido

Acá pondré algunos ejemplos junto con sus conclusiones, para probar tu propia img puedes encontrar la función para esto en:

BPM_validation.py

A continuación pondré algunos casos interesantes encontrados

### Predicción normal con razas parecidas

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2013.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2013.png)

Podemos ver que acertó, ya que efectivamente la raza de esta perrita es collie, aún así identificó que su segunda opción es la raza shetland_sheepdog, que si se busca, es prácticamente identica a la collie, con algunas variaciones. Veamos otra imagen en distinta posición de este mismo ejemplar 

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2014.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2014.png)

Podemos ver que ahora se voltearon las probabilidades, aún así no es una mala predicción del todo, ya que ambas razas se parecen un montón.

### Razas inexistentes en train set

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2015.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2015.png)

En este caso, se detectó con una gran probabilidad de que este perrito es un bulldog francés, desafortunadamente este perrito es bulldog inglés. Luego de una búsqueda por la explicación de esto, se encontró que el dataset inicial no contenía la raza bulldog inglés.

Una solución para esto es agregar las razas en el train set ó agrupar labels (Ej: bulldog francés convertirlo simplemente a bulldog). La segunda opción sólo es valida para algunas razas parecidas (e.g para poodle toy - poodle miniature).

Algunas razas inexistentes encontradas son:

- Bulldog Inglés
- Dalmata
- Akita

### Razas Agrupables

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2016.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2016.png)

Este es un caso parecido con el bulldog, los poodle, al existir distintos tipos pueden causar confusión y una buena decisión sería agrupar esta solución en sólo poodle con 100% prob.

### Gatos

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2017.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2017.png)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2018.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2018.png)

Estos dos lo hace muy bien.

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2019.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2019.png)

Podemos ver que en este ejemplo le cuesta un montón, puede ser debido a la posición del animal y de los colores que piensa que puede ser un terrier. 

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2020.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2020.png)

En este caso no lo hace muy bien, ya que cree que no hay ni un perro ni gato en la foto. Esta muy confiado de que no hay nada, pero almenos presenta un poco de probabilidad de ser un gato.

### No detectado

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2021.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2021.png)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2022.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2022.png)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2023.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2023.png)

Acá destacamos que acierta con No detectado, y no piensa que es un gato.

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2024.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2024.png)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2025.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2025.png)

### Perros y gatos

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2026.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2026.png)

Acá predomina el golden retriever, lo que detecta con prob 1

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2027.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2027.png)

Al parecer (no podemos confirmarlo con tan poco ejemplos), el modelo es mejor clasificando perros, ya que cuando hay perros y gatos detecta al perro. Esto puede deberse a que existen muchos más perros que gatos en el train set

### 2 Perros

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2028.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2028.png)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2029.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2029.png)

Podemos darnos cuenta que identica a ambos en los primeros lugares, pero prioriza 1.

### Ejemplos varios (Facebook)

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2030.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2030.png)

Vemos que este es mestizo, por lo que intenta clasificarlo en algunas razas presentes

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2031.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2031.png)

Lo mismo acá, si buscan las razas, verán que todas son muy semejantes al ejemplar en cuestión

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2032.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2032.png)

Acá podemos ver que acierta a la raza que la dueña reconoce.

## Conclusiones Validación

Hemos terminado eligiendo el modelo 3 como el mejor gracias a los indicadores entregador al validar con 10537 imágenes  de distintas razas de Kaggle, por último hemos probado algunos casos especiales individuales y reales para probar que tan bien generaliza nuestro modelo, y poder ver en que falla, como en problema con la posición del animal (la información que se presenta en la imagen)

Por lo visto el modelo es bastante decente y podemos quedarnos conformes de que entrega buenos resultados, aún así es mejorable aplicando distintas técnicas. A continuación expondremos algunas cosas que podemos mejorar.

# Mejoras propuestas

A continuación se expondrán algunas mejoras al modelo o proceso que podrían implementarse y mejorar el resultado, alguna de ellas son muy importantes, como el agregar razas que son comunes en Chile (Dálmata , bulldog inglés, etc.)

1. Agregar razas: Akita, Dálmata, bulldog inglés, 
2. Agrupar Razas: Poodle
3. Mejorar arquitectura: Claramente esto es algo que puede hacerse con más tiempo, poder proponer nuevas estructuras que puedan resultar en unos mejores indicadores
4. Aumentar Data de Entrenamiento.
5. Traducir razas

# 7. App Deployement

## Template

He hecho a mano el index de la aplicación, es bastante simple ya que sólo se necesita una imagen, quedó algo así

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2033.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2033.png)

Y luego de hacer la predicción se debería mostrar algo así

![Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2034.png](Breed%20Dogs%20Recognition%200a51636cb63943939aa4a833b9ddfba0/Untitled%2034.png)

# Referencias

- [https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/](https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/)
- [https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16 is a convolution neural,competition in 2014.&text=It follows this arrangement of,by a softmax for output](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16%20is%20a%20convolution%20neural,competition%20in%202014.&text=It%20follows%20this%20arrangement%20of,by%20a%20softmax%20for%20output).
- [https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/?utm_source=blog&utm_medium=top4_pre-trained_image_classification_models](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/?utm_source=blog&utm_medium=top4_pre-trained_image_classification_models)
- [https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
- [https://keras.io/api/preprocessing/image/](https://keras.io/api/preprocessing/image/)
- [https://keras.io/api/models/model_training_apis/](https://keras.io/api/models/model_training_apis/)