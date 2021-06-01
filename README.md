# Juego de la Vida de Conway

Análisis comparativo de versiones seriales y paralelas del Juego de la Vida de Conway, implementadas en C++ y desarrolladas en Visual Studio 2019, correspondiente a la Tarea 2 del curso Computación en GPU (CC7515). Las soluciones implementadas corresponden a una versión serial en CPU y dos paralelas en GPU, usando CUDA y OpenCL respectivamente.

## Descripción
A continuación se presentará una descripción de los directorios.

### DataGenerator
Contiene scripts en lenguaje Python que permiten ejecutar las distintas versiones del código del juego y almacenar sus resultados en archivos ```.csv```.

### Data
Carpeta de destino para los archivos ```.csv``` generados con los scripts de ```DataGenerator```.

### GraphsGenerator
Contiene scripts en lenguaje Python que permiten generar gráficos comparativos utilizados en el informe a partir de los datos obtenidos en la carpeta ```Data```.

### images
Carpeta de destino para los datos generados con los scripts de ```GraphsGenerator```.

### GameOfLifeCPU
Contiene los archivos de solución y el código generado para la implementación serial en CPU.

Al compilar genera una ejecutable que recibe de entrada el ancho del mundo, el largo del mundo y el número de iteraciones que tendrá el juego. Si se omiten estos parámetros, se usan valores predeterminados. Retornando el tiempo de ejecución en nanosegundos

### GameOfLifeCUDA
Contiene los archivos de solución y el código generado para la implementación paralela en GPU con CUDA.

Al compilar genera una ejecutable que recibe de entrada el ancho del mundo, el largo del mundo, el número de iteraciones que tendrá el juego y la cantidad de threads que tendrá cada bloque. Si se omiten estos parámetros, se usan valores predeterminados. Retornando el tiempo de ejecución en milisegundos

### GameOfLifeCUDA2
Contiene los archivos de solución y el código generado para la implementación paralela en GPU con CUDA que utiliza *if statements* en el kernel.

Al compilar genera una ejecutable que recibe de entrada el ancho del mundo, el largo del mundo, el número de iteraciones que tendrá el juego y la cantidad de threads que tendrá cada bloque. Si se omiten estos parámetros, se usan valores predeterminados. Retornando el tiempo de ejecución en milisegundos

### GameOfLifeOpenCL
Contiene los archivos de solución y el código generado para la implementación serial en GPU OpenCL.

Al compilar genera una ejecutable que recibe de entrada el ancho del mundo, el largo del mundo, el número de iteraciones que tendrá el juego y la cantidad de threads que tendrá cada bloque. Si se omiten estos parámetros, se usan valores predeterminados. Retornando el tiempo de ejecución en nanosegundos

### GameOfLifeOpenCL2
Contiene los archivos de solución y el código generado para la implementación serial en GPU OpenCL que utiliza *if statements* en el kernel.

Al compilar genera una ejecutable que recibe de entrada el ancho del mundo, el largo del mundo, el número de iteraciones que tendrá el juego y la cantidad de threads que tendrá cada bloque. Si se omiten estos parámetros, se usan valores predeterminados. Retornando el tiempo de ejecución en nanosegundos


## Instrucciones de Uso
### Compilar soluciones
Para compilar las soluciones es necesario abrir el archivo de solución ```.sln``` deseado con Visual Studio 2019 y luego configurarlo según sean las necesidades (CPU, CUDA u OpenCL).

Posteriormente buildear.

En nuestro caso al trabajar con una GPU Nvidia, los programas con CUDA se crearon a partir del template CUDA Runtime, por lo que no fue necesario más configuraciones. Mientras que para los programas con OpenCl solo fue necesario configurar los linkers y las configuraciones generales para agregar los directorios y headers necesarios.

Luego se hizo build en modo Release para plataformas x64

### Ejecutar códigos con scripts de Python
Para ejecutar los códigos con los scripts que registran los resultados es necesario buildear la solución previamente y luego ejecutar el script deseado con Python desde la carpeta del repositorio. Ejemplo : ``python DataGenerator/simple_CPU.py ``.

Los resultados estarán en el archivo ```.csv``` correspondiente.

Recordar que para los programas con OpenCL, se debe ubicar el archivo ``kernel.cl`` en el mismo directorio que el ejcutable generado. 

### Obtención de gráficos
Ejecutar con Python el script generador deseado desde la carpeta del repositorio. Ejemplo : ``python DataGenerator/serial_CPU.py ``.

Requiere tener instaladas las librerías ``numpy`` y  ``matplotlib``.

La figura estará en el archivo ```.png``` correspondiente.