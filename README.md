# Juego de la Vida de Conway

Análisis comparativo de versiones seriales y paralelas del Juego de la Vida de Conway, implementadas en C++ y desarrolladas en Visual Studio, correspondiente a la Tarea 2 del curso Computación en GPU (CC7515). Las soluciones implementadas corresponden a una versión serial en CPU y dos paralelas en GPU, usando CUDA y OpenCL respectivamente.

## Descripción
A continuación se presentará una descripción de los directorios.

### DataGenerator
Contiene scripts en lenguaje Python que permiten ejecutar las distintas versiones del código del juego y almacenar sus resultados en archivos ```.csv```.

### Data
Carpeta de destino para los archivos ```.csv``` generados con los scripts de ```DataGenerator```.

### GraphsGenerator
Contiene scripts en lenguaje Python que permiten generar los gráficos comparativos utilizados en el informe a partir de los datos obtenidos en la carpeta ```Data```.

### images
Carpeta de destino para los datos generados con los scripts de ```GraphsGenerator```.

### GameOfLifeCPU
Contiene los archivos de solución y el código generado para la implementación serial en CPU.

### GameOfLifeCUDA
Contiene los archivos de solución y el código generado para la implementación serial en GPU CUDA.

### GameOfLifeCUDA2
Contiene los archivos de solución y el código generado para la implementación serial en GPU CUDA que utiliza *if statements* en el kernel.

### GameOfLifeOpenCL
Contiene los archivos de solución y el código generado para la implementación serial en GPU OpenCL.

### GameOfLifeOpenCL2
Contiene los archivos de solución y el código generado para la implementación serial en GPU OpenCL que utiliza *if statements* en el kernel.


## Instrucciones de Uso
### Compilar soluciones
Para compilar las soluciones es necesario abrir el archivo de solución ```.sln``` deseado con Visual Studio y luego configurarlo según sean las necesidades (CPU, CUDA u OpenCL).

Posteriormente buildear.

### Ejecutar códigos con scripts de Python
Para ejecutar los códigos con los scripts que registran los resultados es necesario buildear la solución previamente y luego ejecutar el script deseado con Python.

Los resultados estarán en el archivo ```.csv``` correspondiente.

### Obtención de gráficos
Ejecutar con Python el script generador deseado. 

La figura estará en el archivo ```.png``` correspondiente.