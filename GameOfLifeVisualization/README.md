# Game of life visualization

## Descripción
Visualización del proyecto. Representa la evolucion de un conjuntos de celdas en el tiempo y segun las condiciones del juego de la vida de Conway indicadas en el [proyecto](http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA) de Marek Fiser. las celas de color negro se encuentran muertas y las grises o blancas estan con vida. 
Si esta activado el postprocessing, las celdas blancas son las que acaban de cambiar su estado muertas a vivas y las de color rojo son celdas que acaban de morir.

## Muestras

Visualizando un mundo de 1024 x 1024 en distintas escalas
![scale](../captures/gol_scale.gif)

Visualizando el juego con y sin postprocessing
![postprocessing](../captures/gol_pros.gif)

## Librerías usadas
Se uso el lenguaje C++ con la siguientes librerias:
- [Glad](https://glad.dav1d.de/) : Libreria necesaria para cargar los punteros a funciones de OpenGL. En este proyecto se usó OpenGL 3.3
- [GLFW3](https://www.glfw.org/) : Libreria usada con OpenGL que provee una API para manejar ventanas
- [GLM](https://glm.g-truc.net/0.9.9/index.html) : Libreria con funciones matematicas utiles para el uso de aplicaciones con OpenGL

## Como se instalaron las librerías
A continuación se darán los pasos con las que se pudo instalar las diferentes librerías para poder usarlas en el programa Visual Studio 2019:
- Se tiene que configurar el proyecto en VS, creando las carpetas /Libraries/include y /Libraries/lib si no se encuentran
- Seleccionar la plataforma x64 en el editor
- Ir a configuraciones del proyecto y seleccionar en Platform: All Platforms
- Ir a VC++ Directories -> Include Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/include -> ok
- Ir a VC++ Directories -> Library Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/lib -> ok
- Ir a Linker -> Input -> Additional Dependencies -> Edit -> poner en el campo de texto:
```
glfw3.lib
opengl32.lib
```
Luego para instalar las diferentes librerías:
- [Glad](https://glad.dav1d.de/) : Descargar la version OpenGL/GLAD (version 4.5 Core), abrir glad.zip -> ir a /include y copiar carpetas "glad" y "KHR" a la carpeta del proyecto /Libraries/include. Del mismo zip -> ir a /src y copiar el archivo "glad.c" en la carpeta raíz del proyecto.
- [GLFW3](https://www.glfw.org/) : Descargar, y compilar con Cmake en una carpeta build, ir a ../build/src/Debug y copiar el archivo "glfw3.lib" a la carpeta del proyecto Libraries/lib. Ir a ../include y copiar la carpeta "GLFW" a la carpeta del proyecto Libraries/include
- [GLM](https://glm.g-truc.net/0.9.9/index.html) : Descargar, descomprimir y copiar directorio que sea raíz de glm.h y pegarla en Libraries/include

## Cómo ejecutar la aplicación
Para poder ejecutar el proyecto debe tener una GPU compatible con CUDA. La aplicación se puede ejecutar desde el editor Visual Studio 2019, seleccionando la opción Release con la plataforma x64- También se incluye el ejecutable.
Se puede ejecutar la aplicacion con los parametros por defecto o entregando parametros al momento de ejecucion. En el directorio ``GameOfLifeVisualization\GameOfLifeVisualization``:
```
./GameOfLifeVisualization.exe worldWidth worldHeight threads distance
```
Donde:
- ``worlWidth`` es el ancho del mundo
- ``worlHeight`` es el largo del mundo
- ``threads`` son los threads por bloque
- ``distance`` es la distancia en la que se inicializan aleatoriamente las celdas vivas desde el centro del mundo

## Controles:

Los controles de teclado son:
- [SCAPE] Salir de la aplicación
