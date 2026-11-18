Proyecto: Detección de Alimentos en Mal Estado y Alerta de Vencimiento
1. Descripción del proyecto

Este proyecto de Inteligencia Artificial tiene como objetivo detectar alimentos en buen estado o en mal estado mediante técnicas de visión por computadora, reconocimiento de imágenes y aprendizaje por refuerzo.

El sistema permite seleccionar imágenes de frutas y verduras, identificar el alimento mediante un modelo de inteligencia artificial y analizar sus características visuales para determinar su estado.

Además, incorpora un agente de aprendizaje por refuerzo basado en Q-Learning, que registra las correcciones realizadas por el usuario para actualizar sus decisiones.

2. Objetivos del proyecto
Reconocer frutas y verduras mediante imágenes.
Identificar características visuales de los alimentos.
Determinar si un alimento se encuentra en buen estado o podrido.
Estimar la vida útil de los alimentos.
Permitir que el usuario confirme o corrija los resultados.
Implementar aprendizaje por refuerzo mediante Q-Learning.
Guardar los conocimientos aprendidos para futuras ejecuciones.
3. Tecnologías utilizadas
Python: lenguaje principal de programación.
PyTorch: framework de aprendizaje profundo.
Torchvision: procesamiento y transformación de imágenes.
timm: biblioteca para utilizar modelos preentrenados.
MobileNetV3 Small: modelo de reconocimiento de alimentos.
OpenCV: procesamiento de imágenes.
NumPy: operaciones y cálculos numéricos.
Pillow: carga y manipulación de imágenes.
Hugging Face Hub: descarga del modelo preentrenado.
Tkinter: desarrollo de la interfaz gráfica.
4. Algoritmos y agentes utilizados
4.1. Reconocimiento de alimentos

Se utiliza MobileNetV3 Small, un modelo de visión artificial preentrenado para reconocer 30 categorías de frutas y verduras.

El modelo se obtiene desde Hugging Face mediante la biblioteca timm.

4.2. Procesamiento de imágenes

Se utilizan OpenCV y NumPy para analizar características visuales de los alimentos, como:

Intensidad de los píxeles.
Variaciones de color.
Proporción de píxeles oscuros.
Características visuales relacionadas con el deterioro.
4.3. Agente de aprendizaje por refuerzo

Se implementa un agente basado en Q-Learning.

El agente contempla dos acciones principales:

Buen estado.
Podrido.

El usuario puede confirmar o corregir el diagnóstico generado por el sistema.

Las correcciones permiten actualizar los valores de la tabla Q, que se almacena en el archivo:

models/q_table.json

4.4. Estimación de vida útil

El proyecto incorpora un módulo de estimación de vida útil de los alimentos, que permite calcular una aproximación del tiempo restante según los criterios implementados.

Esta estimación es orientativa y no garantiza que un alimento sea seguro para consumir.

5. Estructura del proyecto
text
Proyecto-Deteccion-Alimentos/
│
├── models/
│   └── q_table.json
│
├── src/
│   ├── agente/
│   │   └── q_learning.py
│   │
│   ├── estimacion/
│   │   └── vida_util.py
│   │
│   ├── utils/
│   │
│   └── vision/
│       ├── procesamiento.py
│       └── reconocimiento.py
│
├── interfaz/
│   └── app.py
│
├── datos/
│   └── imagenes_prueba/
│
├── pruebas/
│   ├── probar_vision.py
│   ├── probar_agente.py
│   ├── probar_vida_util.py
│   ├── probar_reconocimiento.py
│   └── probar_integracion.py
│
├── main.py
├── requirements.txt
└── README.md

6. Requisitos previos

Para ejecutar el proyecto se necesita:

Windows 10 o Windows 11.
Python compatible con las versiones de PyTorch y torchvision utilizadas.
Visual Studio Code.
Conexión a Internet para instalar las dependencias y descargar el modelo preentrenado.

Se recomienda utilizar Python 3.12 si se presentan problemas de compatibilidad con Python 3.14.

7. Instalación del proyecto
Paso 1. Descargar el proyecto

Descargar el archivo ZIP y extraer la carpeta Proyecto-Deteccion-Alimentos en el equipo.

Paso 2. Abrir el proyecto

Abrir Visual Studio Code y seleccionar:

Archivo → Abrir carpeta → Proyecto-Deteccion-Alimentos.

Paso 3. Abrir la terminal

Seleccionar:

Terminal → Nuevo terminal.

Verificar que la terminal se encuentre en la carpeta principal del proyecto.

Paso 4. Crear el entorno virtual

Ejecutar:

bash
python -m venv .venv

Paso 5. Activar el entorno virtual

Para PowerShell:

powershell
.\.venv\Scripts\Activate.ps1


Para CMD:

cmd
.venv\Scripts\activate.bat

Paso 6. Actualizar pip
bash
python -m pip install --upgrade pip

Paso 7. Instalar las dependencias
bash
python -m pip install -r requirements.txt

Paso 8. Comprobar las instalaciones
bash
python -c "import cv2, numpy, torch, torchvision, timm; from PIL import Image; import huggingface_hub; print('Todas las librerías están instaladas correctamente')"

8. Descarga del modelo preentrenado

El modelo MobileNetV3 Small se descarga automáticamente desde Hugging Face cuando se utiliza por primera vez, siempre que el código de reconocimiento esté configurado para realizar dicha descarga.

No es necesario descargarlo manualmente si el programa ya realiza este proceso.

Se necesita conexión a Internet durante la primera descarga.

En las siguientes ejecuciones se utilizará la copia almacenada en la caché local, siempre que esté disponible.

9. Ejecución del proyecto

Una vez instalado todo y activado el entorno virtual, ejecutar:

bash
python interfaz/app.py


Se abrirá la interfaz gráfica del sistema.

Desde ella se podrá:

Seleccionar una imagen de una fruta o verdura.
Identificar el alimento mediante el modelo de reconocimiento.
Analizar sus características visuales.
Obtener un diagnóstico sobre su estado.
Confirmar o corregir el resultado.
Registrar la corrección en el agente Q-Learning.
10. Ejecución de las pruebas

Las pruebas se pueden ejecutar individualmente desde la terminal.

Prueba 1. Procesamiento de imágenes
bash
python pruebas/probar_vision.py

Prueba 2. Agente Q-Learning
bash
python pruebas/probar_agente.py

Prueba 3. Estimación de vida útil
bash
python pruebas/probar_vida_util.py

Prueba 4. Reconocimiento de alimentos
bash
python pruebas/probar_reconocimiento.py

Prueba 5. Integración de componentes
bash
python pruebas/probar_integracion.py

11. Archivo requirements.txt

El proyecto utiliza las siguientes dependencias:

text
numpy
opencv-python
Pillow
torch
torchvision
timm
huggingface_hub

12. Consideraciones importantes
Es necesario mantener activado el entorno virtual para ejecutar el proyecto con sus dependencias.
La primera ejecución puede tardar debido a la descarga del modelo preentrenado.
El archivo models/q_table.json almacena los valores aprendidos por el agente.
Las correcciones del usuario pueden modificar el comportamiento del agente según la implementación de Q-Learning.
El reconocimiento visual no garantiza que un alimento sea seguro para consumir.
La estimación de vida útil es aproximada y no sustituye las recomendaciones de seguridad alimentaria.
13. Conclusión

El proyecto integra técnicas de Inteligencia Artificial, visión por computadora y aprendizaje por refuerzo para desarrollar un sistema capaz de reconocer alimentos, analizar su estado y registrar correcciones realizadas por el usuario.

La combinación de MobileNetV3 Small y Q-Learning permite integrar reconocimiento de imágenes con un mecanismo de aprendizaje basado en la retroalimentación del usuario.

El objetivo es desarrollar una herramienta de apoyo para la identificación y evaluación visual de frutas y verduras, contribuyendo a una mejor gestión y aprovechamiento de los alimentos.