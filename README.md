# Proyecto: Detección de Alimentos en Mal Estado y Alerta de Vencimiento

Proyecto de Inteligencia Artificial que utiliza visión por computadora, reconocimiento de imágenes y aprendizaje por refuerzo para trabajar con alimentos y registrar correcciones realizadas por el usuario.

## 1. Requisitos previos

Antes de ejecutar el proyecto, se necesita:

- Windows 10 o Windows 11.
- Python 3.14.
- Visual Studio Code.
- Conexión a Internet para instalar las dependencias y descargar el modelo preentrenado.

## 2. Estructura del proyecto

```text
Proyecto-Deteccion-Alimentos/
│
├── models/
│   └── q_table.json
│
├── src/
│   ├── agente/
│   │   └── q_learning.py
│   ├── estimacion/
│   │   └── vida_util.py
│   ├── utils/
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
```

## 3. Tecnologías y herramientas utilizadas

- **Python:** lenguaje de programación.
- **PyTorch:** framework de aprendizaje profundo.
- **Torchvision:** herramientas para trabajar con imágenes.
- **timm:** biblioteca para utilizar modelos de visión artificial.
- **MobileNetV3 Small:** modelo preentrenado para reconocer 30 categorías de frutas y verduras.
- **OpenCV:** procesamiento de imágenes.
- **NumPy:** cálculos y extracción de características.
- **Pillow:** carga y transformación de imágenes.
- **Hugging Face Hub:** descarga del modelo preentrenado.
- **Tkinter:** desarrollo de la interfaz gráfica.

## 4. Agentes y algoritmos utilizados

### 4.1. Reconocimiento de alimentos

Se utiliza MobileNetV3 Small, un modelo preentrenado que permite identificar 30 categorías de alimentos a partir de imágenes.

El modelo se obtiene desde Hugging Face mediante la biblioteca timm.

### 4.2. Procesamiento de imágenes

Se utiliza OpenCV y NumPy para analizar características visuales de las imágenes, como intensidad, variación y proporción de píxeles oscuros.

### 4.3. Agente de aprendizaje por refuerzo

Se implementa Q-Learning para actualizar los valores de las acciones según las respuestas del usuario.

El sistema contempla dos acciones:

- Buen estado.
- Podrido.

El usuario puede confirmar o corregir el diagnóstico. Los valores aprendidos se almacenan en `models/q_table.json`.

## 5. Instalación del proyecto

### Paso 1. Descargar y descomprimir

Descargar el archivo ZIP del proyecto y extraer la carpeta `Proyecto-Deteccion-Alimentos` en el equipo.

### Paso 2. Abrir en Visual Studio Code

Abrir Visual Studio Code y seleccionar:

**Archivo → Abrir carpeta → Proyecto-Deteccion-Alimentos**

### Paso 3. Abrir la terminal

En Visual Studio Code, seleccionar:

**Terminal → Nuevo terminal**

Verificar que la terminal esté ubicada en la carpeta principal del proyecto.

### Paso 4. Crear el entorno virtual

Ejecutar:

```bash
python -m venv .venv
```

### Paso 5. Activar el entorno virtual

Si la terminal es PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si la terminal es CMD:

```cmd
.venv\Scripts\activate.bat
```

Cuando esté activado, aparecerá `(.venv)` al inicio de la terminal.

### Paso 6. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### Paso 7. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

Este comando instala las librerías necesarias para ejecutar el proyecto.

### Paso 8. Comprobar las instalaciones

```bash
python -c "import cv2, numpy, torch, torchvision, timm; from PIL import Image; import huggingface_hub; print('Todas las librerías están instaladas correctamente')"
```

Si aparece el mensaje de confirmación, las dependencias principales están disponibles.

## 6. Descarga del modelo preentrenado

El modelo MobileNetV3 Small se descarga automáticamente desde Hugging Face cuando se utiliza por primera vez.

No es necesario descargarlo manualmente ni incluirlo en el archivo ZIP.

**Importante:** se necesita conexión a Internet durante la primera descarga. En las siguientes ejecuciones se utilizará la copia almacenada en la caché local, siempre que siga disponible.

## 7. Ejecución del proyecto

Con el entorno virtual activado, ejecutar:

```bash
python interfaz/app.py
```

Se abrirá la interfaz gráfica del sistema.

Desde ella se puede seleccionar una imagen, reconocer el alimento, obtener un diagnóstico y confirmar o corregir el resultado.

## 8. Ejecución de las pruebas

Las pruebas se pueden ejecutar individualmente desde la terminal.

### Prueba 1. Procesamiento de imágenes

```bash
python pruebas/probar_vision.py
```

### Prueba 2. Agente Q-Learning

```bash
python pruebas/probar_agente.py
```

### Prueba 3. Estimación de vida útil

```bash
python pruebas/probar_vida_util.py
```

### Prueba 4. Reconocimiento de alimentos

```bash
python pruebas/probar_reconocimiento.py
```

### Prueba 5. Integración de componentes

```bash
python pruebas/probar_integracion.py
```

## 9. Archivo requirements.txt

El archivo contiene las dependencias necesarias:

```text
numpy
opencv-python
Pillow
torch
torchvision
timm
huggingface_hub
```
