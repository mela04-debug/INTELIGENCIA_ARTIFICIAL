
import sys
from pathlib import Path

# Identificar la carpeta principal del proyecto
PROYECTO = Path(__file__).resolve().parents[1]

# Agregar la carpeta principal a las rutas de búsqueda de Python
sys.path.insert(0, str(PROYECTO))

from src.vision.procesamiento import analizar_imagen

# Ubicación de la fotografía
ruta_imagen = (
    PROYECTO / "datos" / "imagenes_prueba" / "manzana.jpg"
)

try:
    resultado = analizar_imagen(str(ruta_imagen))

    print("ANÁLISIS DE IMAGEN")
    print("--------------------------")
    print("Intensidad media:", resultado["intensidad_media"])
    print("Variación:", resultado["variacion"])
    print(
        "Proporción de píxeles oscuros:",
        resultado["proporcion_oscuros"]
    )

    print("\nImagen procesada correctamente.")

except Exception as error:
    print("Error:", error)
