
import cv2
import numpy as np


def analizar_imagen(ruta_imagen):
    """
    Carga una imagen y extrae características visuales
    básicas del alimento.
    """

    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        raise ValueError("No se pudo cargar la imagen.")

    # Redimensionar para mantener un tamaño uniforme
    imagen = cv2.resize(imagen, (300, 300))

    # Convertir de BGR a RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Calcular características
    intensidad_media = float(np.mean(gris))
    variacion = float(np.std(gris))

    # Calcular proporción de píxeles oscuros
    pixeles_oscuros = np.sum(gris < 80)
    proporcion_oscuros = float(pixeles_oscuros / gris.size)

    return {
        "intensidad_media": intensidad_media,
        "variacion": variacion,
        "proporcion_oscuros": proporcion_oscuros,
        "imagen_rgb": imagen_rgb,
        "imagen_gris": gris
    }
