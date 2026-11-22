import cv2
import numpy as np

def analizar_imagen(ruta_imagen):
    """
    Analiza la imagen y extrae características abstractas de color y textura.
    En lugar de guardar píxeles, devuelve un diccionario con categorías genéricas
    para que el agente Q-Learning aprenda patrones aplicables a diferentes frutas.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return {"tono": "desconocido", "manchas": "desconocido", "textura": "desconocida"}

    # Redimensionar para homogeneizar el análisis y acelerar el proceso
    imagen = cv2.resize(imagen, (224, 224))
    
    # Convertir a espacio HSV para analizar colores de manera más efectiva
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # 1. ANÁLISIS DE COLOR (Brillo y Tono global)
    # Extraemos el canal Value (V) que representa el brillo
    brillo_medio = np.mean(hsv[:, :, 2])
    if brillo_medio < 90:
        tono = "color_muy_oscuro"
    elif brillo_medio < 130:
        tono = "color_apagado"
    else:
        tono = "color_brillante"
        
    # 2. DETECCIÓN DE MANCHAS (Áreas típicamente podridas / marrones / negras)
    # Rango de color genérico para manchas de pudrición en HSV
    rango_bajo = np.array([0, 40, 0])
    rango_alto = np.array([40, 255, 100])
    mascara_manchas = cv2.inRange(hsv, rango_bajo, rango_alto)
    
    # Calculamos qué porcentaje de la fruta está cubierta por estas manchas
    porcentaje_manchas = (np.sum(mascara_manchas > 0) / (224 * 224)) * 100
    
    if porcentaje_manchas > 15:
        manchas = "muchas_manchas"
    elif porcentaje_manchas > 5:
        manchas = "algunas_manchas"
    else:
        manchas = "sin_manchas"

    # 3. ANÁLISIS DE TEXTURA (Detección de irregularidades y arrugas)
    # Convertimos a escala de grises para detectar bordes
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos un filtro Canny para detectar líneas marcadas (textura rugosa)
    bordes = cv2.Canny(gris, 50, 150)
    densidad_bordes = (np.sum(bordes > 0) / (224 * 224)) * 100
    
    if densidad_bordes > 12:
        textura = "textura_irregular"
    else:
        textura = "textura_lisa"
        
    # Retornamos el estado abstracto al agente
    return f"{tono}_{manchas}_{textura}"

def obtener_imagenes_procesamiento(ruta_imagen):
    """
    Obtiene la imagen original y sus componentes visuales:
    RGB, canales individuales y escala de grises.

    No interviene en el diagnóstico del agente.
    """
    imagen_bgr = cv2.imread(str(ruta_imagen), cv2.IMREAD_COLOR)

    if imagen_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

    # OpenCV carga las imágenes en formato BGR.
    # Convertimos a RGB para mantener el orden correcto de los canales.
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    # Extraemos cada canal de color individualmente.
    canal_r = imagen_rgb[:, :, 0]
    canal_g = imagen_rgb[:, :, 1]
    canal_b = imagen_rgb[:, :, 2]

    # Conversión a escala de grises.
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    alto, ancho = imagen_rgb.shape[:2]

    return {
        "imagen_rgb": imagen_rgb,
        "canal_r": canal_r,
        "canal_g": canal_g,
        "canal_b": canal_b,
        "imagen_gris": imagen_gris,
        "ancho": ancho,
        "alto": alto,
        "media_r": float(np.mean(canal_r)),
        "media_g": float(np.mean(canal_g)),
        "media_b": float(np.mean(canal_b)),
        "media_gris": float(np.mean(imagen_gris))
    }
