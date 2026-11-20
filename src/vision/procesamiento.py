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