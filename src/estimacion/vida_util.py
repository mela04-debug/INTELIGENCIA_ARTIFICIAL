# Rangos de conservación de referencia (en días) por alimento.
# Son valores aproximados con almacenamiento refrigerado o fresco;
# deben contrastarse con fuentes confiables (FAO, USDA, etc.) en el informe.
VIDA_UTIL_REFERENCIA = {
    "Manzana": (10, 15),
    "Banana": (3, 5),
    "Naranja": (7, 14),
    "Zanahoria": (14, 21),
    "Pepino": (5, 7),
    "Tomate": (5, 7),
    "Brócoli": (3, 5),
    "Espinaca": (3, 5),
    "Pimiento": (7, 10),
    "Papa": (14, 30),
    "Camote": (14, 21),
    "Berenjena": (5, 7),
    "Calabacín": (5, 7),
    "Quimbombó": (3, 5),
    "Arvejas": (3, 5),
    "Maíz": (3, 5),
    "Coliflor": (5, 7),
    "Repollo": (14, 21),
    "Lechuga": (5, 7),
    "Cebolla": (14, 30),
    "Mango": (5, 7),
    "Piña": (3, 5),
    "Granada": (14, 21),
    "Pera": (5, 7),
    "Kiwi": (7, 14),
    "Uvas": (5, 7),
    "Frutilla": (3, 5),
    "Sandía": (5, 7),
    "Limón": (14, 21),
    "Higo": (2, 3),
}

VIDA_UTIL_POR_DEFECTO = (3, 7)


def estimar_vida_util(
    tipo_alimento,
    dias_almacenado,
    vida_util_min,
    vida_util_max,
    estado_visual,
    almacenamiento
):
    """
    Estima los días restantes usando un rango de conservación
    de referencia y los días que el alimento lleva almacenado.
    """

    if dias_almacenado < 0:
        raise ValueError("Los días almacenados no pueden ser negativos.")

    if vida_util_min < 0 or vida_util_max < vida_util_min:
        raise ValueError("El rango de vida útil no es válido.")

    # Calcular los días restantes
    dias_minimos = max(0, vida_util_min - dias_almacenado)
    dias_maximos = max(0, vida_util_max - dias_almacenado)

    # Generar alerta según el resultado y el estado visual
    if estado_visual == "Podrido":
        dias_minimos = 0
        dias_maximos = 0
        alerta = "Descartar: se detectaron señales visuales de deterioro."
        nivel_alerta = "URGENTE"

    elif dias_maximos == 0:
        alerta = "Se alcanzó o superó el rango de conservación de referencia."
        nivel_alerta = "URGENTE"

    elif dias_minimos <= 2:
        alerta = "El alimento podría deteriorarse pronto. Se recomienda consumirlo."
        nivel_alerta = "PRECAUCIÓN"

    else:
        alerta = "El alimento se encuentra dentro del rango estimado."
        nivel_alerta = "NORMAL"

    return {
        "alimento": tipo_alimento,
        "estado_visual": estado_visual,
        "almacenamiento": almacenamiento,
        "dias_restantes_min": dias_minimos,
        "dias_restantes_max": dias_maximos,
        "nivel_alerta": nivel_alerta,
        "alerta": alerta
    }


def estimar_desde_imagen(tipo_alimento, estado_visual):
    """
    Estima la vida útil solo con lo que se obtiene de la foto:
    el tipo de alimento (red neuronal) y su estado (agente).
    No requiere que el usuario ingrese ninguna fecha.
    """

    minimo, maximo = VIDA_UTIL_REFERENCIA.get(
        tipo_alimento,
        VIDA_UTIL_POR_DEFECTO
    )

    return estimar_vida_util(
        tipo_alimento=tipo_alimento,
        dias_almacenado=0,
        vida_util_min=minimo,
        vida_util_max=maximo,
        estado_visual=estado_visual,
        almacenamiento="Refrigerado"
    )