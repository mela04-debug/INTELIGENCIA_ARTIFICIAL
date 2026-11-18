
import sys
from pathlib import Path

# Identificar la carpeta principal del proyecto
PROYECTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROYECTO))

# Importar los tres módulos
from src.vision.procesamiento import analizar_imagen
from src.agente.q_learning import AgenteQLearning
from src.estimacion.vida_util import estimar_vida_util


def ejecutar_analisis():

    print("\n======================================")
    print(" DETECCIÓN DE ALIMENTOS Y VENCIMIENTO")
    print("======================================")

    # 1. Seleccionar imagen
    ruta_imagen = PROYECTO / "datos" / "imagenes_prueba" / "manzana.jpg"

    try:
        caracteristicas = analizar_imagen(str(ruta_imagen))
    except (ValueError, OSError) as error:
        print("\nError al analizar la imagen:", error)
        return

    print("\n1. ANÁLISIS VISUAL")
    print("------------------------------")
    print("Intensidad media:", round(caracteristicas["intensidad_media"], 2))
    print("Variación:", round(caracteristicas["variacion"], 2))
    print(
        "Proporción de píxeles oscuros:",
        round(caracteristicas["proporcion_oscuros"], 2)
    )

    # 2. Diagnóstico del agente
    agente = AgenteQLearning()
    estado, diagnostico = agente.predecir(caracteristicas)

    print("\n2. DIAGNÓSTICO DEL AGENTE")
    print("------------------------------")
    print("Estado analizado:", estado)
    print("Diagnóstico:", diagnostico)

    # 3. Estimación de vida útil
    resultado = estimar_vida_util(
        tipo_alimento="Manzana",
        dias_almacenado=8,
        vida_util_min=10,
        vida_util_max=15,
        estado_visual=diagnostico,
        almacenamiento="Refrigerado"
    )

    print("\n3. ESTIMACIÓN DE VIDA ÚTIL")
    print("------------------------------")
    print("Alimento:", resultado["alimento"])
    print("Estado visual:", resultado["estado_visual"])
    print("Almacenamiento:", resultado["almacenamiento"])
    print(
        "Días restantes:",
        resultado["dias_restantes_min"],
        "a",
        resultado["dias_restantes_max"]
    )
    print("Nivel de alerta:", resultado["nivel_alerta"])
    print("Mensaje:", resultado["alerta"])

    print("\n======================================")
    print(" ANÁLISIS FINALIZADO")
    print("======================================")


if __name__ == "__main__":
    ejecutar_analisis()
