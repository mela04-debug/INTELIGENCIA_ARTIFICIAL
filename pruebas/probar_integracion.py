
import sys
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.vision.reconocimiento import ReconocedorAlimentos
from src.vision.procesamiento import analizar_imagen
from src.agente.q_learning import AgenteQLearning


def ejecutar_prueba():

    print("\n====================================")
    print(" SISTEMA DE DETECCIÓN DE ALIMENTOS")
    print("====================================")

    ruta = PROYECTO / "datos" / "imagenes_prueba" / "manzana.jpg"

    # 1. Reconocer alimento
    reconocedor = ReconocedorAlimentos()
    identificacion = reconocedor.reconocer(ruta)

    if not identificacion["identificado"]:
        print("\nNo se pudo identificar el alimento.")
        return

    alimento = identificacion["alimento"]

    print("\nALIMENTO IDENTIFICADO")
    print("Alimento:", alimento)
    print("Confianza:", round(identificacion["confianza"] * 100, 2), "%")

    # 2. Analizar características visuales
    caracteristicas = analizar_imagen(str(ruta))

    # 3. Obtener diagnóstico del agente
    agente = AgenteQLearning()

    estado, diagnostico = agente.predecir(
        caracteristicas,
        alimento
    )

    print("\nDIAGNÓSTICO")
    print("Estado:", diagnostico)

    # 4. Solicitar retroalimentación
    print("\n¿El diagnóstico fue correcto?")
    print("1. Correcto")
    print("2. Corregir")

    opcion = input("Selecciona una opción: ").strip()

    if opcion == "1":

        resultado = agente.registrar_feedback(
            estado,
            diagnostico,
            correcto=True
        )

        print("\nDiagnóstico confirmado.")
        print("Recompensa:", resultado["recompensa"])

    elif opcion == "2":

        print("\nSelecciona el estado real:")
        print("1. Buen estado")
        print("2. Podrido")

        seleccion = input("Selecciona una opción: ").strip()

        if seleccion not in ("1", "2"):
            print("Opción no válida. No se guardó el aprendizaje.")
            return

        estado_real = (
            "Buen estado" if seleccion == "1"
            else "Podrido"
        )

        resultado = agente.registrar_feedback(
            estado,
            diagnostico,
            correcto=False,
            estado_real=estado_real
        )

        print("\nCorrección registrada.")
        print("Estado real:", estado_real)
        print("Recompensa:", resultado["recompensa"])

    else:
        print("\nOpción no válida. No se registró aprendizaje.")
        return

    print("\nVALORES DE APRENDIZAJE")
    for accion, valor in resultado["valores"].items():
        print(f"{accion}: {valor:.3f}")

    print("\nAprendizaje guardado en models/q_table.json")


if __name__ == "__main__":
    ejecutar_prueba()
