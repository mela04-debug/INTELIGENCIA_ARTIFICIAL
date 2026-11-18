
import json
import random
from pathlib import Path


class AgenteQLearning:

    ACCIONES = ["Buen estado", "Podrido"]

    def __init__(self, alpha=0.1, epsilon=0.05):
        self.alpha = alpha
        self.epsilon = epsilon

        proyecto = Path(__file__).resolve().parents[2]
        self.ruta_tabla = proyecto / "models" / "q_table.json"

        self.q_table = {}
        self.cargar_tabla()

    def cargar_tabla(self):
        if self.ruta_tabla.exists():
            with open(self.ruta_tabla, "r", encoding="utf-8") as archivo:
                self.q_table = json.load(archivo)

    def guardar_tabla(self):
        self.ruta_tabla.parent.mkdir(parents=True, exist_ok=True)

        with open(self.ruta_tabla, "w", encoding="utf-8") as archivo:
            json.dump(
                self.q_table,
                archivo,
                indent=4,
                ensure_ascii=False
            )

    def obtener_estado(self, caracteristicas, tipo_alimento="Desconocido"):

        intensidad = caracteristicas["intensidad_media"]
        variacion = caracteristicas["variacion"]
        oscuros = caracteristicas["proporcion_oscuros"]

        nivel_intensidad = (
            "baja" if intensidad < 70
            else "media" if intensidad < 150
            else "alta"
        )

        nivel_variacion = (
            "baja" if variacion < 30
            else "media" if variacion < 70
            else "alta"
        )

        nivel_oscuros = (
            "baja" if oscuros < 0.20
            else "media" if oscuros < 0.50
            else "alta"
        )

        alimento = tipo_alimento.lower().replace(" ", "_")

        return (
            f"{alimento}_{nivel_intensidad}_"
            f"{nivel_variacion}_{nivel_oscuros}"
        )

    def predecir(self, caracteristicas, tipo_alimento="Desconocido"):

        estado = self.obtener_estado(
            caracteristicas,
            tipo_alimento
        )

        if estado not in self.q_table:
            self.q_table[estado] = {
                accion: 0.0 for accion in self.ACCIONES
            }

        valores = self.q_table[estado]

        if random.random() < self.epsilon:
            accion = random.choice(self.ACCIONES)
        else:
            mayor_valor = max(valores.values())

            mejores = [
                accion
                for accion, valor in valores.items()
                if valor == mayor_valor
            ]

            accion = random.choice(mejores)

        return estado, accion

    def actualizar_valor(self, estado, accion, recompensa):

        valor_actual = self.q_table[estado][accion]

        nuevo_valor = valor_actual + self.alpha * (
            recompensa - valor_actual
        )

        self.q_table[estado][accion] = nuevo_valor

        return nuevo_valor

    def registrar_feedback(
        self,
        estado,
        accion_predicha,
        correcto,
        estado_real=None
    ):

        if estado not in self.q_table:
            self.q_table[estado] = {
                accion: 0.0 for accion in self.ACCIONES
            }

        if accion_predicha not in self.ACCIONES:
            raise ValueError("Diagnóstico predicho no válido.")

        if correcto:
            self.actualizar_valor(
                estado,
                accion_predicha,
                recompensa=1
            )

            recompensa = 1

        else:
            if estado_real not in self.ACCIONES:
                raise ValueError(
                    "Debes indicar el estado real del alimento."
                )

            self.actualizar_valor(
                estado,
                accion_predicha,
                recompensa=-1
            )

            self.actualizar_valor(
                estado,
                estado_real,
                recompensa=1
            )

            recompensa = -1

        self.guardar_tabla()

        return {
            "recompensa": recompensa,
            "valores": self.q_table[estado].copy()
        }
