import json
import os
from pathlib import Path


class AgenteQLearning:
    """Agente Q-Learning con valores independientes por alimento y patrón visual."""

    def __init__(self, archivo_qtable="q_table.json"):
        ruta_base = Path(__file__).resolve().parents[2]
        self.archivo_qtable = os.path.join(ruta_base, archivo_qtable)
        self.tasa_aprendizaje = 0.2
        self.acciones = ["Buen estado", "Mal estado"]
        self.q_table = self.cargar_qtable()

    @staticmethod
    def normalizar_alimento(nombre):
        return str(nombre or "desconocido").strip().casefold()

    def _clave_estado(self, estado_visual, alimento):
        return f"{self.normalizar_alimento(alimento)}::{estado_visual}"

    def cargar_qtable(self):
        if os.path.exists(self.archivo_qtable):
            try:
                with open(self.archivo_qtable, "r", encoding="utf-8") as archivo:
                    datos = json.load(archivo)
                if not isinstance(datos, dict):
                    raise ValueError("Formato de Q-table inválido")
                datos.setdefault("_nombres_aprendidos", {})
                return datos
            except (OSError, json.JSONDecodeError, ValueError):
                return {"_nombres_aprendidos": {}}
        return {"_nombres_aprendidos": {}}

    def guardar_qtable(self):
        os.makedirs(os.path.dirname(self.archivo_qtable), exist_ok=True)
        temporal = self.archivo_qtable + ".tmp"
        with open(temporal, "w", encoding="utf-8") as archivo:
            json.dump(self.q_table, archivo, indent=4, ensure_ascii=False)
        os.replace(temporal, self.archivo_qtable)

    def predecir(self, estado_visual, alimento_detectado):
        clave_nombre = f"{self.normalizar_alimento(alimento_detectado)}|{estado_visual}"
        alimento_final = self.q_table.get("_nombres_aprendidos", {}).get(
            clave_nombre, alimento_detectado
        )

        clave_estado = self._clave_estado(estado_visual, alimento_detectado)
        if clave_estado not in self.q_table:
            self.q_table[clave_estado] = {accion: 0.0 for accion in self.acciones}
            self.guardar_qtable()

        valores = self.q_table[clave_estado]
        mejor_accion = max(self.acciones, key=lambda accion: valores.get(accion, 0.0))
        return estado_visual, mejor_accion, alimento_final

    def registrar_feedback(
        self, estado_visual, accion_tomada, correcto, estado_real=None,
        alimento_ia=None, alimento_real=None, alimento_contexto=None
    ):
        alimento = alimento_contexto or alimento_ia or alimento_real or "desconocido"
        clave_estado = self._clave_estado(estado_visual, alimento)

        if clave_estado not in self.q_table:
            self.q_table[clave_estado] = {accion: 0.0 for accion in self.acciones}

        valores = self.q_table[clave_estado]
        if correcto:
            recompensa = 10.0
            valores[accion_tomada] = valores.get(accion_tomada, 0.0) + self.tasa_aprendizaje * (
                recompensa - valores.get(accion_tomada, 0.0)
            )
        else:
            castigo = -10.0
            valores[accion_tomada] = valores.get(accion_tomada, 0.0) + self.tasa_aprendizaje * (
                castigo - valores.get(accion_tomada, 0.0)
            )

            if estado_real in self.acciones:
                valor_real = valores.get(estado_real, 0.0)
                valores[estado_real] = valor_real + self.tasa_aprendizaje * (10.0 - valor_real)

            if alimento_ia and alimento_real:
                if self.normalizar_alimento(alimento_ia) != self.normalizar_alimento(alimento_real):
                    clave_nombre = f"{self.normalizar_alimento(alimento_ia)}|{estado_visual}"
                    self.q_table["_nombres_aprendidos"][clave_nombre] = alimento_real.strip().capitalize()

        self.guardar_qtable()
