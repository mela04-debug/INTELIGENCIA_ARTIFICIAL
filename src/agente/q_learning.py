import json
import os
from pathlib import Path

class AgenteQLearning:
    def __init__(self, archivo_qtable="q_table.json"):
        ruta_base = Path(__file__).resolve().parents[2]
        self.archivo_qtable = os.path.join(ruta_base, archivo_qtable)
        
        self.tasa_aprendizaje = 0.2
        self.acciones = ["Buen estado", "Mal estado"]
        
        self.q_table = self.cargar_qtable()

    def cargar_qtable(self):
        if os.path.exists(self.archivo_qtable):
            try:
                with open(self.archivo_qtable, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                    # Aseguramos que exista la sección de nombres aprendidos
                    if "_nombres_aprendidos" not in datos:
                        datos["_nombres_aprendidos"] = {}
                    return datos
            except:
                return {"_nombres_aprendidos": {}}
        return {"_nombres_aprendidos": {}}

    def guardar_qtable(self):
        with open(self.archivo_qtable, 'w', encoding='utf-8') as f:
            json.dump(self.q_table, f, indent=4, ensure_ascii=False)

    def predecir(self, estado_visual, alimento_detectado):
        """
        1. Revisa si el agente ya aprendió a corregir este alimento bajo este patrón visual.
        2. Decide si está en Buen o Mal estado según los puntos acumulados.
        """
        # --- 1. CORRECCIÓN DE NOMBRE APRENDIDA ---
        clave_nombre = f"{alimento_detectado.lower()}|{estado_visual}"
        alimento_final = self.q_table.get("_nombres_aprendidos", {}).get(clave_nombre, alimento_detectado)

        # --- 2. DIAGNÓSTICO SANITARIO (Q-LEARNING) ---
        if estado_visual not in self.q_table:
            self.q_table[estado_visual] = {accion: 0.0 for accion in self.acciones}
            self.guardar_qtable()

        valores = self.q_table[estado_visual]
        mejor_accion = max(valores, key=valores.get)
        
        return estado_visual, mejor_accion, alimento_final

    def registrar_feedback(self, estado_visual, accion_tomada, correcto, estado_real=None, alimento_ia=None, alimento_real=None):
        """
        Guarda premios/castigos para el estado sanitario y aprende nuevos nombres si fueron corregidos.
        """
        if estado_visual not in self.q_table:
            self.q_table[estado_visual] = {accion: 0.0 for accion in self.acciones}

        if correcto:
            recompensa = 10.0
            self.q_table[estado_visual][accion_tomada] += self.tasa_aprendizaje * (recompensa - self.q_table[estado_visual][accion_tomada])
        else:
            castigo = -10.0
            recompensa_correcta = 10.0
            
            # 1. Castigar estado sanitario equivocado y premiar el real
            self.q_table[estado_visual][accion_tomada] += self.tasa_aprendizaje * (castigo - self.q_table[estado_visual][accion_tomada])
            if estado_real in self.q_table[estado_visual]:
                self.q_table[estado_visual][estado_real] += self.tasa_aprendizaje * (recompensa_correcta - self.q_table[estado_visual][estado_real])

            # 2. Si el usuario escribió o eligió un nombre diferente, el agente lo memoriza
            if alimento_ia and alimento_real and alimento_ia.lower().strip() != alimento_real.lower().strip():
                clave_nombre = f"{alimento_ia.lower().strip()}|{estado_visual}"
                self.q_table["_nombres_aprendidos"][clave_nombre] = alimento_real.strip().capitalize()

        self.guardar_qtable()