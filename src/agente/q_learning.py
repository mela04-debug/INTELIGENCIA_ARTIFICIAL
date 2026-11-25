
import json

from src.almacenamiento.memoria_agente import MemoriaAgente


class AgenteQLearning:
    """
    Agente Q-Learning con aprendizaje independiente
    por alimento y patrón visual.

    Los valores Q se guardan en SQLite.
    """

    def __init__(self, memoria=None, tasa_aprendizaje=0.2):
        self.memoria = memoria or MemoriaAgente()
        self.tasa_aprendizaje = float(tasa_aprendizaje)

        if not 0 < self.tasa_aprendizaje <= 1:
            raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1.")

        self.acciones = [
            "Buen estado",
            "Mal estado"
        ]

    @staticmethod
    def normalizar_alimento(nombre):
        return str(nombre or "desconocido").strip().casefold()

    @staticmethod
    def normalizar_estado_visual(estado_visual):
        if isinstance(estado_visual, dict):
            return json.dumps(
                estado_visual,
                sort_keys=True,
                ensure_ascii=False,
                default=str
            )

        if isinstance(estado_visual, (list, tuple)):
            return json.dumps(
                estado_visual,
                ensure_ascii=False,
                default=str
            )

        return str(estado_visual or "desconocido")

    def obtener_acciones(self, estado_real=None):
        acciones = list(self.acciones)

        if estado_real:
            estado_real = str(estado_real).strip()

            if estado_real and estado_real not in acciones:
                acciones.append(estado_real)

        return acciones

    def obtener_valores(self, estado_visual, alimento):
        estado = self.normalizar_estado_visual(estado_visual)
        alimento = self.normalizar_alimento(alimento)

        valores = {}

        for accion in self.acciones:
            valores[accion] = self.memoria.obtener_valor_q(
                alimento,
                estado,
                accion
            )

        return valores

    def predecir(self, estado_visual, alimento_detectado=None):
        sin_alimento = alimento_detectado is None

        if sin_alimento:
            alimento_detectado = "desconocido"

        alimento_final = str(alimento_detectado).strip()
        estado = self.normalizar_estado_visual(estado_visual)
        alimento_clave = self.normalizar_alimento(alimento_final)

        valores = self.obtener_valores(
            estado,
            alimento_clave
        )

        mejor_accion = max(
            self.acciones,
            key=lambda accion: valores.get(accion, 0.0)
        )

        if sin_alimento:
            return estado_visual, mejor_accion

        return estado_visual, mejor_accion, alimento_final

    def registrar_feedback(
        self,
        estado_visual,
        accion_tomada,
        correcto,
        estado_real=None,
        alimento_ia=None,
        alimento_real=None,
        alimento_contexto=None,
        caracteristicas=None,
        imagen=None
    ):
        """
        Registra la confirmación o corrección humana.

        Actualiza Q-Learning, experiencias, estadísticas
        y, cuando corresponde, memoria visual.
        """

        alimento = (
            alimento_real
            or alimento_contexto
            or alimento_ia
            or "desconocido"
        )

        alimento_clave = self.normalizar_alimento(alimento)
        estado = self.normalizar_estado_visual(estado_visual)

        accion_tomada = str(accion_tomada or "").strip()
        

        if not accion_tomada:
            raise ValueError("La acción tomada no puede estar vacía.")

        correcto = bool(correcto)

        if estado_real is not None:
            estado_real = str(estado_real).strip()

            if not estado_real:
                estado_real = None

        if correcto and estado_real and estado_real != accion_tomada:
            raise ValueError(
                "El feedback indica que la predicción fue correcta, "
                "pero el estado real es diferente."
            )

        if not correcto and not estado_real:
            raise ValueError(
                "Para corregir una predicción debes proporcionar "
                "el estado real confirmado."
            )

        if caracteristicas is not None and not estado_real:
            raise ValueError(
                "No se pueden guardar características visuales "
                "sin un estado real confirmado."
            )

        estado_final = estado_real if estado_real else accion_tomada

        # Obtener el valor Q de la acción predicha.
        valor_actual = self.memoria.obtener_valor_q(
            alimento_clave,
            estado,
            accion_tomada
        )

        recompensa = 10.0 if correcto else -10.0

        nuevo_valor = valor_actual + self.tasa_aprendizaje * (
            recompensa - valor_actual
        )

        self.memoria.guardar_valor_q(
            alimento_clave,
            estado,
            accion_tomada,
            nuevo_valor
        )

        # Reforzar la etiqueta correcta si hubo corrección.
        if not correcto:
            valor_real = self.memoria.obtener_valor_q(
                alimento_clave,
                estado,
                estado_final
            )

            nuevo_valor_real = valor_real + self.tasa_aprendizaje * (
                10.0 - valor_real
            )

            self.memoria.guardar_valor_q(
                alimento_clave,
                estado,
                estado_final,
                nuevo_valor_real
            )

        # Guardar experiencia.
        self.memoria.guardar_experiencia(
            alimento=alimento_clave,
            estado_visual=estado,
            estado_predicho=accion_tomada,
            estado_real=estado_final,
            correcto=correcto,
            imagen=imagen
        )

        # Actualizar estadísticas.
        self.memoria.actualizar_estadisticas(
            alimento_clave,
            correcto
        )

        # Guardar memoria visual confirmada.
        memoria_visual_guardada = False

        if caracteristicas is not None:
            self.memoria.guardar_memoria_visual(
                alimento=alimento_clave,
                estado=estado_final,
                caracteristicas=caracteristicas,
                imagen=imagen
            )

            memoria_visual_guardada = True
    
        return {
            "alimento": alimento_clave,
            "estado_visual": estado,
            "accion": accion_tomada,
            "estado_real": estado_final,
            "correcto": correcto,
            "recompensa": recompensa,
            "nuevo_valor_q": nuevo_valor,
            "memoria_visual_guardada": memoria_visual_guardada
        }

    def obtener_q_table(self, alimento):
        return self.memoria.obtener_q_table(
            self.normalizar_alimento(alimento)
        )

    def obtener_resumen(self):
        return self.memoria.obtener_resumen()
