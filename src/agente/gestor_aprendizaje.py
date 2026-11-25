
from pathlib import Path

from src.vision.sistema_reconocimiento import SistemaReconocimiento
from src.vision.procesamiento import analizar_imagen as analizar_procesamiento
from src.agente.q_learning import AgenteQLearning
from src.almacenamiento.memoria_agente import MemoriaAgente


class GestorAprendizaje:
    """
    Coordina los módulos de reconocimiento, procesamiento,
    aprendizaje visual, Q-Learning y memoria persistente.
    """

    def __init__(
        self,
        sistema_reconocimiento=None,
        agente=None,
        memoria=None
    ):
        self.memoria = memoria or MemoriaAgente()

        self.sistema_reconocimiento = (
            sistema_reconocimiento or SistemaReconocimiento()
        )

        self.agente = agente or AgenteQLearning(
            memoria=self.memoria
        )

    def analizar_imagen(self, ruta_imagen):
        """
        Analiza una imagen y combina los resultados del
        reconocimiento visual y Q-Learning.
        """

        ruta = Path(ruta_imagen)

        if not ruta.is_file():
            return {
                "exito": False,
                "mensaje": "No se encontró la imagen.",
                "ruta_imagen": str(ruta)
            }

        try:
            # 1. Reconocimiento del alimento y memoria visual
            resultado_visual = self.sistema_reconocimiento.analizar(
                str(ruta)
            )

            if not resultado_visual.get("identificado", False):
                return {
                    "exito": False,
                    "mensaje": "No se pudo identificar el alimento.",
                    "reconocimiento": resultado_visual
                }

            alimento = resultado_visual["alimento"]

            # 2. Procesamiento tradicional de la imagen
            estado_procesamiento = analizar_procesamiento(str(ruta))

            if isinstance(estado_procesamiento, dict):
                estado_procesamiento = "desconocido"

            # 3. Predicción mediante Q-Learning
            prediccion_q = self.agente.predecir(
                estado_procesamiento,
                alimento
            )

            # 4. Resultado unificado
            return {
                "exito": True,
                "alimento": alimento,
                "estado_visual": estado_procesamiento,
                "estado_memoria_visual": resultado_visual.get(
                    "estado",
                    "Sin diagnóstico"
                ),
                "prediccion_q": prediccion_q,
                "confianza_reconocimiento": resultado_visual.get(
                    "confianza_reconocimiento"
                ),
                "confianza_visual": resultado_visual.get(
                    "confianza_visual"
                ),
                "similitud_maxima": resultado_visual.get(
                    "similitud_maxima"
                ),
                "ejemplos_consultados": resultado_visual.get(
                    "ejemplos_consultados",
                    0
                ),
                "ruta_imagen": str(ruta),
                "mensaje": "Análisis completado."
            }

        except Exception as error:
            return {
                "exito": False,
                "mensaje": "Ocurrió un error durante el análisis.",
                "error": str(error)
            }

    def registrar_feedback(
        self,
        ruta_imagen,
        estado_real,
        alimento_real=None
    ):
        """
        Registra la evaluación humana de una imagen y
        transmite la retroalimentación a Q-Learning.
        """

        ruta = Path(ruta_imagen)

        if not ruta.is_file():
            return {
                "exito": False,
                "mensaje": "No se encontró la imagen."
            }

        if estado_real not in ("Buen estado", "Mal estado"):
            return {
                "exito": False,
                "mensaje": "El estado real debe ser 'Buen estado' o 'Mal estado'."
            }

        try:
            # Analizar nuevamente para obtener la predicción
            resultado = self.analizar_imagen(str(ruta))

            if not resultado.get("exito"):
                return resultado

            alimento = alimento_real or resultado["alimento"]
            prediccion = resultado["prediccion_q"]
            estado_visual = resultado["estado_visual"]

            if prediccion not in ("Buen estado", "Mal estado"):
                return {
                    "exito": False,
                    "mensaje": "La predicción de Q-Learning no es válida."
                }

            correcto = prediccion == estado_real

            # Extraer el embedding de MobileNetV3
            caracteristicas = (
                self.sistema_reconocimiento.reconocedor
                .extraer_caracteristicas(str(ruta))
            )

            # Registrar la corrección en Q-Learning
            resultado_feedback = self.agente.registrar_feedback(
                estado_visual=estado_visual,
                accion_tomada=prediccion,
                correcto=correcto,
                estado_real=estado_real,
                alimento_ia=resultado["alimento"],
                alimento_real=alimento_real,
                alimento_contexto=resultado["alimento"],
                caracteristicas=caracteristicas,
                imagen=str(ruta)
            )

            return {
                "exito": True,
                "alimento": alimento,
                "prediccion_anterior": prediccion,
                "estado_confirmado": estado_real,
                "correcto": correcto,
                "feedback": resultado_feedback,
                "mensaje": "Retroalimentación registrada."
            }

        except Exception as error:
            return {
                "exito": False,
                "mensaje": "No se pudo registrar la retroalimentación.",
                "error": str(error)
            }
