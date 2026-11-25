
from pathlib import Path

from src.vision.reconocimiento import ReconocedorAlimentos
from src.agente.aprendizaje_visual import AprendizajeVisual


class SistemaReconocimiento:

    def __init__(self):
        self.reconocedor = ReconocedorAlimentos()
        self.aprendizaje = AprendizajeVisual()

    def analizar(self, ruta_imagen):
        """
        Reconoce el alimento y consulta la memoria visual
        para estimar su estado.
        """

        ruta = Path(ruta_imagen)

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró la imagen: {ruta}"
            )

        # 1. Identificar el alimento.
        resultado_reconocimiento = self.reconocedor.reconocer(
            ruta
        )

        alimento = resultado_reconocimiento["alimento"]

        if not resultado_reconocimiento["identificado"]:
            return {
                "identificado": False,
                "alimento": alimento,
                "estado": "Sin diagnóstico",
                "confianza_reconocimiento": resultado_reconocimiento["confianza"],
                "confianza_visual": 0.0,
                "mensaje": "No se pudo identificar el alimento."
            }

        # 2. Extraer las características reales de MobileNetV3.
        caracteristicas = self.reconocedor.extraer_caracteristicas(
            ruta
        )

        # 3. Consultar los ejemplos visuales guardados.
        resultado_visual = self.aprendizaje.predecir(
            alimento,
            caracteristicas
        )

        estado = resultado_visual["estado"]

        if estado == "Sin coincidencia":
            mensaje = (
                "No existen ejemplos visuales suficientemente "
                "similares. Se necesita una confirmación humana."
            )
        else:
            mensaje = (
                "Diagnóstico estimado mediante la memoria visual."
            )

        return {
            "identificado": True,
            "alimento": alimento,
            "estado": estado,
            "confianza_reconocimiento": resultado_reconocimiento["confianza"],
            "confianza_visual": resultado_visual["confianza"],
            "similitud_maxima": resultado_visual["similitud_maxima"],
            "ejemplos_consultados": resultado_visual["ejemplos_consultados"],
            "coincidencias": resultado_visual["coincidencias"],
            "caracteristicas_dimensiones": len(caracteristicas),
            "mensaje": mensaje
        }
