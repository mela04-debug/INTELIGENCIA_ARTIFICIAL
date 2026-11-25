
import math

from src.almacenamiento.memoria_agente import MemoriaAgente


class AprendizajeVisual:
    """
    Memoria visual supervisada.

    Guarda embeddings de imágenes confirmadas o corregidas
    y compara fotografías nuevas con los ejemplos almacenados.

    MobileNetV3 debe proporcionar las características visuales.
    Este módulo no modifica los pesos de MobileNetV3.
    """

    def __init__(
        self,
        memoria=None,
        similitud_minima=0.75,
        cantidad_vecinos=5
    ):
        self.memoria = memoria or MemoriaAgente()
        self.similitud_minima = similitud_minima
        self.cantidad_vecinos = cantidad_vecinos

    @staticmethod
    def normalizar_alimento(alimento):
        return str(alimento or "desconocido").strip().casefold()

    @staticmethod
    def normalizar_estado(estado):
        return str(estado or "").strip().casefold()

    @staticmethod
    def preparar_caracteristicas(caracteristicas):
        """
        Convierte las características en una lista de números.
        Admite listas, tuplas y arreglos de NumPy.
        """
        if hasattr(caracteristicas, "tolist"):
            caracteristicas = caracteristicas.tolist()

        if not isinstance(caracteristicas, (list, tuple)):
            raise ValueError(
                "Las características deben ser una lista o vector."
            )

        # Aplana vectores anidados sencillos.
        resultado = []

        def recorrer(elemento):
            if isinstance(elemento, (list, tuple)):
                for valor in elemento:
                    recorrer(valor)
            else:
                numero = float(elemento)

                if not math.isfinite(numero):
                    raise ValueError(
                        "El vector contiene valores no válidos."
                    )

                resultado.append(numero)

        recorrer(caracteristicas)

        if not resultado:
            raise ValueError("El vector de características está vacío.")

        return resultado

    @staticmethod
    def similitud_coseno(vector_a, vector_b):
        """
        Calcula la similitud coseno entre dos embeddings.
        Un valor cercano a 1 indica mayor similitud.
        """
        if len(vector_a) != len(vector_b):
            return None

        producto = sum(
            a * b for a, b in zip(vector_a, vector_b)
        )

        norma_a = math.sqrt(sum(a * a for a in vector_a))
        norma_b = math.sqrt(sum(b * b for b in vector_b))

        if norma_a == 0 or norma_b == 0:
            return None

        return producto / (norma_a * norma_b)

    def registrar_ejemplo(
        self,
        alimento,
        estado,
        caracteristicas,
        imagen=None
    ):
        """
        Guarda un ejemplo visual confirmado por el usuario.

        estado debe indicar el estado REAL:
        'Buen estado' o 'Mal estado'.
        """
        alimento = self.normalizar_alimento(alimento)
        estado = str(estado or "").strip()

        if not estado:
            raise ValueError("Debes indicar el estado correcto.")

        vector = self.preparar_caracteristicas(caracteristicas)

        self.memoria.guardar_memoria_visual(
            alimento=alimento,
            estado=estado,
            caracteristicas=vector,
            imagen=imagen
        )

        return {
            "guardado": True,
            "alimento": alimento,
            "estado": estado,
            "dimensiones": len(vector),
            "imagen": str(imagen) if imagen else None
        }

    def buscar_similares(
        self,
        alimento,
        caracteristicas,
        cantidad=None
    ):
        """
        Busca los ejemplos más similares de un alimento.
        """
        alimento = self.normalizar_alimento(alimento)
        vector_nuevo = self.preparar_caracteristicas(caracteristicas)

        ejemplos = self.memoria.obtener_memorias_visuales(
            alimento=alimento
        )

        resultados = []

        for ejemplo in ejemplos:
            try:
                vector_guardado = self.preparar_caracteristicas(
                    ejemplo["caracteristicas"]
                )

                similitud = self.similitud_coseno(
                    vector_nuevo,
                    vector_guardado
                )

                if similitud is None:
                    continue

                resultados.append({
                    "alimento": ejemplo["alimento"],
                    "estado": ejemplo["estado"],
                    "similitud": float(similitud),
                    "imagen": ejemplo.get("imagen")
                })

            except (ValueError, TypeError, KeyError):
                continue

        resultados.sort(
            key=lambda elemento: elemento["similitud"],
            reverse=True
        )

        limite = cantidad or self.cantidad_vecinos

        return resultados[:limite]

    def predecir(
        self,
        alimento,
        caracteristicas
    ):
        """
        Estima el estado usando los ejemplos visuales conocidos.

        Si no encuentra similitud suficiente, devuelve
        'Sin coincidencia' en lugar de inventar un diagnóstico.
        """
        similares = self.buscar_similares(
            alimento,
            caracteristicas
        )

        if not similares:
            return {
                "alimento": self.normalizar_alimento(alimento),
                "estado": "Sin coincidencia",
                "confianza": 0.0,
                "ejemplos_consultados": 0,
                "coincidencias": []
            }

        mejores = [
            ejemplo for ejemplo in similares
            if ejemplo["similitud"] >= self.similitud_minima
        ]

        if not mejores:
            return {
                "alimento": self.normalizar_alimento(alimento),
                "estado": "Sin coincidencia",
                "confianza": 0.0,
                "ejemplos_consultados": len(similares),
                "coincidencias": similares
            }

        votos = {}

        for ejemplo in mejores:
            estado = ejemplo["estado"]

            # Los ejemplos más similares tienen más peso.
            peso = max(0.0, ejemplo["similitud"])

            votos[estado] = votos.get(estado, 0.0) + peso

        estado_ganador = max(
            votos,
            key=votos.get
        )

        total = sum(votos.values())

        confianza = (
            votos[estado_ganador] / total
            if total > 0
            else 0.0
        )

        return {
            "alimento": self.normalizar_alimento(alimento),
            "estado": estado_ganador,
            "confianza": round(confianza, 4),
            "similitud_maxima": round(
                mejores[0]["similitud"], 4
            ),
            "ejemplos_consultados": len(similares),
            "coincidencias": similares
        }

    def obtener_ejemplos(self, alimento=None):
        """
        Consulta los ejemplos guardados en SQLite.
        """
        return self.memoria.obtener_memorias_visuales(
            alimento=alimento
        )
