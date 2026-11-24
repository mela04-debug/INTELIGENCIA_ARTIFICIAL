import json
import os
from pathlib import Path


class EstadisticasEntrenamiento:
    """Persistencia de métricas de retroalimentación, separadas por alimento."""

    def __init__(self, archivo=None):
        raiz = Path(__file__).resolve().parents[2]
        self.archivo = Path(archivo) if archivo else raiz / "datos" / "estadisticas_entrenamiento.json"
        self.datos = self.cargar()

    def cargar(self):
        if not self.archivo.exists():
            return {}
        try:
            with self.archivo.open("r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
            return datos if isinstance(datos, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _clave(alimento):
        return str(alimento or "Desconocido").strip().casefold()

    def registrar(self, alimento, correcto):
        nombre = str(alimento or "Desconocido").strip().capitalize()
        clave = self._clave(nombre)
        registro = self.datos.setdefault(clave, {
            "alimento": nombre,
            "intentos": 0,
            "correctos": 0,
            "fallas": 0,
            "corregidos": 0
        })

        registro["intentos"] += 1
        if correcto:
            registro["correctos"] += 1
        else:
            registro["fallas"] += 1
            registro["corregidos"] += 1

        self.guardar()
        return registro

    def guardar(self):
        self.archivo.parent.mkdir(parents=True, exist_ok=True)
        temporal = str(self.archivo) + ".tmp"
        with open(temporal, "w", encoding="utf-8") as archivo:
            json.dump(self.datos, archivo, indent=4, ensure_ascii=False)
        os.replace(temporal, self.archivo)

    def obtener_registros(self):
        return sorted(
            self.datos.values(),
            key=lambda registro: registro.get("alimento", "").casefold()
        )
