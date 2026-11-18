
import sys
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.estimacion.vida_util import estimar_vida_util

resultado = estimar_vida_util(
    tipo_alimento="Manzana",
    dias_almacenado=8,
    vida_util_min=10,
    vida_util_max=15,
    estado_visual="Buen estado",
    almacenamiento="Refrigerado"
)

print("ESTIMACIÓN DE VIDA ÚTIL")
print("------------------------------")
print("Alimento:", resultado["alimento"])
print("Estado:", resultado["estado_visual"])
print("Almacenamiento:", resultado["almacenamiento"])

print(
    "Días restantes:",
    resultado["dias_restantes_min"],
    "a",
    resultado["dias_restantes_max"]
)

print("Nivel de alerta:", resultado["nivel_alerta"])
print("Mensaje:", resultado["alerta"])
