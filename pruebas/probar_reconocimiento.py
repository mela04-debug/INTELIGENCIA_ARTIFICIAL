
import sys
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.vision.reconocimiento import ReconocedorAlimentos

ruta_imagen = (
    PROYECTO / "datos" / "imagenes_prueba" / "platano.jpg"
)

print("CARGANDO MODELO DE RECONOCIMIENTO...")

reconocedor = ReconocedorAlimentos()

print("\nANALIZANDO IMAGEN...")

resultado = reconocedor.reconocer(ruta_imagen)

print("\nRESULTADO DEL RECONOCIMIENTO")
print("-----------------------------")
print("Identificado:", resultado["identificado"])
print("Alimento:", resultado["alimento"])
print("Categoría original:", resultado["categoria"])
print(
    "Confianza:",
    round(resultado["confianza"] * 100, 2),
    "%"
)
