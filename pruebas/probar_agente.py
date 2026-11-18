
import sys
from pathlib import Path

# Agregar la carpeta principal del proyecto
PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.vision.procesamiento import analizar_imagen
from src.agente.q_learning import AgenteQLearning

# Ubicación de la imagen
ruta_imagen = PROYECTO / "datos" / "imagenes_prueba" / "manzana.jpg"

# Analizar fotografía
resultado = analizar_imagen(str(ruta_imagen))

# Crear agente
agente = AgenteQLearning()

# Obtener diagnóstico
estado, accion = agente.predecir(resultado)

print("DIAGNÓSTICO DEL AGENTE")
print("--------------------------")
print("Estado:", estado)
print("Diagnóstico:", accion)

# Simular que el usuario acepta
recompensa = 1

nuevo_valor = agente.registrar_feedback(
    estado,
    accion,
    recompensa
)

print("\nRETROALIMENTACIÓN")
print("--------------------------")
print("Recompensa:", recompensa)
print("Nuevo valor Q:", nuevo_valor)
print("Aprendizaje guardado correctamente.")
