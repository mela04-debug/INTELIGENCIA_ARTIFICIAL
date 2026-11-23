import os
from src.vision.procesamiento import extraer_caracteristicas, discretizar_estado
from src.agente.q_learning import AgenteQLearning

def entrenar_agente_con_imagen(ruta_imagen, estado_real):
    agente = AgenteQLearning()
    print(f"\nProcesando imagen: {ruta_imagen}")
    
    try:
        porcentaje, s, v = extraer_caracteristicas(ruta_imagen)
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return
    
    estado_actual = discretizar_estado(porcentaje, s, v)
    print(f"Características detectadas -> Estado: {estado_actual}")
    
    accion_predicha = agente.elegir_accion(estado_actual)
    prediccion_texto = "Mal estado" if accion_predicha == 1 else "Buen estado"
    print(f"El agente predice: {prediccion_texto}")
    
    if accion_predicha == estado_real:
        print("¡El agente acertó! Reforzando comportamiento positivo...")
        recompensa = 10
    else:
        print("El agente falló. Aplicando penalización...")
        recompensa = -10
        
    agente.aprender(estado_actual, accion_predicha, recompensa, "FIN")
    print("Conocimiento actualizado y guardado en models/q_table.json")

if __name__ == "__main__":
    print("Iniciando entrenamiento del Agente de Detección de Frutas...")
    
    ruta_manzana = "datos/imagenes_prueba/manzana.jpg"
    ruta_platano = "datos/imagenes_prueba/platano.jpg"
    
    # Entrenar con la manzana (estado 0 = Buena)
    if os.path.exists(ruta_manzana):
        entrenar_agente_con_imagen(ruta_manzana, estado_real=0)
    else:
        print(f"Advertencia: No se encontró {ruta_manzana}")
        
    # Entrenar con el plátano (estado 1 = Mala)
    if os.path.exists(ruta_platano):
        entrenar_agente_con_imagen(ruta_platano, estado_real=1)
    else:
        print(f"Advertencia: No se encontró {ruta_platano}")
