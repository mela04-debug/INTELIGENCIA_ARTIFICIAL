"""Orquesta feedback confirmado, evaluación y progreso por alimento."""
from src.almacenamiento.memoria_agente import MemoriaAgente
from src.agente.aprendizaje_visual import AprendizajeVisual
from src.agente.q_learning import AgenteQLearning
from .evaluacion import Evaluacion
from .niveles_aprendizaje import NivelesAprendizaje
from pathlib import Path

class GestorEntrenamiento:
    ESTADOS = ("Buen estado", "Mal estado")
    def __init__(self, memoria=None, aprendizaje_visual=None, agente_q=None):
        self.memoria = memoria or MemoriaAgente()
        self.visual = aprendizaje_visual or AprendizajeVisual(memoria=self.memoria)
        self.q = agente_q or AgenteQLearning(memoria=self.memoria)
        self.evaluador = Evaluacion()
        self.niveles = NivelesAprendizaje()

    def registrar_confirmacion(self, alimento, etiqueta, embedding, imagen=None,
                               estado_visual=None, prediccion=None):
        alimento = str(alimento).strip()
        etiqueta = str(etiqueta).strip()
        if not alimento: raise ValueError("El alimento es obligatorio")
        if etiqueta not in self.ESTADOS: raise ValueError("Etiqueta inválida")
        vector = self.visual.preparar_caracteristicas(embedding)
        if len(vector) != 1024: raise ValueError("Se espera embedding de 1024 dimensiones")
        estado_visual = estado_visual or "embedding_mobilenetv3"
        accion = prediccion if prediccion in self.ESTADOS else etiqueta
        correcto = accion == etiqueta
        # El Q-agent registra feedback y añade la muestra visual confirmada.
        self.q.registrar_feedback(estado_visual, accion, correcto,
            estado_real=etiqueta, alimento_ia=alimento, alimento_real=alimento,
            caracteristicas=vector, imagen=imagen)
        return {"registrado": True, "alimento": alimento, "etiqueta": etiqueta,
                "feedback_correcto": correcto}

    def contar_ejemplos(self, alimento):
        rows = self.memoria.obtener_memorias_visuales(alimento)
        conteo = {s: 0 for s in self.ESTADOS}
        for r in rows:
            estado = r.get("estado")
            if estado in conteo: conteo[estado] += 1
        return conteo

    def evaluar_desde_directorio(self, alimento, reconocedor, base="datos/evaluacion"):
        """Carga un holdout separado: base/alimento/Buen estado y Mal estado.
        Las fotos de este directorio nunca se agregan a la memoria visual.
        """
        base = Path(base) / str(alimento)
        muestras = []
        extensiones = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for etiqueta in self.ESTADOS:
            carpeta = base / etiqueta
            if not carpeta.is_dir():
                raise FileNotFoundError(f"Falta carpeta independiente: {carpeta}")
            archivos = sorted(p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in extensiones)
            if not archivos:
                raise ValueError(f"No hay imagenes de evaluacion en {carpeta}")
            for archivo in archivos:
                muestras.append((reconocedor.extraer_caracteristicas(str(archivo)), etiqueta))
        return self.evaluar_y_actualizar_nivel(alimento, muestras)

    def evaluar_y_actualizar_nivel(self, alimento, muestras_independientes):
        # No guardar aquí las muestras de evaluación en memoria visual.
        resultado = self.evaluador.evaluar(self.visual, alimento, muestras_independientes)
        conteo = self.contar_ejemplos(alimento)
        nivel = self.niveles.determinar(conteo, resultado["precision_equilibrada"])
        self.memoria.guardar_nivel(alimento, nivel, sum(conteo.values()),
                                  resultado["precision_equilibrada"])
        resultado.update({"alimento": alimento, "nivel": nivel,
                          "ejemplos_entrenamiento": conteo})
        return resultado

    def obtener_progreso(self, alimento):
        conteo = self.contar_ejemplos(alimento)
        niveles_guardados = self.memoria.obtener_niveles()
        nivel_guardado = next((x for x in niveles_guardados
                               if str(x.get("alimento", "")).casefold() == alimento.casefold()), None)
        return {"alimento": alimento, "ejemplos_por_clase": conteo,
                "total": sum(conteo.values()),
                "nivel_evaluado": nivel_guardado.get("nivel") if nivel_guardado else "Básico",
                "nota": "El nivel se asigna mediante evaluación independiente, no por cantidad de feedback solamente."}
