"""Reglas de niveles por alimento; thresholds transparentes y ajustables."""
class NivelesAprendizaje:
    NIVELES = ("Básico", "Intermedio", "Avanzado")
    REGLAS = {
        "Básico": {"min_por_clase": 0, "precision": 0.0},
        "Intermedio": {"min_por_clase": 20, "precision": 0.75},
        "Avanzado": {"min_por_clase": 50, "precision": 0.90},
    }

    def determinar(self, ejemplos_por_estado, precision):
        n_buen = int(ejemplos_por_estado.get("Buen estado", 0))
        n_mal = int(ejemplos_por_estado.get("Mal estado", 0))
        minimo = min(n_buen, n_mal)
        if minimo >= 50 and precision >= .90:
            return "Avanzado"
        if minimo >= 20 and precision >= .75:
            return "Intermedio"
        return "Básico"
