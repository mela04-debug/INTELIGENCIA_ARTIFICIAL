"""Evaluacion independiente; las muestras no se registran como entrenamiento."""
class Evaluacion:
    ESTADOS = ("Buen estado", "Mal estado")

    def evaluar(self, aprendizaje_visual, alimento, muestras):
        muestras = list(muestras)
        presentes = {str(etiqueta).strip() for _, etiqueta in muestras}
        faltantes = set(self.ESTADOS) - presentes
        if faltantes:
            raise ValueError("La evaluacion debe incluir ambas clases: Buen estado y Mal estado.")
        total = correctos = 0
        por_clase = {s: {"total": 0, "correctos": 0, "recall": 0.0} for s in self.ESTADOS}
        detalles = []
        for embedding, real in muestras:
            real = str(real).strip()
            if real not in self.ESTADOS:
                raise ValueError(f"Etiqueta invalida: {real}")
            resultado = aprendizaje_visual.predecir(alimento, embedding)
            pred = resultado.get("estado", "Sin coincidencia")
            ok = pred == real
            total += 1; correctos += int(ok)
            por_clase[real]["total"] += 1
            por_clase[real]["correctos"] += int(ok)
            detalles.append({"real": real, "predicho": pred, "correcto": ok})
        precision = correctos / total if total else 0.0
        recalls = []
        for clase in self.ESTADOS:
            v = por_clase[clase]
            v["recall"] = v["correctos"] / v["total"] if v["total"] else 0.0
            recalls.append(v["recall"])
        balanced = sum(recalls) / 2
        return {"total": total, "correctos": correctos, "precision": precision,
                "precision_equilibrada": balanced, "por_clase": por_clase, "detalles": detalles}
