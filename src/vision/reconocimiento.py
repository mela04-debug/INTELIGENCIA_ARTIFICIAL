
import torch
import timm

from pathlib import Path
from PIL import Image


class ReconocedorAlimentos:

    ETIQUETAS_ES = {
        "apple": "Manzana",
        "banana": "Banana",
        "orange": "Naranja",
        "carrot": "Zanahoria",
        "cucumber": "Pepino",
        "tomato": "Tomate",
        "broccoli": "Brócoli",
        "spinach": "Espinaca",
        "bell_pepper": "Pimiento",
        "potato": "Papa",
        "sweet_potato": "Camote",
        "eggplant": "Berenjena",
        "zucchini": "Calabacín",
        "okra": "Quimbombó",
        "peas": "Arvejas",
        "corn": "Maíz",
        "cauliflower": "Coliflor",
        "cabbage": "Repollo",
        "lettuce": "Lechuga",
        "onion": "Cebolla",
        "mango": "Mango",
        "pineapple": "Piña",
        "pomegranate": "Granada",
        "pear": "Pera",
        "kiwi": "Kiwi",
        "grapes": "Uvas",
        "strawberry": "Frutilla",
        "watermelon": "Sandía",
        "lemon": "Limón",
        "fig": "Higo"
    }

    def __init__(self):
        self.nombre_modelo = (
            "hf-hub:FaisalALMallah/nouri-produce-mobilenetv3"
        )

        self.modelo = timm.create_model(
            self.nombre_modelo,
            pretrained=True
        )

        self.modelo.eval()

        configuracion = timm.data.resolve_data_config(
            self.modelo.pretrained_cfg,
            model=self.modelo
        )

        self.transformar = timm.data.create_transform(
            **configuracion
        )

        self.etiquetas = self.modelo.pretrained_cfg["label_names"]

    def reconocer(self, ruta_imagen, umbral=0.0):
        ruta = Path(ruta_imagen)

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró la imagen: {ruta}"
            )

        try:
            imagen = Image.open(ruta).convert("RGB")
            imagen_tensor = self.transformar(imagen).unsqueeze(0)

            with torch.inference_mode():
                resultados = self.modelo(imagen_tensor)
                probabilidades = torch.softmax(resultados, dim=1)[0]

                confianza, indice = torch.max(
                    probabilidades,
                    dim=0
                )

            etiqueta = self.etiquetas[indice.item()]
            confianza = float(confianza.item())

            if confianza < umbral:
                return {
                    "identificado": False,
                    "alimento": "No identificado",
                    "categoria": etiqueta,
                    "confianza": confianza
                }

            return {
                "identificado": True,
                "alimento": self.ETIQUETAS_ES.get(
                    etiqueta, etiqueta
                ),
                "categoria": etiqueta,
                "confianza": confianza
            }

        except Exception as e:
            print(f"Error procesando la imagen: {e}")

            return {
                "identificado": False,
                "alimento": "Desconocido",
                "categoria": "unknown",
                "confianza": 0.0
            }

    def extraer_caracteristicas(self, ruta_imagen):
        """
        Extrae el embedding visual de MobileNetV3.

        Devuelve una lista de números que puede guardarse
        en SQLite mediante AprendizajeVisual.
        """

        ruta = Path(ruta_imagen)

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró la imagen: {ruta}"
            )

        imagen = Image.open(ruta).convert("RGB")
        imagen_tensor = self.transformar(imagen).unsqueeze(0)

        with torch.inference_mode():

            # Extraemos los mapas de características de la red.
            caracteristicas = self.modelo.forward_features(
                imagen_tensor
            )

            # Aplicamos el pooling y obtenemos el vector
            # anterior a la capa final de clasificación.
            embedding = self.modelo.forward_head(
                caracteristicas,
                pre_logits=True
            )

            embedding = embedding.flatten(start_dim=1)

        # Convertimos el tensor a una lista compatible con JSON/SQLite.
        vector = embedding[0].cpu().numpy().astype("float32")

        return vector.tolist()
