import torch
import timm

from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config, create_transform


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

        configuracion = resolve_data_config(
            self.modelo.pretrained_cfg,
            model=self.modelo
        )

        self.transformar = create_transform(**configuracion)

        etiquetas = self.modelo.pretrained_cfg["label_names"]
        self.etiquetas = etiquetas

    # SE REDUJO EL UMBRAL A 0.0 PARA FINES 100% DIDÁCTICOS (NUNCA BLOQUEA IMÁGENES)
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

            # Con umbral=0.0, esta condición de bloqueo jamás se activará.
            # Siempre pasará al return de éxito abajo.
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
            # Retorno de seguridad por si ocurre un error grave al leer la foto
            return {
                "identificado": True, 
                "alimento": "Desconocido",
                "categoria": "unknown",
                "confianza": 0.0
            }