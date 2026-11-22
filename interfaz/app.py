import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageOps

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.vision.reconocimiento import ReconocedorAlimentos
from src.vision.procesamiento import analizar_imagen, obtener_imagenes_procesamiento
from src.agente.q_learning import AgenteQLearning
from src.estimacion.vida_util import estimar_vida_util

RANGOS_REFERENCIA = {
    "manzana": (7, 14),
    "platano": (3, 7),
    "plátano": (3, 7),
    "banana": (3, 7),
    "frutilla": (2, 4),
    "fresa": (2, 4),
    "naranja": (10, 21),
    "tomate": (4, 7),
    "pan": (3, 5),
    "cebolla": (30, 60),
    "limon": (10, 14),
    "limón": (10, 14),
    "zanahoria": (21, 28),
    "apio": (7, 14),
    "perejil": (5, 10),
    "espinaca": (3, 5),
    "lechuga": (5, 10)
}

COLORES = {
    "fondo": "#F8FAFC",
    "tarjeta": "#FFFFFF",
    "borde": "#E2E8F0",
    "texto_principal": "#0F172A",
    "texto_secundario": "#64748B",
    "primario": "#2563EB",
    "primario_hover": "#1D4ED8",
    "exito_bg": "#DCFCE7",
    "exito_fg": "#166534",
    "alerta_bg": "#FEF3C7",
    "alerta_fg": "#92400E",
    "peligro_bg": "#FEE2E2",
    "peligro_fg": "#991B1B",
    "neutro_bg": "#F1F5F9",
}

class AplicacionAlimentos:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Detección de Alimentos · IA Didáctica")
        self.ventana.geometry("880x800")
        self.ventana.minsize(750, 650)
        self.ventana.configure(bg=COLORES["fondo"])

        self.ruta_imagen = None
        self.imagen_tk = None
        self.estado_actual = None
        self.diagnostico_actual = None
        self.alimento_ia_original = None
        self.feedback_registrado = False

        # Lista inicial dinámica (irá creciendo con los nombres que tú escribas)
        self.opciones_alimentos = [
            "Manzana", "Plátano", "Naranja", "Tomate", "Frutilla",
            "Cebolla", "Limón", "Zanahoria", "Mango", "Piña", "Pera",
            "Espinaca", "Apio", "Perejil", "Lechuga", "Papa"
        ]

        self.reconocedor = None
        self.agente = AgenteQLearning()

        self.configurar_estilos()

        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_analisis = tk.Frame(self.notebook, bg=COLORES["fondo"])
        self.notebook.add(self.tab_analisis, text=" 📷 Análisis y Diagnóstico ")

        self.tab_memoria = tk.Frame(self.notebook, bg=COLORES["fondo"])
        self.notebook.add(self.tab_memoria, text=" 🧠 Cerebro de la IA (Patrones) ")

        self.tab_procesamiento = tk.Frame(self.notebook, bg=COLORES["fondo"])
        self.notebook.add(self.tab_procesamiento,text="  Procesamiento de imagen  ")

        self.configurar_scroll_analisis()
        self.configurar_scroll_procesamiento()

        self.crear_interfaz_analisis()
        self.crear_interfaz_memoria()
        self.crear_interfaz_procesamiento()

        self.ventana.bind_all("<MouseWheel>", self._desplazar_scroll)
        self.imagen_tk = None
        self.estado_texto.set("Inicializando modelo neuronal...")
        self.ventana.after(100, self.cargar_modelo)

    def configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Modern.TCombobox",
            fieldbackground=COLORES["tarjeta"],
            background=COLORES["neutro_bg"],
            bordercolor=COLORES["borde"],
            arrowcolor=COLORES["texto_principal"],
            padding=6
        )
        style.configure("TNotebook", background=COLORES["fondo"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[15, 8])

    def configurar_scroll_analisis(self):
        self.canvas = tk.Canvas(self.tab_analisis, bg=COLORES["fondo"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.tab_analisis, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.contenido_frame = tk.Frame(self.canvas, bg=COLORES["fondo"])
        self.canvas_window = self.canvas.create_window((0, 0), window=self.contenido_frame, anchor="nw")

        self.contenido_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

    def configurar_scroll_procesamiento(self):
        self.canvas_procesamiento = tk.Canvas(self.tab_procesamiento, bg=COLORES["fondo"], highlightthickness=0)

        self.scrollbar_procesamiento = ttk.Scrollbar(self.tab_procesamiento, orient="vertical", command=self.canvas_procesamiento.yview)
        self.canvas_procesamiento.configure(yscrollcommand=self.scrollbar_procesamiento.set)
        self.scrollbar_procesamiento.pack(side="right", fill="y")
        self.canvas_procesamiento.pack(side="left", fill="both", expand=True)

        self.contenido_procesamiento = tk.Frame(self.canvas_procesamiento, bg=COLORES["fondo"])

        self.canvas_procesamiento_window = (self.canvas_procesamiento.create_window((0, 0), window=self.contenido_procesamiento, anchor="nw"))

        self.contenido_procesamiento.bind("<Configure>",lambda e: self.canvas_procesamiento.configure(scrollregion=self.canvas_procesamiento.bbox("all")))

        self.canvas_procesamiento.bind("<Configure>", lambda e: self.canvas_procesamiento.itemconfig(self.canvas_procesamiento_window, width=e.width))

    def _desplazar_scroll(self, event):
        pestaña_actual = self.notebook.select()

        if pestaña_actual == str(self.tab_analisis):
            canvas = self.canvas

        elif pestaña_actual == str(self.tab_procesamiento):
            canvas = self.canvas_procesamiento

        else:
            return

        canvas.yview_scroll(int(-1 * (event.delta / 120)),"units")

        return "break"

    def crear_tarjeta(self, parent, pady=(0, 16)):
        card = tk.Frame(parent, bg=COLORES["tarjeta"], highlightbackground=COLORES["borde"], highlightthickness=1, padx=24, pady=18)
        card.pack(fill="x", padx=45, pady=pady)
        return card

    def crear_interfaz_analisis(self):
        header = tk.Frame(self.contenido_frame, bg=COLORES["fondo"])
        header.pack(fill="x", padx=45, pady=(24, 14))

        tk.Label(header, text="Detección de Alimentos", font=("Segoe UI", 22, "bold"), fg=COLORES["texto_principal"], bg=COLORES["fondo"]).pack(anchor="w")
        tk.Label(header, text="Clasificación visual, estimación de vida útil y aprendizaje por refuerzo", font=("Segoe UI", 10), fg=COLORES["texto_secundario"], bg=COLORES["fondo"]).pack(anchor="w", pady=(2, 0))

        card_visor = self.crear_tarjeta(self.contenido_frame, pady=(0, 14))
        self.contenedor_imagen = tk.Frame(card_visor, width=380, height=220, bg=COLORES["neutro_bg"], highlightbackground=COLORES["borde"], highlightthickness=1)
        self.contenedor_imagen.pack_propagate(False)
        self.contenedor_imagen.pack(pady=(4, 14))

        self.panel_imagen = tk.Label(self.contenedor_imagen, text="Selecciona una fotografía del alimento", font=("Segoe UI", 10), fg=COLORES["texto_secundario"], bg=COLORES["neutro_bg"])
        self.panel_imagen.pack(expand=True, fill="both")

        botones_visor = tk.Frame(card_visor, bg=COLORES["tarjeta"])
        botones_visor.pack()

        self.boton_imagen = tk.Button(botones_visor, text="Cargar imagen", font=("Segoe UI", 10, "bold"), fg="#FFFFFF", bg=COLORES["primario"], relief="flat", cursor="hand2", padx=20, pady=8, command=self.seleccionar_imagen)
        self.boton_imagen.pack(side="left", padx=10)

        self.boton_limpiar = tk.Button(botones_visor, text="🗑️ Limpiar y subir otra", font=("Segoe UI", 10, "bold"), fg=COLORES["texto_principal"], bg=COLORES["borde"], relief="flat", cursor="hand2", padx=20, pady=8, command=self.limpiar_interfaz)
        self.boton_limpiar.pack(side="left", padx=10)

        self.estado_texto = tk.StringVar(value="Preparando sistema...")
        tk.Label(card_visor, textvariable=self.estado_texto, font=("Segoe UI", 9), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(pady=(12, 0))

        card_resultados = self.crear_tarjeta(self.contenido_frame, pady=(0, 14))
        tk.Label(card_resultados, text="DIAGNÓSTICO DEL AGENTE", font=("Segoe UI", 9, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 10))

        grid_metricas = tk.Frame(card_resultados, bg=COLORES["tarjeta"])
        grid_metricas.pack(fill="x", pady=(0, 12))
        grid_metricas.columnconfigure(0, weight=1)
        grid_metricas.columnconfigure(1, weight=1)

        self.alimento_var = tk.StringVar(value="--")
        self.confianza_var = tk.StringVar(value="--")
        self.diagnostico_var = tk.StringVar(value="--")
        self.vida_util_var = tk.StringVar(value="--")

        self._crear_dato(grid_metricas, "Alimento detectado", self.alimento_var, 0, 0)
        self._crear_dato(grid_metricas, "Confianza del modelo", self.confianza_var, 0, 1)
        self._crear_dato(grid_metricas, "Estado sanitario", self.diagnostico_var, 1, 0)
        self._crear_dato(grid_metricas, "Vida útil estimada", self.vida_util_var, 1, 1)

        self.alerta_texto = tk.StringVar(value="En espera de análisis.")
        self.banner_alerta = tk.Frame(card_resultados, bg=COLORES["neutro_bg"], padx=14, pady=10)
        self.banner_alerta.pack(fill="x", pady=(4, 0))
        self.label_alerta = tk.Label(self.banner_alerta, textvariable=self.alerta_texto, font=("Segoe UI", 10, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["neutro_bg"], justify="left")
        self.label_alerta.pack(anchor="w")

        card_feedback = self.crear_tarjeta(self.contenido_frame, pady=(0, 28))
        tk.Label(card_feedback, text="ENTRENAMIENTO POR REFUERZO (Q-LEARNING)", font=("Segoe UI", 9, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 4))
        tk.Label(card_feedback, text="¿El diagnóstico y el alimento son correctos?", font=("Segoe UI", 10), fg=COLORES["texto_principal"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 12))

        botones_frame = tk.Frame(card_feedback, bg=COLORES["tarjeta"])
        botones_frame.pack(anchor="w")

        self.boton_correcto = tk.Button(botones_frame, text="✓  Todo Correcto (+1)", font=("Segoe UI", 10, "bold"), fg=COLORES["exito_fg"], bg=COLORES["exito_bg"], relief="flat", cursor="hand2", padx=18, pady=7, command=self.confirmar_diagnostico, state="disabled")
        self.boton_correcto.pack(side="left", padx=(0, 10))

        self.boton_corregir = tk.Button(botones_frame, text="✕  Corregir a la IA (-1)", font=("Segoe UI", 10, "bold"), fg=COLORES["peligro_fg"], bg=COLORES["peligro_bg"], relief="flat", cursor="hand2", padx=18, pady=7, command=self.mostrar_correccion, state="disabled")
        self.boton_corregir.pack(side="left")

        self.correccion_frame = tk.Frame(card_feedback, bg=COLORES["tarjeta"])
        self.fruta_seleccionada = tk.StringVar()
        self.estado_seleccionado = tk.StringVar()

        tk.Label(self.correccion_frame, text="Elige de la lista o ESCRIBE el nombre nuevo del alimento y su estado:", font=("Segoe UI", 9, "bold"), fg=COLORES["primario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(12, 4))

        fila_corr = tk.Frame(self.correccion_frame, bg=COLORES["tarjeta"])
        fila_corr.pack(anchor="w")

        # CAMBIO CLAVE: state="normal" para permitir escribir nombres nuevos
        self.lista_frutas = ttk.Combobox(fila_corr, textvariable=self.fruta_seleccionada, values=self.opciones_alimentos, state="normal", style="Modern.TCombobox", width=18, font=("Segoe UI", 10))
        self.lista_frutas.pack(side="left", padx=(0, 10))

        self.lista_estados = ttk.Combobox(fila_corr, textvariable=self.estado_seleccionado, values=["Buen estado", "Mal estado"], state="readonly", style="Modern.TCombobox", width=15, font=("Segoe UI", 10))
        self.lista_estados.pack(side="left", padx=(0, 10))

        tk.Button(fila_corr, text="Guardar aprendizaje", font=("Segoe UI", 9, "bold"), fg="#FFFFFF", bg=COLORES["texto_principal"], relief="flat", cursor="hand2", padx=14, pady=6, command=self.guardar_correccion).pack(side="left")

        self.mensaje_texto = tk.StringVar(value="")
        tk.Label(card_feedback, textvariable=self.mensaje_texto, font=("Segoe UI", 9, "bold"), fg=COLORES["primario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(10, 0))

    def crear_interfaz_memoria(self):
        header = tk.Frame(self.tab_memoria, bg=COLORES["fondo"])
        header.pack(fill="x", padx=45, pady=(24, 14))

        tk.Label(header, text="Análisis de Patrones Visuales", font=("Segoe UI", 22, "bold"), fg=COLORES["texto_principal"], bg=COLORES["fondo"]).pack(anchor="w")
        tk.Label(header, text="Aquí puedes ver exactamente qué características extrae la IA para aprender.", font=("Segoe UI", 10), fg=COLORES["texto_secundario"], bg=COLORES["fondo"]).pack(anchor="w", pady=(2, 0))

        card_info = self.crear_tarjeta(self.tab_memoria, pady=(10, 14))

        tk.Label(card_info, text="EL AGENTE NO VE PÍXELES. VE LO SIGUIENTE:", font=("Segoe UI", 9, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 15))

        self.patron_var = tk.StringVar(value="Sube una imagen para ver el patrón.")
        self.tono_var = tk.StringVar(value="--")
        self.manchas_var = tk.StringVar(value="--")
        self.textura_var = tk.StringVar(value="--")

        grid_memoria = tk.Frame(card_info, bg=COLORES["tarjeta"])
        grid_memoria.pack(fill="x", pady=5)

        self._crear_dato_memoria(grid_memoria, "🎨 Tono de Color:", self.tono_var, 0)
        self._crear_dato_memoria(grid_memoria, "🦠 Nivel de Manchas:", self.manchas_var, 1)
        self._crear_dato_memoria(grid_memoria, "🧊 Tipo de Textura:", self.textura_var, 2)

        tk.Frame(card_info, height=1, bg=COLORES["borde"]).pack(fill="x", pady=20)

        tk.Label(card_info, text="CÓDIGO DE MEMORIA (El que se guarda en su JSON):", font=("Segoe UI", 9, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 5))

        lbl_patron = tk.Label(card_info, textvariable=self.patron_var, font=("Consolas", 11, "bold"), fg=COLORES["exito_fg"], bg=COLORES["exito_bg"], padx=15, pady=10)
        lbl_patron.pack(fill="x", pady=(5, 10))

        texto_explicativo = "Al corregir a la IA en la pestaña anterior (incluso si escribes un nombre nuevo), el agente guarda esa asociación en su JSON para aplicarla en futuros análisis."
        msg = tk.Message(card_info, text=texto_explicativo, font=("Segoe UI", 9), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"], width=600)
        msg.pack(anchor="w")

    def crear_interfaz_procesamiento(self):
        header = tk.Frame(self.contenido_procesamiento, bg=COLORES["fondo"])
        header.pack(fill="x", padx=45, pady=(24, 14))
        tk.Label(header, text="Procesamiento de Imagen", font=("Segoe UI", 22, "bold"), fg=COLORES["texto_principal"], bg=COLORES["fondo"]).pack(anchor="w")
        tk.Label(header, text="Visualización de la imagen original, sus canales RGB y su conversión a escala de grises.", font=("Segoe UI", 10), fg=COLORES["texto_secundario"], bg=COLORES["fondo"], wraplength=700, justify="left").pack(anchor="w", pady=(2, 0))
        card = self.crear_tarjeta(self.contenido_procesamiento, pady=(10, 14))

        tk.Label(card, text="VISUALIZACIÓN DE COMPONENTES", font=("Segoe UI", 9, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 12))

        grid = tk.Frame(card, bg=COLORES["tarjeta"])
        grid.pack(fill="both", expand=True)

        for columna in range(2):
            grid.columnconfigure(columna, weight=1)

        self.paneles_procesamiento = {}
        self.imagenes_procesamiento_tk = {}

        componentes = [("original", "Imagen original", 0, 0), ("gris", "Escala de grises", 0, 1), ("r", "Canal R - Rojo", 1, 0), ("g", "Canal G - Verde", 1, 1), ("b", "Canal B - Azul", 2, 0)]

        for clave, titulo, fila, columna in componentes:
            panel = tk.Frame(grid, bg=COLORES["neutro_bg"], padx=8, pady=8)
            panel.grid(row=fila, column=columna, sticky="nsew", padx=6, pady=6)

            tk.Label(panel, text=titulo, font=("Segoe UI", 10, "bold"), fg=COLORES["texto_principal"], bg=COLORES["neutro_bg"]).pack(pady=(0, 6))

            imagen_label = tk.Label(panel, text="Selecciona una imagen para visualizar", font=("Segoe UI", 9), fg=COLORES["texto_secundario"], bg=COLORES["neutro_bg"], anchor="center", justify="center")
            imagen_label.pack(fill="both", expand=True)

            self.paneles_procesamiento[clave] = imagen_label

        card_info = tk.Frame(grid, bg=COLORES["tarjeta"], padx=10, pady=10)
        card_info.grid(row=2, column=1, sticky="nsew", padx=6, pady=6)

        tk.Label(card_info, text="INFORMACIÓN DE LA IMAGEN", font=("Segoe UI", 9, "bold"), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w", pady=(0, 10))

        self.dimension_var = tk.StringVar(value="--")
        self.media_r_var = tk.StringVar(value="--")
        self.media_g_var = tk.StringVar(value="--")
        self.media_b_var = tk.StringVar(value="--")
        self.media_gris_var = tk.StringVar(value="--")

        datos = [("Dimensiones:", self.dimension_var), ("Media R:", self.media_r_var), ("Media G:", self.media_g_var), ("Media B:", self.media_b_var), ("Media gris:", self.media_gris_var)]

        for etiqueta, variable in datos:
            fila = tk.Frame(card_info, bg=COLORES["tarjeta"])
            fila.pack(fill="x", pady=4)

            tk.Label(fila, text=etiqueta, font=("Segoe UI", 9), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(side="left")
            tk.Label(fila, textvariable=variable, font=("Segoe UI", 9, "bold"), fg=COLORES["texto_principal"], bg=COLORES["tarjeta"]).pack(side="right")

    def actualizar_vista_procesamiento(self):
        """
        Obtiene los componentes de la imagen y actualiza los paneles de visualización.
        """
        if self.ruta_imagen is None:
            return

        try:
            datos = obtener_imagenes_procesamiento(self.ruta_imagen)

            componentes = {
            "original": datos["imagen_rgb"],
            "r": datos["canal_r"],
            "g": datos["canal_g"],
            "b": datos["canal_b"],
            "gris": datos["imagen_gris"]
            }

            for clave, matriz in componentes.items():

                imagen = Image.fromarray(matriz)

                imagen = ImageOps.contain(imagen, (260, 160),method=Image.Resampling.LANCZOS)

                imagen_tk = ImageTk.PhotoImage(imagen)

                self.imagenes_procesamiento_tk[clave] = imagen_tk

                self.paneles_procesamiento[clave].config(image=imagen_tk, text="")

            self.dimension_var.set(f'{datos["ancho"]} x {datos["alto"]} píxeles')

            self.media_r_var.set(f'{datos["media_r"]:.2f}')
            self.media_g_var.set(f'{datos["media_g"]:.2f}')
            self.media_b_var.set(f'{datos["media_b"]:.2f}')
            self.media_gris_var.set(f'{datos["media_gris"]:.2f}')

        except Exception as error:
            messagebox.showwarning("Procesamiento de imagen", f"No se pudo generar la visualización:\n{error}")

    def _crear_dato(self, parent, etiqueta, variable, fila, columna):
        bloque = tk.Frame(parent, bg=COLORES["tarjeta"])
        bloque.grid(row=fila, column=columna, sticky="w", pady=6, padx=(0, 15))
        tk.Label(bloque, text=etiqueta, font=("Segoe UI", 9), fg=COLORES["texto_secundario"], bg=COLORES["tarjeta"]).pack(anchor="w")
        tk.Label(bloque, textvariable=variable, font=("Segoe UI", 13, "bold"), fg=COLORES["texto_principal"], bg=COLORES["tarjeta"]).pack(anchor="w")

    def _crear_dato_memoria(self, parent, etiqueta, variable, fila):
        bloque = tk.Frame(parent, bg=COLORES["tarjeta"])
        bloque.grid(row=fila, column=0, sticky="w", pady=8)
        tk.Label(bloque, text=etiqueta, font=("Segoe UI", 10, "bold"), fg=COLORES["texto_principal"], bg=COLORES["tarjeta"], width=20, anchor="w").pack(side="left")
        tk.Label(bloque, textvariable=variable, font=("Segoe UI", 11), fg=COLORES["primario"], bg=COLORES["tarjeta"]).pack(side="left", padx=10)

    def cargar_modelo(self):
        try:
            self.reconocedor = ReconocedorAlimentos()
            self.estado_texto.set("Modelo listo · Selecciona una imagen para analizar")
            self.boton_imagen.config(state="normal")
        except Exception as error:
            self.estado_texto.set("Error al cargar el modelo de reconocimiento.")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{error}")

    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp"), ("Todos los archivos", "*.*")])
        if not ruta: return
        self.limpiar_resultados()
        self.ruta_imagen = Path(ruta)
        self.mostrar_imagen()
        self.actualizar_vista_procesamiento()
        self.analizar_alimento()

    def limpiar_interfaz(self):
        self.ruta_imagen = None
        self.imagen_tk = None
        self.panel_imagen.config(image="", text="Selecciona una fotografía del alimento")
        self.imagenes_procesamiento_tk.clear()

        for panel in self.paneles_procesamiento.values():
            panel.config(image="", text="Selecciona una imagen para visualizar")

        self.dimension_var.set("--")
        self.media_r_var.set("--")
        self.media_g_var.set("--")
        self.media_b_var.set("--")
        self.media_gris_var.set("--")

        self.limpiar_resultados()
        self.estado_texto.set("Pantalla limpia. Listo para una nueva imagen.")

    def limpiar_resultados(self):
        self.estado_actual = None
        self.diagnostico_actual = None
        self.alimento_ia_original = None
        self.feedback_registrado = False
        self.alimento_var.set("--")
        self.confianza_var.set("--")
        self.diagnostico_var.set("--")
        self.vida_util_var.set("--")
        self.alerta_texto.set("Analizando...")
        self.banner_alerta.config(bg=COLORES["neutro_bg"])
        self.label_alerta.config(bg=COLORES["neutro_bg"], fg=COLORES["texto_secundario"])
        self.mensaje_texto.set("")
        self.boton_correcto.config(state="disabled")
        self.boton_corregir.config(state="disabled")
        self.correccion_frame.pack_forget()

        self.patron_var.set("Sube una imagen para ver el patrón.")
        self.tono_var.set("--")
        self.manchas_var.set("--")
        self.textura_var.set("--")

    def mostrar_imagen(self):
        imagen = Image.open(self.ruta_imagen).convert("RGB")
        imagen = ImageOps.pad(imagen, (370, 210), method=Image.Resampling.LANCZOS, color=COLORES["neutro_bg"])
        self.imagen_tk = ImageTk.PhotoImage(imagen)
        self.panel_imagen.config(image=self.imagen_tk, text="")

    def actualizar_vida_util(self, alimento, diagnostico):
        clave = alimento.lower().strip()
        # Si escribiste un alimento nuevo que no está en RANGOS_REFERENCIA, usa (5, 10) días por defecto
        vida_min, vida_max = RANGOS_REFERENCIA.get(clave, (5, 10))

        es_malo = diagnostico.lower().strip() == "mal estado"
        estado_visual = "En mal estado" if es_malo else "Buen estado"

        estimacion = estimar_vida_util(tipo_alimento=alimento, dias_almacenado=0, vida_util_min=vida_min, vida_util_max=vida_max, estado_visual=estado_visual, almacenamiento="Ambiente")

        if es_malo:
            self.vida_util_var.set("0 días (No apto)")
        else:
            self.vida_util_var.set(f"{estimacion['dias_restantes_min']} a {estimacion['dias_restantes_max']} días (Ambiente)")

        nivel = estimacion["nivel_alerta"]
        self.alerta_texto.set(f"{nivel} · {estimacion['alerta']}")

        bg, fg = COLORES["exito_bg"], COLORES["exito_fg"]
        if nivel == "URGENTE": bg, fg = COLORES["peligro_bg"], COLORES["peligro_fg"]
        elif nivel == "PRECAUCIÓN": bg, fg = COLORES["alerta_bg"], COLORES["alerta_fg"]

        self.banner_alerta.config(bg=bg)
        self.label_alerta.config(bg=bg, fg=fg)

    def analizar_alimento(self):
        try:
            self.estado_texto.set("Procesando imagen...")
            identificacion = self.reconocedor.reconocer(self.ruta_imagen)

            if not identificacion["identificado"]:
                self.alerta_texto.set("Confianza insuficiente para identificar.")
                self.estado_texto.set("Intenta con una fotografía más clara.")
                return

            alimento_raw = identificacion["alimento"].capitalize()
            self.alimento_ia_original = alimento_raw

            caracteristicas = analizar_imagen(str(self.ruta_imagen))

            # El agente devuelve el estado sanitario y el nombre (corregido si ya lo aprendió antes)
            estado, diagnostico, alimento_aprendido = self.agente.predecir(caracteristicas, alimento_raw)

            self.estado_actual = estado
            self.diagnostico_actual = diagnostico

            self.alimento_var.set(alimento_aprendido)
            self.confianza_var.set(f"{identificacion['confianza'] * 100:.2f}%")
            self.diagnostico_var.set(diagnostico)

            self.actualizar_vida_util(alimento_aprendido, diagnostico)

            self.patron_var.set(estado)

            if "color_muy_oscuro" in estado: tono = "Muy oscuro"
            elif "color_apagado" in estado: tono = "Apagado"
            elif "color_brillante" in estado: tono = "Brillante"
            else: tono = "Desconocido"

            if "muchas_manchas" in estado: manchas = "Alto porcentaje de manchas"
            elif "algunas_manchas" in estado: manchas = "Ligeras manchas detectadas"
            elif "sin_manchas" in estado: manchas = "Sin manchas (Limpio)"
            else: manchas = "Desconocido"

            if "textura_irregular" in estado: textura = "Irregular o con arrugas"
            elif "textura_lisa" in estado: textura = "Superficie lisa"
            else: textura = "Desconocida"

            self.tono_var.set(tono)
            self.manchas_var.set(manchas)
            self.textura_var.set(textura)

            self.estado_texto.set("Análisis completado. ¿Es correcto?")
            self.boton_correcto.config(state="normal")
            self.boton_corregir.config(state="normal")

        except Exception as error:
            messagebox.showerror("Error", str(error))

    def confirmar_diagnostico(self):
        if self.feedback_registrado: return
        self.agente.registrar_feedback(self.estado_actual, self.diagnostico_actual, correcto=True)
        self.finalizar_feedback("✓ Aprendizaje validado y guardado en JSON.")

    def mostrar_correccion(self):
        if self.feedback_registrado: return
        self.fruta_seleccionada.set(self.alimento_var.get())
        self.estado_seleccionado.set(self.diagnostico_var.get())
        self.correccion_frame.pack(fill="x", pady=(4, 0))

    def guardar_correccion(self):
        fruta_real = self.fruta_seleccionada.get().strip().capitalize()
        estado_real = self.estado_seleccionado.get().strip()

        if not fruta_real or not estado_real:
            messagebox.showwarning("Atención", "Escribe o selecciona el alimento y su estado real.")
            return

        # Si escribiste un nombre nuevo, lo agregamos automáticamente al menú desplegable para el futuro
        if fruta_real not in self.opciones_alimentos:
            self.opciones_alimentos.append(fruta_real)
            self.lista_frutas['values'] = self.opciones_alimentos

        # Enviamos tanto el estado como el nombre nuevo al agente Q-Learning
        self.agente.registrar_feedback(
            self.estado_actual,
            self.diagnostico_actual,
            correcto=False,
            estado_real=estado_real,
            alimento_ia=self.alimento_ia_original,
            alimento_real=fruta_real
        )

        self.alimento_var.set(fruta_real)
        self.diagnostico_var.set(estado_real)
        self.actualizar_vida_util(fruta_real, estado_real)

        self.finalizar_feedback(f"✕ IA Corregida: Aprendió '{fruta_real}' en '{estado_real}'.")

    def finalizar_feedback(self, mensaje):
        self.feedback_registrado = True
        self.boton_correcto.config(state="disabled")
        self.boton_corregir.config(state="disabled")
        self.correccion_frame.pack_forget()
        self.mensaje_texto.set(mensaje)

if __name__ == "__main__":
    ventana = tk.Tk()
    aplicacion = AplicacionAlimentos(ventana)
    ventana.mainloop()