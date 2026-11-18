import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageOps

# Ubicación principal del proyecto
PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.vision.reconocimiento import ReconocedorAlimentos
from src.vision.procesamiento import analizar_imagen
from src.agente.q_learning import AgenteQLearning
from src.estimacion.vida_util import estimar_vida_util


# Rangos de conservación de referencia (días mínimos y máximos a temperatura ambiente)
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
}

# Paleta Minimalista Moderna
COLORES = {
    "fondo": "#F8FAFC",          # Gris muy claro (Slate 50)
    "tarjeta": "#FFFFFF",        # Blanco puro
    "borde": "#E2E8F0",          # Borde sutil (Slate 200)
    "texto_principal": "#0F172A",# Azul marino casi negro (Slate 900)
    "texto_secundario": "#64748B",# Gris medio (Slate 500)
    "primario": "#2563EB",       # Azul moderno (Blue 600)
    "primario_hover": "#1D4ED8", # Azul oscuro (Blue 700)
    "exito_bg": "#DCFCE7",       # Verde suave
    "exito_fg": "#166534",       # Verde texto
    "alerta_bg": "#FEF3C7",      # Ámbar suave
    "alerta_fg": "#92400E",      # Ámbar texto
    "peligro_bg": "#FEE2E2",     # Rojo suave
    "peligro_fg": "#991B1B",     # Rojo texto
    "neutro_bg": "#F1F5F9",      # Gris suave
}


class AplicacionAlimentos:

    def __init__(self, ventana):

        self.ventana = ventana
        self.ventana.title("Detección de Alimentos · IA")
        self.ventana.geometry("820x720")
        self.ventana.minsize(720, 620)
        self.ventana.configure(bg=COLORES["fondo"])

        self.ruta_imagen = None
        self.imagen_tk = None

        self.estado_actual = None
        self.diagnostico_actual = None
        self.feedback_registrado = False

        # Componentes del agente
        self.reconocedor = None
        self.agente = AgenteQLearning()

        self.configurar_estilos()
        self.configurar_scroll()
        self.crear_interfaz()

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

    def configurar_scroll(self):

        self.canvas = tk.Canvas(
            self.ventana,
            bg=COLORES["fondo"],
            highlightthickness=0
        )
        self.scrollbar = ttk.Scrollbar(
            self.ventana,
            orient="vertical",
            command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.contenido_frame = tk.Frame(self.canvas, bg=COLORES["fondo"])
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.contenido_frame,
            anchor="nw"
        )

        self.contenido_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def crear_tarjeta(self, parent, pady=(0, 16)):
        """Crea un contenedor estilo tarjeta moderna con borde sutil."""
        card = tk.Frame(
            parent,
            bg=COLORES["tarjeta"],
            highlightbackground=COLORES["borde"],
            highlightthickness=1,
            padx=24,
            pady=18
        )
        card.pack(fill="x", padx=45, pady=pady)
        return card

    def crear_interfaz(self):

        # --- ENCABEZADO MINIMALISTA ---
        header = tk.Frame(self.contenido_frame, bg=COLORES["fondo"])
        header.pack(fill="x", padx=45, pady=(24, 14))

        tk.Label(
            header,
            text="Detección de Alimentos",
            font=("Segoe UI", 22, "bold"),
            fg=COLORES["texto_principal"],
            bg=COLORES["fondo"]
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Clasificación visual, estimación de vida útil y aprendizaje por refuerzo",
            font=("Segoe UI", 10),
            fg=COLORES["texto_secundario"],
            bg=COLORES["fondo"]
        ).pack(anchor="w", pady=(2, 0))

        # --- TARJETA 1: VISOR Y CARGA DE IMAGEN ---
        card_visor = self.crear_tarjeta(self.contenido_frame, pady=(0, 14))

        self.contenedor_imagen = tk.Frame(
            card_visor,
            width=380,
            height=220,
            bg=COLORES["neutro_bg"],
            highlightbackground=COLORES["borde"],
            highlightthickness=1
        )
        self.contenedor_imagen.pack_propagate(False)
        self.contenedor_imagen.pack(pady=(4, 14))

        self.panel_imagen = tk.Label(
            self.contenedor_imagen,
            text="Selecciona una fotografía del alimento\n(JPG, PNG, WEBP)",
            font=("Segoe UI", 10),
            fg=COLORES["texto_secundario"],
            bg=COLORES["neutro_bg"]
        )
        self.panel_imagen.pack(expand=True, fill="both")

        self.boton_imagen = tk.Button(
            card_visor,
            text="Cargar imagen",
            font=("Segoe UI", 10, "bold"),
            fg="#FFFFFF",
            bg=COLORES["primario"],
            activebackground=COLORES["primario_hover"],
            activeforeground="#FFFFFF",
            relief="flat",
            cursor="hand2",
            padx=24,
            pady=9,
            command=self.seleccionar_imagen
        )
        self.boton_imagen.pack()
        self.boton_imagen.bind("<Enter>", lambda e: self.boton_imagen.config(bg=COLORES["primario_hover"]))
        self.boton_imagen.bind("<Leave>", lambda e: self.boton_imagen.config(bg=COLORES["primario"]))

        self.estado_texto = tk.StringVar(value="Preparando sistema...")
        tk.Label(
            card_visor,
            textvariable=self.estado_texto,
            font=("Segoe UI", 9),
            fg=COLORES["texto_secundario"],
            bg=COLORES["tarjeta"]
        ).pack(pady=(8, 0))

        # --- TARJETA 2: RESULTADOS DEL ANÁLISIS ---
        card_resultados = self.crear_tarjeta(self.contenido_frame, pady=(0, 14))

        tk.Label(
            card_resultados,
            text="DIAGNÓSTICO DEL AGENTE",
            font=("Segoe UI", 9, "bold"),
            fg=COLORES["texto_secundario"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w", pady=(0, 10))

        # Grid limpio de 2 columnas para métricas clave
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

        # Banner de alerta moderno
        self.alerta_texto = tk.StringVar(value="En espera de análisis de imagen.")
        self.banner_alerta = tk.Frame(
            card_resultados,
            bg=COLORES["neutro_bg"],
            padx=14,
            pady=10
        )
        self.banner_alerta.pack(fill="x", pady=(4, 0))

        self.label_alerta = tk.Label(
            self.banner_alerta,
            textvariable=self.alerta_texto,
            font=("Segoe UI", 10, "bold"),
            fg=COLORES["texto_secundario"],
            bg=COLORES["neutro_bg"],
            wraplength=660,
            justify="left"
        )
        self.label_alerta.pack(anchor="w")

        # --- TARJETA 3: RETROALIMENTACIÓN Q-LEARNING ---
        card_feedback = self.crear_tarjeta(self.contenido_frame, pady=(0, 28))

        tk.Label(
            card_feedback,
            text="ENTRENAMIENTO POR REFUERZO (Q-LEARNING)",
            font=("Segoe UI", 9, "bold"),
            fg=COLORES["texto_secundario"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            card_feedback,
            text="¿El diagnóstico emitido coincide con el estado real del alimento?",
            font=("Segoe UI", 10),
            fg=COLORES["texto_principal"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w", pady=(0, 12))

        botones_frame = tk.Frame(card_feedback, bg=COLORES["tarjeta"])
        botones_frame.pack(anchor="w")

        self.boton_correcto = tk.Button(
            botones_frame,
            text="✓  Confirmar (+1)",
            font=("Segoe UI", 10, "bold"),
            fg=COLORES["exito_fg"],
            bg=COLORES["exito_bg"],
            activebackground="#BBF7D0",
            activeforeground=COLORES["exito_fg"],
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=7,
            command=self.confirmar_diagnostico,
            state="disabled"
        )
        self.boton_correcto.pack(side="left", padx=(0, 10))

        self.boton_corregir = tk.Button(
            botones_frame,
            text="✕  Corregir (-1)",
            font=("Segoe UI", 10, "bold"),
            fg=COLORES["peligro_fg"],
            bg=COLORES["peligro_bg"],
            activebackground="#FECACA",
            activeforeground=COLORES["peligro_fg"],
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=7,
            command=self.mostrar_correccion,
            state="disabled"
        )
        self.boton_corregir.pack(side="left")

        # Panel desplegable de corrección
        self.correccion_frame = tk.Frame(card_feedback, bg=COLORES["tarjeta"])
        self.estado_seleccionado = tk.StringVar()

        tk.Label(
            self.correccion_frame,
            text="Selecciona el estado real para actualizar la Q-Table:",
            font=("Segoe UI", 9),
            fg=COLORES["texto_secundario"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w", pady=(12, 4))

        fila_corr = tk.Frame(self.correccion_frame, bg=COLORES["tarjeta"])
        fila_corr.pack(anchor="w")

        self.lista_estados = ttk.Combobox(
            fila_corr,
            textvariable=self.estado_seleccionado,
            values=["Buen estado", "Podrido"],
            state="readonly",
            style="Modern.TCombobox",
            width=18,
            font=("Segoe UI", 10)
        )
        self.lista_estados.pack(side="left", padx=(0, 10))

        tk.Button(
            fila_corr,
            text="Guardar aprendizaje",
            font=("Segoe UI", 9, "bold"),
            fg="#FFFFFF",
            bg=COLORES["texto_principal"],
            activebackground="#334155",
            activeforeground="#FFFFFF",
            relief="flat",
            cursor="hand2",
            padx=14,
            pady=6,
            command=self.guardar_correccion
        ).pack(side="left")

        self.mensaje_texto = tk.StringVar(value="")
        tk.Label(
            card_feedback,
            textvariable=self.mensaje_texto,
            font=("Segoe UI", 9, "bold"),
            fg=COLORES["primario"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w", pady=(10, 0))

    def _crear_dato(self, parent, etiqueta, variable, fila, columna):
        bloque = tk.Frame(parent, bg=COLORES["tarjeta"])
        bloque.grid(row=fila, column=columna, sticky="w", pady=6, padx=(0, 15))

        tk.Label(
            bloque,
            text=etiqueta,
            font=("Segoe UI", 9),
            fg=COLORES["texto_secundario"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w")

        tk.Label(
            bloque,
            textvariable=variable,
            font=("Segoe UI", 13, "bold"),
            fg=COLORES["texto_principal"],
            bg=COLORES["tarjeta"]
        ).pack(anchor="w")

    def cargar_modelo(self):
        try:
            self.reconocedor = ReconocedorAlimentos()
            self.estado_texto.set("Modelo listo · Selecciona una imagen para analizar")
            self.boton_imagen.config(state="normal")
        except Exception as error:
            self.estado_texto.set("Error al cargar el modelo de reconocimiento.")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{error}")

    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen de alimento",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not ruta:
            return

        self.ruta_imagen = Path(ruta)
        self.limpiar_resultados()
        self.mostrar_imagen()
        self.analizar_alimento()

    def limpiar_resultados(self):
        self.estado_actual = None
        self.diagnostico_actual = None
        self.feedback_registrado = False

        self.alimento_var.set("--")
        self.confianza_var.set("--")
        self.diagnostico_var.set("--")
        self.vida_util_var.set("--")

        self.alerta_texto.set("Analizando estado del producto...")
        self._estilo_banner(COLORES["neutro_bg"], COLORES["texto_secundario"])
        self.mensaje_texto.set("")

        self.boton_correcto.config(state="disabled")
        self.boton_corregir.config(state="disabled")
        self.correccion_frame.pack_forget()

    def _estilo_banner(self, bg, fg):
        self.banner_alerta.config(bg=bg)
        self.label_alerta.config(bg=bg, fg=fg)

    def mostrar_imagen(self):
        imagen = Image.open(self.ruta_imagen).convert("RGB")
        imagen = ImageOps.pad(
            imagen,
            (370, 210),
            method=Image.Resampling.LANCZOS,
            color=COLORES["neutro_bg"]
        )
        self.imagen_tk = ImageTk.PhotoImage(imagen)
        self.panel_imagen.config(image=self.imagen_tk, text="")

    def actualizar_vida_util(self, alimento, diagnostico):
        clave = alimento.lower().strip()
        vida_min, vida_max = RANGOS_REFERENCIA.get(clave, (3, 7))
        es_podrido = diagnostico.lower().strip() == "podrido"
        estado_visual = "En mal estado" if es_podrido else "Buen estado"

        estimacion = estimar_vida_util(
            tipo_alimento=alimento,
            dias_almacenado=0,
            vida_util_min=vida_min,
            vida_util_max=vida_max,
            estado_visual=estado_visual,
            almacenamiento="Ambiente"
        )

        if es_podrido:
            self.vida_util_var.set("0 días (No apto)")
        else:
            self.vida_util_var.set(
                f"{estimacion['dias_restantes_min']} a {estimacion['dias_restantes_max']} días (Ambiente)"
            )

        nivel = estimacion["nivel_alerta"]
        self.alerta_texto.set(f"{nivel} · {estimacion['alerta']}")

        if nivel == "URGENTE":
            self._estilo_banner(COLORES["peligro_bg"], COLORES["peligro_fg"])
        elif nivel == "PRECAUCIÓN":
            self._estilo_banner(COLORES["alerta_bg"], COLORES["alerta_fg"])
        else:
            self._estilo_banner(COLORES["exito_bg"], COLORES["exito_fg"])

    def analizar_alimento(self):
        try:
            self.estado_texto.set("Procesando imagen...")
            identificacion = self.reconocedor.reconocer(self.ruta_imagen)

            if not identificacion["identificado"]:
                self.alimento_var.set("No identificado")
                self.confianza_var.set(f"{identificacion['confianza'] * 100:.1f}%")
                self.diagnostico_var.set("No disponible")
                self.vida_util_var.set("No disponible")
                self.alerta_texto.set("Confianza insuficiente para identificar el alimento.")
                self._estilo_banner(COLORES["alerta_bg"], COLORES["alerta_fg"])
                self.estado_texto.set("Intenta con una fotografía más clara o centrada.")
                return

            alimento = identificacion["alimento"]
            caracteristicas = analizar_imagen(str(self.ruta_imagen))
            estado, diagnostico = self.agente.predecir(caracteristicas, alimento)

            self.estado_actual = estado
            self.diagnostico_actual = diagnostico

            self.alimento_var.set(alimento.capitalize())
            self.confianza_var.set(f"{identificacion['confianza'] * 100:.2f}%")
            self.diagnostico_var.set(diagnostico)

            self.actualizar_vida_util(alimento, diagnostico)
            self.estado_texto.set("Análisis completado correctamente.")

            self.boton_correcto.config(state="normal")
            self.boton_corregir.config(state="normal")

        except Exception as error:
            self.estado_texto.set("Ocurrió un error durante el análisis.")
            messagebox.showerror("Error de análisis", str(error))

    def confirmar_diagnostico(self):
        if self.feedback_registrado:
            return

        self.agente.registrar_feedback(
            self.estado_actual,
            self.diagnostico_actual,
            correcto=True
        )
        self.finalizar_feedback("✓ Recompensa (+1) aplicada y guardada en Q-Table.")

    def mostrar_correccion(self):
        if self.feedback_registrado:
            return
        self.estado_seleccionado.set("")
        self.correccion_frame.pack(fill="x", pady=(4, 0))

    def guardar_correccion(self):
        estado_real = self.estado_seleccionado.get()

        if estado_real not in ["Buen estado", "Podrido"]:
            messagebox.showwarning("Atención", "Selecciona el estado real del alimento.")
            return

        if estado_real.lower() == self.diagnostico_actual.lower():
            messagebox.showwarning(
                "Corrección no válida",
                "El estado seleccionado coincide con el diagnóstico actual."
            )
            return

        self.agente.registrar_feedback(
            self.estado_actual,
            self.diagnostico_actual,
            correcto=False,
            estado_real=estado_real
        )

        self.diagnostico_var.set(estado_real)
        alimento_actual = self.alimento_var.get()
        self.actualizar_vida_util(alimento_actual, estado_real)

        self.finalizar_feedback(
            f"✕ Penalización (-1) registrada. Política actualizada a: {estado_real}."
        )

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