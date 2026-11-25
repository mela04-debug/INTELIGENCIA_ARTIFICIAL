
import sqlite3
import json

from contextlib import contextmanager
from pathlib import Path
from datetime import datetime


class MemoriaAgente:
    """
    Administra la memoria persistente del agente mediante SQLite.

    Guarda:
    - Experiencias y correcciones.
    - Características visuales.
    - Valores Q-Learning.
    - Estadísticas por alimento.
    - Niveles de aprendizaje.
    """

    def __init__(self, ruta_db=None):

        raiz = Path(__file__).resolve().parents[2]

        self.ruta_db = (
            Path(ruta_db)
            if ruta_db
            else raiz / "datos" / "aprendizaje.db"
        )

        self.ruta_db.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.inicializar_base_datos()

    # ==========================================
    # CONEXIÓN SQLITE
    # ==========================================

    @contextmanager
    def conectar(self):

        conexion = sqlite3.connect(self.ruta_db)

        conexion.row_factory = sqlite3.Row

        conexion.execute("PRAGMA foreign_keys = ON")

        try:
            yield conexion
            conexion.commit()

        except Exception:
            conexion.rollback()
            raise

        finally:
            conexion.close()

    # ==========================================
    # INICIALIZAR BASE DE DATOS
    # ==========================================

    def inicializar_base_datos(self):

        with self.conectar() as conexion:

            # 1. EXPERIENCIAS DEL AGENTE

            conexion.execute("""
                CREATE TABLE IF NOT EXISTS experiencias (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    alimento TEXT NOT NULL,

                    estado_visual TEXT,

                    estado_predicho TEXT,

                    estado_real TEXT,

                    imagen TEXT,

                    correcto INTEGER NOT NULL,

                    fecha TEXT NOT NULL
                )
            """)

            # 2. MEMORIA VISUAL

            conexion.execute("""
                CREATE TABLE IF NOT EXISTS memoria_visual (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    alimento TEXT NOT NULL,

                    estado TEXT NOT NULL,

                    caracteristicas TEXT NOT NULL,

                    imagen TEXT,

                    fecha TEXT NOT NULL
                )
            """)

            # 3. VALORES Q-LEARNING

            conexion.execute("""
                CREATE TABLE IF NOT EXISTS valores_q (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    alimento TEXT NOT NULL,

                    estado_visual TEXT NOT NULL,

                    accion TEXT NOT NULL,

                    valor REAL NOT NULL DEFAULT 0,

                    UNIQUE(alimento, estado_visual, accion)
                )
            """)

            # 4. ESTADÍSTICAS

            conexion.execute("""
                CREATE TABLE IF NOT EXISTS estadisticas (

                    alimento TEXT PRIMARY KEY,

                    intentos INTEGER DEFAULT 0,

                    correctos INTEGER DEFAULT 0,

                    fallas INTEGER DEFAULT 0,

                    corregidos INTEGER DEFAULT 0
                )
            """)

            # 5. NIVELES DE APRENDIZAJE

            conexion.execute("""
                CREATE TABLE IF NOT EXISTS niveles (

                    alimento TEXT PRIMARY KEY,

                    nivel TEXT DEFAULT 'Básico',

                    ejemplos INTEGER DEFAULT 0,

                    precision REAL DEFAULT 0,

                    fecha_actualizacion TEXT
                )
            """)

    # ==========================================
    # NORMALIZAR ALIMENTOS
    # ==========================================

    @staticmethod
    def normalizar_alimento(alimento):

        return str(
            alimento or "Desconocido"
        ).strip().casefold()

    # ==========================================
    # EXPERIENCIAS
    # ==========================================

    def guardar_experiencia(
        self,
        alimento,
        estado_visual,
        estado_predicho,
        estado_real,
        correcto,
        imagen=None
    ):

        fecha = datetime.now().isoformat(
            timespec="seconds"
        )

        with self.conectar() as conexion:

            conexion.execute("""
                INSERT INTO experiencias (

                    alimento,
                    estado_visual,
                    estado_predicho,
                    estado_real,
                    imagen,
                    correcto,
                    fecha

                )
                VALUES (?, ?, ?, ?, ?, ?, ?)

            """, (
                alimento,
                estado_visual,
                estado_predicho,
                estado_real,
                str(imagen) if imagen else None,
                int(bool(correcto)),
                fecha
            ))

    def obtener_experiencias(self, alimento=None):

        consulta = """
            SELECT *
            FROM experiencias
            WHERE 1=1
        """

        parametros = []

        if alimento:

            consulta += " AND LOWER(alimento) = LOWER(?)"

            parametros.append(alimento)

        consulta += " ORDER BY id DESC"

        with self.conectar() as conexion:

            filas = conexion.execute(
                consulta,
                parametros
            ).fetchall()

        return [dict(fila) for fila in filas]

    # ==========================================
    # MEMORIA VISUAL
    # ==========================================

    def guardar_memoria_visual(
        self,
        alimento,
        estado,
        caracteristicas,
        imagen=None
    ):

        fecha = datetime.now().isoformat(
            timespec="seconds"
        )

        caracteristicas_json = json.dumps(
            caracteristicas,
            ensure_ascii=False
        )

        with self.conectar() as conexion:

            conexion.execute("""
                INSERT INTO memoria_visual (

                    alimento,
                    estado,
                    caracteristicas,
                    imagen,
                    fecha

                )
                VALUES (?, ?, ?, ?, ?)

            """, (
                alimento,
                estado,
                caracteristicas_json,
                str(imagen) if imagen else None,
                fecha
            ))
            return True


    def obtener_memorias_visuales(
        self,
        alimento=None,
        estado=None
    ):

        consulta = """
            SELECT *
            FROM memoria_visual
            WHERE 1=1
        """

        parametros = []

        if alimento:

            consulta += " AND LOWER(alimento) = LOWER(?)"

            parametros.append(alimento)

        if estado:

            consulta += " AND estado = ?"

            parametros.append(estado)

        consulta += " ORDER BY id DESC"

        with self.conectar() as conexion:

            filas = conexion.execute(
                consulta,
                parametros
            ).fetchall()

        resultados = []

        for fila in filas:

            registro = dict(fila)

            registro["caracteristicas"] = json.loads(
                registro["caracteristicas"]
            )

            resultados.append(registro)

        return resultados

    # ==========================================
    # VALORES Q-LEARNING
    # ==========================================

    def obtener_valor_q(
        self,
        alimento,
        estado_visual,
        accion
    ):

        with self.conectar() as conexion:

            fila = conexion.execute("""
                SELECT valor

                FROM valores_q

                WHERE LOWER(alimento) = LOWER(?)

                AND estado_visual = ?

                AND accion = ?

            """, (
                alimento,
                estado_visual,
                accion
            )).fetchone()

        return float(fila["valor"]) if fila else 0.0

    def guardar_valor_q(
        self,
        alimento,
        estado_visual,
        accion,
        valor
    ):

        with self.conectar() as conexion:

            conexion.execute("""
                INSERT INTO valores_q (

                    alimento,
                    estado_visual,
                    accion,
                    valor

                )

                VALUES (?, ?, ?, ?)

                ON CONFLICT(
                    alimento,
                    estado_visual,
                    accion
                )

                DO UPDATE SET
                    valor = excluded.valor

            """, (
                alimento,
                estado_visual,
                accion,
                float(valor)
            ))

    def obtener_q_table(self, alimento):

        with self.conectar() as conexion:

            filas = conexion.execute("""
                SELECT
                    estado_visual,
                    accion,
                    valor

                FROM valores_q

                WHERE LOWER(alimento) = LOWER(?)

            """, (alimento,)).fetchall()

        tabla = {}

        for fila in filas:

            estado = fila["estado_visual"]

            accion = fila["accion"]

            tabla.setdefault(estado, {})

            tabla[estado][accion] = float(
                fila["valor"]
            )

        return tabla

    # ==========================================
    # ESTADÍSTICAS
    # ==========================================

    def actualizar_estadisticas(
        self,
        alimento,
        correcto
    ):

        with self.conectar() as conexion:

            conexion.execute("""
                INSERT OR IGNORE INTO estadisticas (
                    alimento
                )
                VALUES (?)

            """, (alimento,))

            if correcto:

                conexion.execute("""
                    UPDATE estadisticas

                    SET
                        intentos = intentos + 1,
                        correctos = correctos + 1

                    WHERE LOWER(alimento) = LOWER(?)

                """, (alimento,))

            else:

                conexion.execute("""
                    UPDATE estadisticas

                    SET
                        intentos = intentos + 1,
                        fallas = fallas + 1,
                        corregidos = corregidos + 1

                    WHERE LOWER(alimento) = LOWER(?)

                """, (alimento,))

    def obtener_estadisticas(self, alimento=None):

        with self.conectar() as conexion:

            if alimento:

                filas = conexion.execute("""
                    SELECT *

                    FROM estadisticas

                    WHERE LOWER(alimento) = LOWER(?)

                """, (alimento,)).fetchall()

            else:

                filas = conexion.execute("""
                    SELECT *

                    FROM estadisticas

                    ORDER BY alimento

                """).fetchall()

        return [dict(fila) for fila in filas]

    # ==========================================
    # NIVELES DE APRENDIZAJE
    # ==========================================

    def guardar_nivel(
        self,
        alimento,
        nivel,
        ejemplos,
        precision
    ):

        fecha = datetime.now().isoformat(
            timespec="seconds"
        )

        with self.conectar() as conexion:

            conexion.execute("""
                INSERT INTO niveles (

                    alimento,
                    nivel,
                    ejemplos,
                    precision,
                    fecha_actualizacion

                )

                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(alimento)

                DO UPDATE SET

                    nivel = excluded.nivel,

                    ejemplos = excluded.ejemplos,

                    precision = excluded.precision,

                    fecha_actualizacion =
                        excluded.fecha_actualizacion

            """, (
                alimento,
                nivel,
                int(ejemplos),
                float(precision),
                fecha
            ))

    def obtener_niveles(self):

        with self.conectar() as conexion:

            filas = conexion.execute("""
                SELECT *

                FROM niveles

                ORDER BY alimento

            """).fetchall()

        return [dict(fila) for fila in filas]

    # ==========================================
    # RESUMEN GENERAL
    # ==========================================

    def obtener_resumen(self):

        with self.conectar() as conexion:

            experiencias = conexion.execute("""
                SELECT COUNT(*) AS total

                FROM experiencias

            """).fetchone()["total"]

            memorias = conexion.execute("""
                SELECT COUNT(*) AS total

                FROM memoria_visual

            """).fetchone()["total"]

            alimentos = conexion.execute("""
                SELECT COUNT(*) AS total

                FROM estadisticas

            """).fetchone()["total"]

            valores = conexion.execute("""
                SELECT COUNT(*) AS total

                FROM valores_q

            """).fetchone()["total"]

        return {

            "experiencias": experiencias,

            "memorias_visuales": memorias,

            "alimentos_registrados": alimentos,

            "valores_q": valores

        }
