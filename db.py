# db.py
import sqlite3

DB_NAME = "finansys.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Empresas
    c.execute("""
        CREATE TABLE IF NOT EXISTS empresas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            sector TEXT,
            fecha_registro TEXT
        );
    """)

    # Estados financieros
    c.execute("""
        CREATE TABLE IF NOT EXISTS estados_financieros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            tipo_estado TEXT,
            año INTEGER,
            cuenta TEXT,
            monto REAL,
            FOREIGN KEY (empresa_id) REFERENCES empresas(id)
        );
    """)

    # Análisis vertical
    c.execute("""
        CREATE TABLE IF NOT EXISTS analisis_vertical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            año INTEGER,
            cuenta TEXT,
            porcentaje REAL
        );
    """)

    # Análisis horizontal
    c.execute("""
        CREATE TABLE IF NOT EXISTS analisis_horizontal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            año_base INTEGER,
            año_comp INTEGER,
            cuenta TEXT,
            variacion_abs REAL,
            variacion_pct REAL
        );
    """)

    # Razones financieras
    c.execute("""
        CREATE TABLE IF NOT EXISTS razones_financieras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            año INTEGER,
            razon TEXT,
            valor REAL
        );
    """)

    # Sistema DuPont
    c.execute("""
        CREATE TABLE IF NOT EXISTS dupont (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            año INTEGER,
            margen REAL,
            rotacion REAL,
            apalancamiento REAL,
            roe REAL
        );
    """)

    # Flujo de efectivo
    c.execute("""
        CREATE TABLE IF NOT EXISTS flujo_efectivo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            año INTEGER,
            actividad TEXT,
            concepto TEXT,
            monto REAL
        );
    """)

    conn.commit()
    conn.close()
