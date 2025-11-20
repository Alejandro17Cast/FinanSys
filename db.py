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

    # Estados financieros (BG y ER)
    c.execute("""
       CREATE TABLE IF NOT EXISTS estados_financieros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            tipo_estado TEXT,      -- BG o ER
            periodicidad TEXT,     -- "anual" o "mensual"
            año INTEGER,
            mes INTEGER,           -- 1 a 12, NULL si es anual
            cuenta TEXT,
             monto REAL
        );
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------------
# ✔ FUNCIONES CRUD
# -------------------------------------------------------

# 🟦 1. Registrar empresa
def crear_empresa(nombre, sector, fecha_registro):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO empresas (nombre, sector, fecha_registro)
        VALUES (?, ?, ?)
    """, (nombre, sector, fecha_registro))
    conn.commit()
    conn.close()

# 🟩 2. Listar empresas
def obtener_empresas():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, nombre FROM empresas")
    data = c.fetchall()
    conn.close()
    return data

# 🟧 3. Guardar un estado financiero (BG o ER)
def guardar_estado(empresa_id, tipo_estado, año, cuenta, monto):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO estados_financieros (empresa_id, tipo_estado, año, cuenta, monto)
        VALUES (?, ?, ?, ?, ?)
    """, (empresa_id, tipo_estado, año, cuenta, monto))
    conn.commit()
    conn.close()

# 🟨 4. Obtener estados financieros por empresa y año
def obtener_estado(empresa_id, tipo_estado, año):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT cuenta, monto
        FROM estados_financieros
        WHERE empresa_id = ? AND tipo_estado = ? AND año = ?
    """, (empresa_id, tipo_estado, año))
    rows = c.fetchall()
    conn.close()
    return rows
