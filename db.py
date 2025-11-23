import sqlite3

DB_NAME = "finansys.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# -------------------------------
# Inicialización de BD
# -------------------------------
def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Crear tabla empresas
    c.execute("""
        CREATE TABLE IF NOT EXISTS empresas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            sector TEXT,
            fecha_registro TEXT
        );
    """)

    # Crear tabla estados financieros
    c.execute("""
       CREATE TABLE IF NOT EXISTS estados_financieros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa_id INTEGER,
            tipo_estado TEXT,      -- BG o ER
            periodicidad TEXT,     -- "anual" o "mensual"
            año INTEGER,
            mes INTEGER,           -- 1 a 12 o NULL
            cuenta TEXT,
            monto REAL
        );
    """)

    conn.commit()
    conn.close()

# -------------------------------
# Migraciones
# -------------------------------
def migrate_add_periodicidad():
    """
    Agrega la columna 'periodicidad' a la tabla estados_financieros
    si no existe aún.
    Normalmente: 'anual' o 'mensual'.
    """
    conn = get_connection()
    c = conn.cursor()

    try:
        c.execute("ALTER TABLE estados_financieros ADD COLUMN periodicidad TEXT")
        conn.commit()
        print("✅ Columna 'periodicidad' agregada")
    except Exception as e:
        print("ℹ La columna 'periodicidad' ya existe o no se pudo:", e)

    conn.close()


def migrate_add_mes():
    """
    Agrega la columna 'mes' a la tabla estados_financieros
    (1-12 para mensual, NULL para anual).
    """
    conn = get_connection()
    c = conn.cursor()

    try:
        c.execute("ALTER TABLE estados_financieros ADD COLUMN mes INTEGER")
        conn.commit()
        print("✅ Columna 'mes' agregada")
    except Exception as e:
        print("ℹ La columna 'mes' ya existe o no se pudo:", e)

    conn.close()
# -------------------------------
# CRUD Empresas
# -------------------------------
def crear_empresa(nombre, sector, fecha_registro):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO empresas (nombre, sector, fecha_registro)
        VALUES (?, ?, ?)
    """, (nombre, sector, fecha_registro))

    conn.commit()
    conn.close()

def obtener_empresas():
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT id, nombre FROM empresas")
    data = c.fetchall()

    conn.close()
    return data

# -------------------------------
# CRUD Estados financieros
# -------------------------------
def guardar_estado(empresa_id, tipo_estado, periodicidad, año, mes, cuenta, monto):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO estados_financieros 
        (empresa_id, tipo_estado, periodicidad, año, mes, cuenta, monto)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (empresa_id, tipo_estado, periodicidad, año, mes, cuenta, monto))

    conn.commit()
    conn.close()

def obtener_estado(empresa_id, tipo_estado, año, periodicidad=None, mes=None):
    """
    Obtiene un estado financiero filtrando por:
    - empresa
    - tipo (BG / ER)
    - año
    - opcional: periodicidad (anual / mensual)
    - opcional: mes (1-12)
    """

    conn = get_connection()
    c = conn.cursor()

    query = """
        SELECT cuenta, monto
        FROM estados_financieros
        WHERE empresa_id = ?
        AND tipo_estado = ?
        AND año = ?
    """

    params = [empresa_id, tipo_estado, año]

    # Filtro opcional por periodicidad
    if periodicidad is not None:
        query += " AND periodicidad = ?"
        params.append(periodicidad)

    # Filtro opcional por mes
    if mes is not None:
        query += " AND mes = ?"
        params.append(mes)

    c.execute(query, params)
    rows = c.fetchall()

    conn.close()

    return rows
