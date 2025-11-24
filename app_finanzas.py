# app.py - FinanSys (versión completa con export y flujo directo/indirecto y KPIs)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from google import genai
import io
import os
import re
import base64
import traceback
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime

# Base de datos
from db import init_db, migrate_add_mes, migrate_add_periodicidad, crear_empresa, obtener_empresas, guardar_estado


# Inicializar base de datos
init_db()
migrate_add_periodicidad()
migrate_add_mes()

import streamlit as st



if "empresa_activa" not in st.session_state:
    st.session_state.empresa_activa = None

if "anio_activo" not in st.session_state:
    st.session_state.anio_activo = None

if "periodicidad_activa" not in st.session_state:
    st.session_state.periodicidad_activa = None

if "mes_activo" not in st.session_state:
    st.session_state.mes_activo = None



def cargar_estados_desde_bd():
    
    conn = get_connection()
    query = """
    SELECT 
        e.nombre AS empresa,
        ef.tipo_estado,
        ef.periodicidad,
        ef.año,
        ef.mes,
        ef.cuenta,
        ef.monto
    FROM estados_financieros ef
    JOIN empresas e ON ef.empresa_id = e.id
    ORDER BY e.nombre, ef.año, ef.mes
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df




if "db_estados" not in st.session_state:
    st.session_state.db_estados = {}

if "estado_seleccionado" not in st.session_state:
    st.session_state.estado_seleccionado = None




def generate_finasys_logo(size=512):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    center = size//2
    radius = size//2 - 20

    # ===== Círculo exterior con degradado =====
    for i in range(25):
        draw.ellipse(
            [
                (center-radius+i, center-radius+i),
                (center+radius-i, center+radius-i)
            ],
            outline=(0, 255 - i*5, 255, 255),
            width=4
        )

    # ===== Barras financieras =====
    bar_colors = [(0,255,255,255), (0,210,255,255), (0,170,255,255)]

    bar_width = size // 10
    spacing = size // 20
    base_y = center + size//8

    heights = [size//6, size//4, size//2.9]  # barras ascendentes

    for i, h in enumerate(heights):
        x1 = center - bar_width*2 + i*(bar_width + spacing)
        y1 = base_y - h
        x2 = x1 + bar_width
        y2 = base_y

        draw.rounded_rectangle([x1, y1, x2, y2], radius=20, fill=bar_colors[i])

    # Glow exterior
    img = img.filter(ImageFilter.GaussianBlur(1.5))

    return img


# Crear imagen en memoria
logo_img = generate_finasys_logo(450)

buf = io.BytesIO()
logo_img.save(buf, format="PNG")
logo_bytes = buf.getvalue()
# ----------------------------
# Config página y paletas
# ----------------------------
st.set_page_config(page_title="FinanSys", layout="wide", initial_sidebar_state="expanded", page_icon="💼")

# Colores (tema oscuro)
BACKGROUND_COLOR = "#0b1020"
CARD_COLOR = "#0f1724"
ACCENT_TITLE = "#9A7BFF"
ACCENT_SUBHEADER = "#2DD4BF"
TEXT_COLOR = "#E6EDF3"
BORDER_COLOR = "#1f2937"
COLORS = ["#9A7BFF", "#2DD4BF", "#FFD166", "#7C4DFF", "#FF6B6B", "#4DB6FF"]
#barra de carga
st.markdown("""
<style>
.progress-container {
    width: 100%;
    background-color: #1f1f1f;
    border-radius: 10px;
    padding: 4px;
    margin-top: 15px;
    margin-bottom: 10px;
}

.progress-bar {
    width: 0%;
    height: 20px;
    background: linear-gradient(90deg, #00d4ff, #0066ff);
    border-radius: 8px;
    animation: loading 2s ease-in-out infinite;
}

@keyframes loading {
    0% { width: 0%; }
    50% { width: 80%; }
    100% { width: 0%; }
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Fondo general */
.stApp {
    background-color: #020617 !important;
    color: #f1f5f9 !important;
}

/* Contenedor principal */
.block-container {
    padding-top: 1.5rem !important;
    background-color: #020617 !important;
}

/* Dataframes nativos */
[data-testid="stDataFrame"] {
    background-color: #020617 !important;
    border: 1px solid #1e293b !important;
    color: #e5e7eb !important;
}

/* Encabezados de tablas */
thead tr th {
    background-color: #1e293b !important;
    color: #f8fafc !important;
}

/* Filas de tablas */
tbody tr td {
    background-color: #020617 !important;
    color: #cbd5e1 !important;
}

/* Scroll tablas */
[data-testid="stDataFrame"] div {
    background-color: #020617 !important;
}
</style>
""", unsafe_allow_html=True)
# CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp {{background-color:{BACKGROUND_COLOR}; color:{TEXT_COLOR}; font-family: Inter, sans-serif;}}
    .card {{background:{CARD_COLOR}; border-radius:12px; padding:16px; border:1px solid {BORDER_COLOR}; margin-bottom:14px;}}
    .big-title {{color:{ACCENT_TITLE}; font-weight:700; font-size:2.1rem; text-align:center; padding:6px 0;}}
    .subheader {{color:{ACCENT_SUBHEADER}; font-weight:600; margin-bottom:6px;}}
    .small-muted {{color:#9aa3b2; font-size:0.9rem;}}
    hr.st-sep {{border:0; height:1px; background:#1e293b; margin:18px 0;}}
    .footer {{text-align:center; color:#8b94a3; font-size:0.85rem; padding:10px;}}
    </style>
     
""", unsafe_allow_html=True)
st.markdown("""
<style>
#pulse-loader {
    position: fixed;
    top: 40%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 28px;
    color: #9A7BFF;
    animation: pulse 1s infinite;
    display: none;
    z-index: 99999;
}

@keyframes pulse {
  0% { transform: scale(0.95) translate(-50%, -50%); opacity: 0.7; }
  50% { transform: scale(1.05) translate(-50%, -50%); opacity: 1; }
  100% { transform: scale(0.95) translate(-50%, -50%); opacity: 0.7; }
}
</style>

<div id="pulse-loader">Cargando sección...</div>

<script>
let oldSection = window.sessionStorage.getItem("current_section");
let newSection = "{{section}}";

if (oldSection !== newSection) {
    document.getElementById("pulse-loader").style.display = "block";
    setTimeout(() => {
        document.getElementById("pulse-loader").style.display = "none";
    }, 700);  // duración del efecto
}

window.sessionStorage.setItem("current_section", newSection);
</script>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>DashBoard - FinanSys</div>", unsafe_allow_html=True)

# ----------------------------
# Session state init
# ----------------------------
initial_keys = {
    "balances": [],
    "resultados": [],
    "ratios": [],
    "dupont_data": None,
    "efe_data": None,
    "ia_interpretation": ""
}
for k, v in initial_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Account mapping (robusto)
# ----------------------------
ACCOUNT_MAP_BALANCE = {
    "ACTIVO_CORRIENTE": ["activo corriente", "activos corrientes", "activo circulante"],
    "PASIVO_CORRIENTE": ["pasivo corriente", "pasivos corrientes", "pasivo circulante"],
    "ACTIVO_TOTAL": ["activo total", "total activos", "total de activos", "total activo"],
    "PASIVO_TOTAL": ["pasivo total", "pasivos totales", "total pasivo", "total pasivos"],
    "PATRIMONIO": ["patrimonio", "capital", "capital social", "patrimonio neto"],
    "INVENTARIO": ["inventario", "inventarios"],
    "CUENTAS_POR_COBRAR": ["cuentas por cobrar", "deudores", "clientes", "cuentas por cobrar neto"],
    "CAJA_BANCOS": ["caja", "bancos", "efectivo"],
    "ACTIVO_FIJO": ["activo fijo", "activo no corriente", "propiedad planta", "propiedades", "propiedad planta y equipo", "propiedad planta y equipo (ppe)"],
    "PROVEEDORES": ["proveedores", "cuentas por pagar", "proveedor"],
    "DEUDAS_FINANCIERAS": ["deuda", "prestamo", "préstamo", "credito", "crédito", "obligaciones", "deuda financiera"]
}

ACCOUNT_MAP_RESULTS = {
    "VENTAS_NETAS": ["ventas netas", "ventas", "ingresos por ventas", "ingresos"],
    "COSTO_VENTAS": ["costo de ventas", "costo de bienes vendidos", "costo ventas", "costo de lo vendido"],
    "UTILIDAD_BRUTA": ["utilidad bruta", "margen bruto"],
    "UTILIDAD_OPERATIVA": ["utilidad operativa", "ebit", "resultado operativo", "operativo"],
    "GASTO_INTERESES": ["gasto intereses", "gastos financieros", "intereses", "gasto de intereses"],
    "UTILIDAD_NETA": ["utilidad neta", "resultado neto", "ganancia neta", "utilidad (perdida) neta"],
    "DEPRECIACION": ["depreciación", "depreciacion", "amortizacion", "amortización", "depreciacion y amortizacion"]
}

DUPONT_BALANCE_CUENTAS = {

    "activo": [
        # Totales primero
        "total activo",
        "activo total",

        # Activos corrientes
        "caja", "banco", "bancos",
        "clientes", "cuentas por cobrar",
        "inventario", "inventarios",

        # Activos no corrientes
        "activo fijo",
        "propiedad",
        "planta",
        "equipo",
        "maquinaria",
        "vehiculos",
        "terrenos",
        "intangibles"
    ],

    "patrimonio": [
        "total patrimonio",
        "patrimonio",
        "capital social",
        "capital contable",
        "utilidades retenidas",
        "resultado acumulado",
        "superavit",
        "reservas"
    ]
}

DUPONT_ER_CUENTAS = {
    "ventas": [
        "ventas", "ingresos", "otros ingresos"
    ],
    "costos": [
        "costo de ventas", "costo ventas"
    ],
    "gastos": [
        "gastos administrativos", "gastos operativos",
        "gastos financieros", "gastos generales"
    ]
}
# ----------------------------
# Utilidades robustas
# ----------------------------


def ensure_columns(df: pd.DataFrame, name_col_candidates=None, value_col_candidates=None):
    """
    Asegura que el DataFrame devuelva columnas 'Cuenta' y 'Monto' (exactas).
    name_col_candidates y value_col_candidates son listas de alternativas a buscar.
    """
    if df is None:
        return None
    d = df.copy()
    # Normalizar nombres de columnas a minúsculas sin espacios
    cols_map = {c: c.strip() for c in d.columns}
    d.rename(columns=cols_map, inplace=True)
    lower_cols = {c.lower(): c for c in d.columns}

    # candidatos por defecto
    if name_col_candidates is None:
        name_col_candidates = ["cuenta", "nombre", "account", "concepto"]
    if value_col_candidates is None:
        value_col_candidates = ["monto", "valor", "amount", "importe"]

    name_col = None
    value_col = None

    for cand in name_col_candidates:
        if cand in lower_cols:
            name_col = lower_cols[cand]
            break

    for cand in value_col_candidates:
        if cand in lower_cols:
            value_col = lower_cols[cand]
            break

    # Si no encontró, intenta columnas que contengan las palabras
    if name_col is None:
        for k, orig in lower_cols.items():
            if "cuenta" in k or "nombre" in k or "account" in k:
                name_col = orig
                break

    if value_col is None:
        for k, orig in lower_cols.items():
            if "monto" in k or "valor" in k or "amount" in k or "importe" in k:
                value_col = orig
                break

    # Si no encontramos columnas válidas devolvemos None (caller lo manejará)
    if name_col is None or value_col is None:
        return None

    # renombrar a exacto
    d = d.rename(columns={name_col: "Cuenta", value_col: "Monto"})
    # aseguramos tipos
    d["Cuenta"] = d["Cuenta"].astype(str)
    d["Monto"] = to_numeric_series(d["Monto"])
    return d[["Cuenta", "Monto"]]

def clean_text(s: str) -> str:
    """Normalize a string: trim, lower, remove accents, collapse spaces."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # Replace accented chars
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n").replace("ç", "c")
    # unify hyphens and parentheses spacing
    s = re.sub(r"\s+", " ", s)
    return s

def norm_account_name(s):
    if pd.isna(s):
        return ""
    return clean_text(str(s))

def parse_number(x):
    """Parse numeric text robustly: handles parentheses, thousand separators, currency symbols."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    # parentheses => negative
    if re.match(r'^\(.*\)$', s):
        s = "-" + s.strip("()")
    # remove currency symbols, letters, spaces except - and .
    s = re.sub(r"[^\d\-\.\,]", "", s)
    # if comma used as decimal (e.g., 1.234,56) convert to standard
    if s.count(",") > 0 and s.count(".") > 0:
        # assume dot thousands, comma decimal
        s = s.replace(".", "").replace(",", ".")
    else:
        # if only commas, decide: if more than 1 comma -> thousands separators
        if s.count(",") > 1:
            s = s.replace(",", "")
        elif s.count(",") == 1 and s.count(".") == 0:
            # single comma: could be decimal separator
            s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def to_numeric_series(s):
    return pd.Series([parse_number(v) for v in s]).fillna(0.0)

def make_word_regex(term: str) -> str:
    # escape then add word boundaries
    t = re.escape(term.strip().lower())
    # allow optional plural/small variations by spaces/hyphens
    return r"\b" + t + r"\b"

def find_account_amount(df, patterns):
    """
    Sum amounts for patterns. Patterns are plain strings; we'll match with word boundaries to avoid partial matches.
    """
    if df is None or df.empty:
        return 0.0
    s = df.copy()
    s["Cuenta_norm"] = s["Cuenta"].apply(norm_account_name)
    s["Monto"] = to_numeric_series(s["Monto"])
    total = 0.0
    for p in patterns:
        pat = norm_account_name(p)
        if not pat:
            continue
        # If pattern has spaces, we'll match as phrase; otherwise word boundary
        try:
            mask = s["Cuenta_norm"].str.contains(make_word_regex(pat), regex=True, na=False)
        except Exception:
            mask = s["Cuenta_norm"].str.contains(pat, na=False)
        total += float(s.loc[mask, "Monto"].sum())
    return round(total, 2)
def safe_get(dictionary, key):
    try:
        val = dictionary.get(key, 0)
        if val is None or np.isnan(val):
            return 0
        return float(val)
    except:
        return 0

def map_accounts(df: pd.DataFrame, account_map: dict) -> dict:
    if df is None or "Cuenta" not in df.columns or "Monto" not in df.columns:
        return {k: 0.0 for k in account_map.keys()}

    d = df.copy()

    d["Cuenta"] = d["Cuenta"].astype(str)
    d["Cuenta_norm"] = d["Cuenta"].apply(norm_account_name)
    d["Monto"] = to_numeric_series(d["Monto"])

    result = {}

    for key, patterns in account_map.items():
        total = 0
        encontrados = []

        for p in patterns:
            mask = d["Cuenta_norm"].str.contains(norm_account_name(p), na=False)
            coincidencias = d[mask]

            if not coincidencias.empty:
                total += coincidencias["Monto"].sum()
                encontrados.extend(coincidencias["Cuenta"].tolist())

        result[key] = total

        # Debug: qué encontró realmente
        print(f"MAPEO [{key}] -> Total: {total}")
        if encontrados:
            print(f"  Coincidencias: {set(encontrados)}")
        else:
            print("  ⚠ No encontró ninguna cuenta")

    return result
    

def validar_balance(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df["cuenta_norm"] = df["cuenta"].str.lower()

    activos = df[df["cuenta_norm"].str.contains("activo", na=False)]["monto"].sum()
    pasivos = df[df["cuenta_norm"].str.contains("pasivo", na=False)]["monto"].sum()
    capital = df[df["cuenta_norm"].str.contains("capital|patrimonio", na=False)]["monto"].sum()

    diferencia = round(activos - (pasivos + capital), 2)

    return {
        "activo": activos,
        "pasivo": pasivos,
        "capital": capital,
        "diferencia": diferencia,
        "cuadra": abs(diferencia) < 1
    }

def detectar_cuentas_peligrosas(df_h):
    crit = df_h[df_h["Variación (%)"].abs() > 50]

    if not crit.empty:
        crit["Nivel de alerta"] = crit["Variación (%)"].apply(
            lambda x: "🚨 Crítico" if abs(x) > 100 else "⚠ Atención"
        )

    return crit
def clasificar_categoria(nombre):
    nombre = str(nombre).lower()

    if any(x in nombre for x in ["activo", "caja", "banco", "cliente", "inventario"]):
        return "Activos"
    if any(x in nombre for x in ["pasivo", "deuda", "proveedor"]):
        return "Pasivos"
    if any(x in nombre for x in ["capital", "patrimonio"]):
        return "Capital"
    if any(x in nombre for x in ["venta", "ingreso"]):
        return "Ingresos"
    if any(x in nombre for x in ["costo", "gasto"]):
        return "Gastos"

    return "Otros"

# ----------------------------
# Vertical & Horizontal analysis (robusto)
# ----------------------------
def detect_base_balance(df):
    """Try to detect total activo robustly. If not found, fallback to sum of numeric rows."""
    if df is None or df.empty:
        return 0.0
    d = df.copy()
    d["Cuenta_norm"] = d["Cuenta"].apply(norm_account_name)
    d["Monto"] = to_numeric_series(d["Monto"])
    # flexible patterns
    patterns = [
        r"\btotal\b.*\bactivo\b",
        r"\bactivo\b.*\btotal\b",
        r"\btotal de activos\b",
        r"\bactivos totales\b",
        r"\bactivo total\b",
        r"\btotal activos\b",
        r"\btotal\b.*\bactivo(s)?\b"
    ]
    mask = pd.Series(False, index=d.index)
    for p in patterns:
        try:
            mask = mask | d["Cuenta_norm"].str.contains(p, regex=True, na=False)
        except:
            mask = mask | d["Cuenta_norm"].str.contains(p, na=False)
    if mask.any():
        # if multiple matches, sum them (sometimes subtotal + total)
        val = float(d.loc[mask, "Monto"].sum())
        if val != 0:
            return val
    # fallback: try map_accounts ACTIVO_TOTAL
    bal_map = map_accounts(d, ACCOUNT_MAP_BALANCE)
    at = safe_get(bal_map, "ACTIVO_TOTAL")
    if at and at != 0:
        return at
    # last resort: sum of numeric rows, excluding rows that look like totals/subtotals
    exclude_mask = d["Cuenta_norm"].str.contains(r"\btotal\b|\bsubtotal\b|\btotal\b|\bsuma\b", regex=True, na=False)
    s = d.loc[~exclude_mask, "Monto"].sum()
    return float(s)

def detect_base_results(df):
    """Detect ventas/ingresos total, fallback to mapping or sum."""
    if df is None or df.empty:
        return 0.0
    d = df.copy()
    d["Cuenta_norm"] = d["Cuenta"].apply(norm_account_name)
    d["Monto"] = to_numeric_series(d["Monto"])
    patterns = [r"\bventas netas\b", r"\bventas\b", r"\bingresos\b", r"\bingreso\b"]
    mask = pd.Series(False, index=d.index)
    for p in patterns:
        try:
            mask = mask | d["Cuenta_norm"].str.contains(p, regex=True, na=False)
        except:
            mask = mask | d["Cuenta_norm"].str.contains(p, na=False)
    if mask.any():
        val = float(d.loc[mask, "Monto"].sum())
        if val != 0:
            return val
    res_map = map_accounts(d, ACCOUNT_MAP_RESULTS)
    vn = safe_get(res_map, "VENTAS_NETAS")
    if vn and vn != 0:
        return vn
    # fallback: sum of positive revenue-like lines or sum of all (last resort)
    positive_revenues = d.loc[d["Cuenta_norm"].str.contains("venta|ingreso|ingresos", na=False), "Monto"].sum()
    if positive_revenues != 0:
        return float(positive_revenues)
    return float(d["Monto"].sum())

def vertical_analysis(titulo, df, tipo='BG'):
    import pandas as pd
    import streamlit as st

    dfv = df.copy()

    # Normalizar columnas
    dfv.columns = dfv.columns.str.strip().str.lower()

    if "cuenta" not in dfv.columns or "monto" not in dfv.columns:
        st.error(f"Columnas incorrectas: {dfv.columns.tolist()}")
        return None

    # Normalizar nombres
    dfv["cuenta_norm"] = (
        dfv["cuenta"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )

    # Convertir montos a numérico seguro
    dfv["monto"] = pd.to_numeric(dfv["monto"], errors="coerce").fillna(0)

    total = None

    # =====================================================
    # 🏦 BALANCE GENERAL
    # =====================================================
    if tipo == "BG":

        # A. Buscar Total Activo explícito
        mask_total = dfv["cuenta_norm"].str.contains(
            r"total\s+activo|activo\s+total|activos\s+totales",
            regex=True,
            na=False
        )

        if mask_total.any():
            total = dfv.loc[mask_total, "monto"].sum()

        # B. Si no se encuentra, inferir activos
        if total is None or total == 0:

            activos_mask = dfv["cuenta_norm"].str.contains(
                r"activo|caja|banco|cliente|clientes|inventario|propiedad|planta|equipo|maquinaria|terreno|edificio|intangible",
                na=False
            )

            total = dfv.loc[activos_mask, "monto"].sum()

        if total == 0:
            st.error("❌ No se pudo calcular Total Activo.")
            return None

    # =====================================================
    # 📊 ESTADO DE RESULTADOS (VENTAS COMO BASE)
    # =====================================================
    elif tipo == "ER":

        # A. Buscar TOTAL VENTAS explícito
        mask_ventas = dfv["cuenta_norm"].str.contains(
            r"total\s+ventas|ventas\s+netas|ventas\s+totales",
            regex=True,
            na=False
        )

        if mask_ventas.any():
            total = dfv.loc[mask_ventas, "monto"].sum()

        # B. Si no encuentra explícito, buscar ventas normales
        if total is None or total == 0:

            ventas_mask = dfv["cuenta_norm"].str.contains(
                r"\bventas\b|\bingresos\s+operacionales\b|ingreso\s+por\s+ventas",
                regex=True,
                na=False
            )

            total = dfv.loc[ventas_mask, "monto"].sum()

        if total == 0:
            st.error("❌ No se pudo calcular Ventas Totales como base del análisis vertical.")
            return None

    else:
        st.error("Tipo inválido. Usa 'BG' o 'ER'")
        return None

    # =====================================================
    # 📐 CÁLCULO DE PORCENTAJES
    # =====================================================
    dfv["porcentaje"] = (dfv["monto"] / total) * 100

    # =====================================================
    # FORMATO FINAL
    # =====================================================
    dfv = dfv.rename(columns={
        "cuenta": "Cuenta",
        "monto": "Monto",
        "porcentaje": "Porcentaje",
        "cuenta_norm": "Cuenta_norm"
    })

    dfv["Porcentaje"] = dfv["Porcentaje"].round(2)

    return dfv
def horizontal_analysis(df_prev, df_act, period_prev, period_act):
    import pandas as pd
    import numpy as np

    dp = df_prev.copy()
    da = df_act.copy()

    # =============================
    # 1. Normalizar columnas
    # =============================
    dp.columns = dp.columns.str.strip().str.lower()
    da.columns = da.columns.str.strip().str.lower()

    # =============================
    # 2. Verificación estricta
    # =============================
    for df, periodo in [(dp, period_prev), (da, period_act)]:
        if "cuenta" not in df.columns or "monto" not in df.columns:
            raise Exception(
                f"❌ Columnas inválidas en {periodo}. "
                f"Se detectaron: {df.columns.tolist()}"
            )

    # =============================
    # 3. Normalizar cuentas
    # =============================
    dp["cuenta_norm"] = dp["cuenta"].astype(str).apply(norm_account_name)
    da["cuenta_norm"] = da["cuenta"].astype(str).apply(norm_account_name)

    # =============================
    # 4. Convertir montos
    # =============================
    dp["monto"] = to_numeric_series(dp["monto"])
    da["monto"] = to_numeric_series(da["monto"])

    # =============================
    # 5. Agrupar por cuenta normalizada
    # (por si hay repetidas)
    # =============================
    dp_group = (
        dp.groupby("cuenta_norm", as_index=False)["monto"]
        .sum()
        .rename(columns={"monto": "monto_anterior"})
    )

    da_group = (
        da.groupby(["cuenta_norm", "cuenta"], as_index=False)["monto"]
        .sum()
    )

    # =============================
    # 6. Merge inteligente
    # =============================
    merged = da_group.merge(
        dp_group,
        on="cuenta_norm",
        how="left"
    )

    merged["monto_anterior"] = merged["monto_anterior"].fillna(0.0)

    # =============================
    # 7. Crear columnas finales
    # =============================
    merged[f"{period_prev} (Monto)"] = merged["monto_anterior"].round(2)
    merged[f"{period_act} (Monto)"] = merged["monto"].round(2)

    merged["Variación"] = (
        merged[f"{period_act} (Monto)"] - merged[f"{period_prev} (Monto)"]
    ).round(2)

    prev_series = merged[f"{period_prev} (Monto)"].replace({0.0: np.nan})

    merged["Variación (%)"] = (
        (merged[f"{period_act} (Monto)"] / prev_series - 1) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).round(2)

    # =============================
    # 8. Limpieza final y orden
    # =============================
    merged = merged.rename(columns={
        "cuenta": "Cuenta"
    })

    merged = merged[[
        "Cuenta",
        f"{period_prev} (Monto)",
        f"{period_act} (Monto)",
        "Variación",
        "Variación (%)"
    ]]

    merged = merged.sort_values("Variación", key=lambda x: abs(x), ascending=False)

    return merged.reset_index(drop=True)

# ----------------------------
# Ratios, DuPont, CNT/CNO
# ----------------------------
def calc_cnt_cno(balance_df: pd.DataFrame):
    bal_map = map_accounts(balance_df, ACCOUNT_MAP_BALANCE)
    activo_corr = safe_get(bal_map, "ACTIVO_CORRIENTE")
    pasivo_corr = safe_get(bal_map, "PASIVO_CORRIENTE")
    CNT = round(activo_corr - pasivo_corr, 2)
    ACO = round(safe_get(bal_map, "INVENTARIO") + safe_get(bal_map, "CUENTAS_POR_COBRAR"), 2)
    PCO = round(safe_get(bal_map, "PROVEEDORES"), 2)
    CNO = round(ACO - PCO, 2)
    return CNT, CNO

def compute_ratios(bg, er):
    import pandas as pd
    import numpy as np

    # ========================
    # Limpieza básica
    # ========================
    bg = bg.copy()
    er = er.copy()

    bg.columns = bg.columns.str.lower().str.strip()
    er.columns = er.columns.str.lower().str.strip()

    for df in [bg, er]:
        df["cuenta"] = (
            df["cuenta"].astype(str)
            .str.lower()
            .str.strip()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )

        df["monto"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0)

    # ========================
    # BALANCE GENERAL
    # ========================
    activos = bg[bg["cuenta"].str.contains(r"activo", na=False)]["monto"].sum()

    # Si no hay Total Activos explícito, sumamos todo lo que sea activo
    if activos == 0:
        activos = bg[bg["cuenta"].str.contains(
            r"caja|banco|cliente|inventario|propiedad|planta|equipo|terreno|edificio|activo",
            na=False
        )]["monto"].sum()

    pasivos = bg[bg["cuenta"].str.contains(r"pasivo", na=False)]["monto"].sum()

    capital = bg[bg["cuenta"].str.contains(r"capital|patrimonio", na=False)]["monto"].sum()

    activos_corr = bg[bg["cuenta"].str.contains(
        r"activo circulante|activo corriente|caja|banco|efectivo",
        na=False
    )]["monto"].sum()

    pasivos_corr = bg[bg["cuenta"].str.contains(
        r"pasivo circulante|pasivo corriente|proveedores|cuentas por pagar",
        na=False
    )]["monto"].sum()

    activos_fijos = bg[bg["cuenta"].str.contains(
        r"activo fijo|propiedad|planta|equipo|maquinaria",
        na=False
    )]["monto"].sum()

    inventarios = bg[bg["cuenta"].str.contains("inventario|existencia", na=False)]["monto"].sum()
    cuentas_cobrar = bg[bg["cuenta"].str.contains("clientes|cuentas por cobrar", na=False)]["monto"].sum()

    # ========================
    # ESTADO DE RESULTADOS
    # ========================

    # Ventas
    ventas = er[er["cuenta"].str.contains(
        r"\bventas\b|ingresos operacionales|ingreso por ventas|facturacion",
        na=False
    )]["monto"].sum()

    # Si no encuentra ventas, suma todos los ingresos
    if ventas == 0:
        ventas = er[er["cuenta"].str.contains("ingreso", na=False)]["monto"].sum()

    # Costo de ventas
    costo_ventas = er[er["cuenta"].str.contains(
        r"costo de venta|costo|costos de produccion",
        na=False
    )]["monto"].sum()

    # Si no viene utilidad bruta, la calculamos
    utilidad_bruta = ventas - costo_ventas

    # Utilidad operativa
    utilidad_operativa = er[er["cuenta"].str.contains(
        r"utilidad operativa|resultado operativo|utilidad de operacion",
        na=False
    )]["monto"].sum()

    # Si no existe, la calculamos
    if utilidad_operativa == 0:
        gastos_operativos = er[er["cuenta"].str.contains(
            r"gastos operativos|gastos de operacion|gastos administrativos|gastos de venta",
            na=False
        )]["monto"].sum()

        utilidad_operativa = utilidad_bruta - gastos_operativos

    # Utilidad neta
    utilidad_neta = er[er["cuenta"].str.contains(
        r"utilidad neta|resultado neto|ganancia del ejercicio",
        na=False
    )]["monto"].sum()

    # Si no existe, la calculamos
    if utilidad_neta == 0:
        gastos_financieros = er[er["cuenta"].str.contains(
            r"gastos financieros|intereses",
            na=False
        )]["monto"].sum()

        impuestos = er[er["cuenta"].str.contains(
            r"impuesto|isr|renta",
            na=False
        )]["monto"].sum()

        utilidad_neta = utilidad_operativa - gastos_financieros - impuestos

    intereses = er[er["cuenta"].str.contains("interes|gasto financiero", na=False)]["monto"].sum()

    # ========================
    # RAZONES
    # ========================
    razon_circulante = activos_corr / pasivos_corr if pasivos_corr != 0 else np.nan
    razon_rapida = (activos_corr - inventarios) / pasivos_corr if pasivos_corr != 0 else np.nan

    rotacion_inventarios = costo_ventas / inventarios if inventarios != 0 else np.nan
    rotacion_cxc = ventas / cuentas_cobrar if cuentas_cobrar != 0 else np.nan
    periodo_cobro = 360 / rotacion_cxc if rotacion_cxc not in [0, np.nan] else np.nan
    rotacion_activos_fijos = ventas / activos_fijos if activos_fijos != 0 else np.nan
    rotacion_activos_totales = ventas / activos if activos != 0 else np.nan

    razon_endeudamiento = pasivos / activos if activos != 0 else np.nan
    pasivos_capital = pasivos / capital if capital != 0 else np.nan
    cobertura_intereses = utilidad_operativa / intereses if intereses != 0 else np.nan

    margen_bruto = utilidad_bruta / ventas if ventas != 0 else np.nan
    margen_operativo = utilidad_operativa / ventas if ventas != 0 else np.nan
    margen_neto = utilidad_neta / ventas if ventas != 0 else np.nan
    roa = utilidad_neta / activos if activos != 0 else np.nan

    # ========================
    # RESULTADO
    # ========================
    razones = {
        "Razón Circulante": razon_circulante,
        "Razón Rápida": razon_rapida,
        "Rotación de Inventarios": rotacion_inventarios,
        "Rotación Cuentas por Cobrar": rotacion_cxc,
        "Periodo Promedio de Cobro": periodo_cobro,
        "Rotación de Activos Fijos": rotacion_activos_fijos,
        "Rotación de Activos Totales": rotacion_activos_totales,
        "Razón de Endeudamiento": razon_endeudamiento,
        "Pasivo / Capital": pasivos_capital,
        "Cobertura de Intereses": cobertura_intereses,
        "Margen Bruto": margen_bruto,
        "Margen Operativo": margen_operativo,
        "Margen Neto": margen_neto,
        "ROA": roa
    }

    return razones

def obtener_valor_clasificado(df, keywords):
    df = df.copy()

    df["Cuenta_norm"] = df["Cuenta"].astype(str).str.lower().str.strip()
    df["Monto"] = to_numeric_series(df["Monto"])

    total = df[df["Cuenta_norm"].apply(
        lambda x: any(k in x for k in keywords)
    )]["Monto"].sum()

    return total


def compute_dupont(balance_df, results_df):
    import pandas as pd
    eps = 1e-9

    # ==============================
    # Normalizador de texto
    # ==============================
    def normalizar(txt):
        return (
            str(txt).lower()
            .strip()
            .replace("á", "a").replace("é", "e")
            .replace("í", "i").replace("ó", "o")
            .replace("ú", "u").replace("ñ", "n")
        )

    # ==============================
    # Detectar columnas
    # ==============================
    def detectar_columna(df, posibles):
        for col in df.columns:
            if normalizar(col) in posibles:
                return col
        raise Exception(f"No se encontró columna válida en: {list(df.columns)}")

    col_cuenta_bg = detectar_columna(balance_df, ["cuenta", "concepto", "nombre", "descripcion", "detalle"])
    col_monto_bg  = detectar_columna(balance_df, ["monto", "valor", "importe", "total"])

    col_cuenta_er = detectar_columna(results_df, ["cuenta", "concepto", "nombre", "descripcion", "detalle"])
    col_monto_er  = detectar_columna(results_df, ["monto", "valor", "importe", "total"])

    # ==============================
    # Copias con normalización
    # ==============================
    bg = balance_df.copy()
    er = results_df.copy()

    bg[col_cuenta_bg] = bg[col_cuenta_bg].apply(normalizar)
    er[col_cuenta_er] = er[col_cuenta_er].apply(normalizar)

    bg[col_monto_bg] = pd.to_numeric(bg[col_monto_bg], errors="coerce").fillna(0.0)
    er[col_monto_er] = pd.to_numeric(er[col_monto_er], errors="coerce").fillna(0.0)

    # ==============================
    # Función de suma
    # ==============================
    def sumar_por_palabras(df, col_cuenta, col_monto, palabras):
        total = 0
        for _, row in df.iterrows():
            cuenta = row[col_cuenta]
            for palabra in palabras:
                if palabra in cuenta:
                    total += row[col_monto]
                    break
        return total

    # ==============================
    # ACTIVO TOTAL
    # ==============================
    activo_total = sumar_por_palabras(
        bg, col_cuenta_bg, col_monto_bg,
        ["total activo", "activo total"]
    )

    if activo_total == 0:
        activo_total = sumar_por_palabras(
            bg, col_cuenta_bg, col_monto_bg,
            DUPONT_BALANCE_CUENTAS["activo"]
        )

    # ==============================
    # PATRIMONIO
    # ==============================
    patrimonio = sumar_por_palabras(
        bg, col_cuenta_bg, col_monto_bg,
        ["total patrimonio", "patrimonio"]
    )

    if patrimonio == 0:
        patrimonio = sumar_por_palabras(
            bg, col_cuenta_bg, col_monto_bg,
            DUPONT_BALANCE_CUENTAS["patrimonio"]
        )

    # ==============================
    # VENTAS
    # ==============================
    ventas = sumar_por_palabras(
        er, col_cuenta_er, col_monto_er,
        ["ventas netas", "ventas", "ingresos"]
    )

    # ==============================
    # COSTOS Y GASTOS
    # ==============================
    costos = sumar_por_palabras(
        er, col_cuenta_er, col_monto_er,
        ["costo de ventas", "costo ventas", "costos"]
    )

    gastos = sumar_por_palabras(
        er, col_cuenta_er, col_monto_er,
        ["gastos operativos", "gastos administrativos", "gastos generales", "gastos"]
    )

    # ==============================
    # UTILIDAD NETA
    # ==============================
    utilidad_neta = sumar_por_palabras(
        er, col_cuenta_er, col_monto_er,
        ["utilidad neta", "resultado neto", "ganancia del ejercicio"]
    )

    if utilidad_neta == 0:
        utilidad_neta = ventas - costos - gastos

    # ==============================
    # DUPONT
    # ==============================
    margen = utilidad_neta / max(ventas, eps)
    rotacion = ventas / max(activo_total, eps)
    apalancamiento = activo_total / max(patrimonio, eps)
    roe = margen * rotacion * apalancamiento

    # ==============================
    # Resultado
    # ==============================
    return pd.DataFrame({
        "Componente": [
            "Margen Neto",
            "Rotación de Activos",
            "Apalancamiento Financiero",
            "ROE DuPont"
        ],
        "Valor": [
            round(margen, 4),
            round(rotacion, 4),
            round(apalancamiento, 4),
            round(roe, 4)
        ]
    })
# ----------------------------
# Flujo de efectivo (Indirecto + Directo)
# ----------------------------

def remove_totals(df):
    d = df.copy()

    # Normalizar nombres de columnas
    col_map = {c.lower(): c for c in d.columns}

    # Columna cuenta (tu BD la guarda como 'cuenta')
    if "cuenta" in col_map:
        col_cuenta = col_map["cuenta"]
    else:
        raise Exception(f"No se encontró columna 'cuenta'. Columnas disponibles: {list(d.columns)}")

    # Crear columna normalizada
    d["Cuenta_norm"] = d[col_cuenta].astype(str).apply(norm_account_name)

    # Eliminar totales, sumas, subtotales
    d = d[~d["Cuenta_norm"].str.contains(
        r"total|totales|suma|resultado del ejercicio|utilidad del ejercicio",
        case=False,
        na=False
    )]

    return d


def compute_cashflow_indirect(balance_df_act: pd.DataFrame,
                              results_df_act: pd.DataFrame,
                              balance_df_prev: pd.DataFrame) -> pd.DataFrame:

    bal_a = remove_totals(balance_df_act)
    bal_p = remove_totals(balance_df_prev)
    res_a = remove_totals(results_df_act)

    # Detectar columnas monto
    col_bal_a = "Monto" if "Monto" in bal_a.columns else "monto"
    col_bal_p = "Monto" if "Monto" in bal_p.columns else "monto"
    col_res_a = "Monto" if "Monto" in res_a.columns else "monto"

    # --- Utilidad Neta ---
    utilidad_neta = res_a.loc[
        res_a["Cuenta_norm"].str.contains(r"utilidad neta|resultado neto|ganancia", na=False),
        col_res_a
    ].sum()

    # --- Depreciación ---
    depreciacion = res_a.loc[
        res_a["Cuenta_norm"].str.contains(r"deprecia|amortiz", na=False),
        col_res_a
    ].sum()

    # --- Variaciones operativas ---
    def get_val(df, pattern, col):
        return df.loc[df["Cuenta_norm"].str.contains(pattern, na=False), col].sum()

    cxc_a = get_val(bal_a, r"cobrar|cliente|deudor", col_bal_a)
    cxc_p = get_val(bal_p, r"cobrar|cliente|deudor", col_bal_p)

    inv_a = get_val(bal_a, r"invent", col_bal_a)
    inv_p = get_val(bal_p, r"invent", col_bal_p)

    prov_a = get_val(bal_a, r"pagar|proveedor", col_bal_a)
    prov_p = get_val(bal_p, r"pagar|proveedor", col_bal_p)

    # Variaciones
    cambio_cxc = cxc_a - cxc_p
    cambio_inv = inv_a - inv_p
    cambio_prov = prov_a - prov_p

    flujo_operativo = utilidad_neta + depreciacion - cambio_cxc - cambio_inv + cambio_prov

    # --- Inversión (activo fijo) ---
    af_a = get_val(bal_a, r"propiedad|activo fijo|equipo|ppe", col_bal_a)
    af_p = get_val(bal_p, r"propiedad|activo fijo|equipo|ppe", col_bal_p)

    flujo_inversion = -(af_a - af_p)

    # --- Financiamiento ---
    deuda_a = get_val(bal_a, r"deuda|obligacion|prestamo|credito", col_bal_a)
    deuda_p = get_val(bal_p, r"deuda|obligacion|prestamo|credito", col_bal_p)

    capital_a = get_val(bal_a, r"patrimonio|capital social", col_bal_a)
    capital_p = get_val(bal_p, r"patrimonio|capital social", col_bal_p)

    flujo_financiamiento = (deuda_a - deuda_p) + (capital_a - capital_p)

    flujo_total = flujo_operativo + flujo_inversion + flujo_financiamiento

    df = pd.DataFrame({
        "Concepto": [
            "Utilidad Neta",
            "Depreciación y Amortización",
            "Ajuste Cuentas por Cobrar",
            "Ajuste Inventarios",
            "Ajuste Proveedores",
            "Flujo de Operación (A)",
            "Flujo de Inversión (B)",
            "Flujo de Financiamiento (C)",
            "Flujo Neto (A+B+C)"
        ],
        "Monto": [
            round(utilidad_neta, 2),
            round(depreciacion, 2),
            round(-cambio_cxc, 2),
            round(-cambio_inv, 2),
            round(cambio_prov, 2),
            round(flujo_operativo, 2),
            round(flujo_inversion, 2),
            round(flujo_financiamiento, 2),
            round(flujo_total, 2)
        ]
    })

    return df
    return df
def compute_cashflow_direct(balance_df_act: pd.DataFrame, results_df_act: pd.DataFrame) -> pd.DataFrame:
    """
    Flujo de Efectivo Método DIRECTO adaptado a tu BD:
    Soporta 'monto' o 'Monto', 'cuenta' o 'Cuenta'
    """

    bal = remove_totals(balance_df_act)
    res = remove_totals(results_df_act)

    # Detectar columna monto automáticamente
    col_monto_bal = "Monto" if "Monto" in bal.columns else "monto"
    col_monto_res = "Monto" if "Monto" in res.columns else "monto"

    # -----------------------------
    # COBROS A CLIENTES
    # -----------------------------
    ventas = res.loc[
        res["Cuenta_norm"].str.contains(r"venta|ingreso", na=False),
        col_monto_res
    ].sum()

    cxc = bal.loc[
        bal["Cuenta_norm"].str.contains(r"cobrar|cliente|deudor", na=False),
        col_monto_bal
    ].sum()

    cobros_clientes = ventas - cxc

    # -----------------------------
    # PAGOS A PROVEEDORES
    # -----------------------------
    compras = res.loc[
        res["Cuenta_norm"].str.contains(r"costo", na=False),
        col_monto_res
    ].sum()

    cxp = bal.loc[
        bal["Cuenta_norm"].str.contains(r"pagar|proveedor", na=False),
        col_monto_bal
    ].sum()

    pagos_proveedores = -(compras - cxp)

    # -----------------------------
    # GASTOS OPERATIVOS PAGADOS
    # -----------------------------
    gastos_operativos = res.loc[
        res["Cuenta_norm"].str.contains(r"gasto|operaci|servicio", na=False),
        col_monto_res
    ].sum()

    pagos_operativos = -gastos_operativos

    # -----------------------------
    # IMPUESTOS & INTERESES
    # -----------------------------
    impuestos = res.loc[
        res["Cuenta_norm"].str.contains(r"impuesto|isr|iva|tribut", na=False),
        col_monto_res
    ].sum()

    intereses = res.loc[
        res["Cuenta_norm"].str.contains(r"interes|financ", na=False),
        col_monto_res
    ].sum()

    pagos_impuestos = -impuestos
    pagos_intereses = -intereses

    # -----------------------------
    # TOTAL
    # -----------------------------
    flujo_operativo = cobros_clientes + pagos_proveedores + pagos_operativos + pagos_impuestos + pagos_intereses

    df = pd.DataFrame({
        "Concepto": [
            "Cobros por ventas",
            "Pagos a proveedores",
            "Pagos operativos",
            "Pagos de impuestos",
            "Pagos de intereses",
            "Flujo Operativo Directo"
        ],
        "Monto": [
            round(cobros_clientes, 2),
            round(pagos_proveedores, 2),
            round(pagos_operativos, 2),
            round(pagos_impuestos, 2),
            round(pagos_intereses, 2),
            round(flujo_operativo, 2)
        ]
    })

    return df


# ----------------------------
# Estado de Origen y Aplicación de Fondos (EOAF)
# ----------------------------

def compute_eoaf(balance_prev, balance_act, results_act):

    import pandas as pd

    origenes = []
    aplicaciones = []

    # ======================================================
    # 1. Validación ER
    # ======================================================
    if not isinstance(results_act, pd.DataFrame) or results_act is None or results_act.empty:
        results_act = None

    # ======================================================
    # 2. Columnas
    # ======================================================
    col_map_prev = {c.lower(): c for c in balance_prev.columns}
    col_map_act  = {c.lower(): c for c in balance_act.columns}

    col_cuenta_prev = col_map_prev.get("cuenta")
    col_monto_prev  = col_map_prev.get("monto")
    col_tipo_prev   = col_map_prev.get("tipo") or col_map_prev.get("clasificacion") or col_map_prev.get("grupo")

    col_cuenta_act = col_map_act.get("cuenta")
    col_monto_act  = col_map_act.get("monto")
    col_tipo_act   = col_map_act.get("tipo") or col_map_act.get("clasificacion") or col_map_act.get("grupo")

    # ======================================================
    # 3. Normalización
    # ======================================================
    df_prev = balance_prev.copy()
    df_act  = balance_act.copy()

    df_prev["Cuenta_norm"] = df_prev[col_cuenta_prev].astype(str).apply(norm_account_name)
    df_act["Cuenta_norm"]  = df_act[col_cuenta_act].astype(str).apply(norm_account_name)

    df_prev["Monto"] = pd.to_numeric(df_prev[col_monto_prev], errors="coerce").fillna(0)
    df_act["Monto"]  = pd.to_numeric(df_act[col_monto_act], errors="coerce").fillna(0)

    df_prev["Tipo_norm"] = df_prev[col_tipo_prev].astype(str).str.lower() if col_tipo_prev else ""
    df_act["Tipo_norm"] = df_act[col_tipo_act].astype(str).str.lower() if col_tipo_act else ""

    df_prev = df_prev[~df_prev["Cuenta_norm"].str.contains("total", na=False)]
    df_act  = df_act[~df_act["Cuenta_norm"].str.contains("total", na=False)]

    # ======================================================
    # 4. Comparación de períodos
    # ======================================================
    merged = df_act.merge(
        df_prev[["Cuenta_norm", "Monto"]],
        on="Cuenta_norm",
        how="left"
    ).rename(columns={"Monto_x": "Monto", "Monto_y": "Monto_Anterior"})

    merged["Monto_Anterior"] = merged["Monto_Anterior"].fillna(0)
    merged["Variacion"] = merged["Monto"] - merged["Monto_Anterior"]

    # Total de financiamiento
    financiamiento_total = merged.loc[
        merged["Cuenta_norm"].str.contains("proveedor|deuda|prestamo|credito", na=False),
        "Variacion"
    ].clip(lower=0).sum()

    # ======================================================
    # 5. Funciones auxiliares
    # ======================================================
    def add_origen(texto, monto):
        if abs(monto) > 1:
            origenes.append({
                "Concepto": texto,
                "Monto": round(float(monto), 2)
            })

    def add_aplic(texto, monto):
        if abs(monto) > 1:
            aplicaciones.append({
                "Concepto": texto,
                "Monto": round(float(monto), 2)
            })

    # ======================================================
    # 6. Clasificación de movimientos
    # ======================================================
    for _, row in merged.iterrows():

        cuenta = row[col_cuenta_act]
        variacion = row["Variacion"]
        tipo = row["Tipo_norm"]
        cuenta_norm = row["Cuenta_norm"]

        # ========== ACTIVO ==========
        if "activo" in tipo or any(x in cuenta_norm for x in [
            "caja", "banco", "invent", "cobrar",
            "fijo", "maquinaria", "equipo",
            "vehiculo", "terreno", "mobiliario"
        ]):

            # Si es activo fijo, evitar duplicarlo si fue financiado
            if any(x in cuenta_norm for x in ["fijo", "maquinaria", "equipo"]):

                if variacion > 0:

                    parte_efectivo = max(0, variacion - financiamiento_total)

                    if parte_efectivo > 0:
                        add_aplic(f"Aumento de activo fijo (parte pagada): {cuenta}", parte_efectivo)

                elif variacion < 0:
                    add_origen(f"Disminución de activo fijo: {cuenta}", abs(variacion))

            else:
                if variacion > 0:
                    add_aplic(f"Aumento de {cuenta}", variacion)
                elif variacion < 0:
                    add_origen(f"Disminución de {cuenta}", abs(variacion))

        # ========== PASIVO ==========
        elif "pasivo" in tipo or any(x in cuenta_norm for x in [
            "proveedor", "deuda", "pagar", "prestamo", "credito"
        ]):

            if variacion > 0:
                add_origen(f"Financiamiento obtenido: {cuenta}", variacion)
            elif variacion < 0:
                add_aplic(f"Pago de {cuenta}", abs(variacion))

        # ========== PATRIMONIO ==========
        elif "patrimonio" in tipo or any(x in cuenta_norm for x in [
            "capital", "reserva", "utilidad retenida", "resultado acumulado"
        ]):

            if variacion > 0:
                add_origen(f"Movimiento Patrimonial: {cuenta}", variacion)
            elif variacion < 0:
                add_aplic(f"Movimiento Patrimonial: {cuenta}", abs(variacion))

    # ======================================================
    # 7. Estado de Resultados
    # ======================================================
    if results_act is not None:

        res = results_act.copy()
        col_map_res = {c.lower(): c for c in res.columns}

        col_cuenta_res = col_map_res.get("cuenta")
        col_monto_res  = col_map_res.get("monto")

        res["Cuenta_norm"] = res[col_cuenta_res].astype(str).apply(norm_account_name)
        res["Monto"] = pd.to_numeric(res[col_monto_res], errors="coerce").fillna(0)

        utilidad = res.loc[
            res["Cuenta_norm"].str.contains(
                "utilidad neta|resultado del ejercicio|ganancia|excedente", na=False),
            "Monto"
        ].sum()

        depreciacion = res.loc[
            res["Cuenta_norm"].str.contains("depreci|amortiz", na=False),
            "Monto"
        ].sum()

        if abs(utilidad) > 1:
            add_origen("Utilidad Neta del Ejercicio", utilidad)

        if abs(depreciacion) > 1:
            add_origen("Depreciación y Amortización", depreciacion)

    # ======================================================
    # 8. Resultados finales
    # ======================================================
    df_origen = pd.DataFrame(origenes)
    df_aplic  = pd.DataFrame(aplicaciones)

    total_origen = df_origen["Monto"].sum() if not df_origen.empty else 0
    total_aplic  = df_aplic["Monto"].sum() if not df_aplic.empty else 0

    df_resumen = pd.DataFrame({
        "Tipo": ["Total Orígenes", "Total Aplicaciones", "Diferencia"],
        "Monto": [
            round(total_origen, 2),
            round(total_aplic, 2),
            round(total_origen - total_aplic, 2)
        ]
    })

    return df_origen, df_aplic, df_resumen
# ----------------------------
# OpenAI: interpretación IA 
# ----------------------------
from google import genai 

def generate_interpretation_gemini(summary: str) -> str:
    """
    Genera una interpretación financiera a partir de un resumen de datos usando Gemini.
    Ahora utiliza st.secrets para leer la clave API.
    """
    try:
        # 1. OBTENER LA CLAVE API usando st.secrets
        # Intentamos leer la clave del archivo .streamlit/secrets.toml
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            return "Error: La clave 'GEMINI_API_KEY' no se encontró en el archivo .streamlit/secrets.toml."
        
        # 2. INICIALIZAR EL CLIENTE CON LA CLAVE
        client = genai.Client(api_key=api_key) 
        
        # Define un prompt de sistema para dar contexto al modelo
        system_prompt = (
            "Eres un analista financiero experto. Tu tarea es analizar el siguiente resumen de "
            "estados financieros, razones financieras (ratios) y métricas (KPIs). "
            "Genera un informe no tan extenso que sea conciso y profesional, destacando los puntos clave "
            "de la salud financiera, la rentabilidad y la liquidez."
        )

        # Genera la respuesta
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # Modelo rápido y eficiente
            contents=[system_prompt, f"Resumen financiero para analizar:\n\n{summary}"],
        )
        
        return response.text

    except Exception as e:
        # Este 'except' capturará cualquier otro error (como problemas de red o la API)
        return f"Error al generar la interpretación con Gemini: {e}"
# ----------------------------
# AGGrid helper
# ----------------------------
import uuid

def aggrid_dark(df: pd.DataFrame, height: int = 300, key: str = None):
    
    if key is None:
        key = f"aggrid_{uuid.uuid4().hex}"

    # 🔥 Estilos visuales mejorados
    st.markdown("""
    <style>
    /* Fila seleccionada */
    .ag-theme-alpine-dark .ag-row-selected {
        background-color: rgba(154, 123, 255, 0.35) !important;
        border-left: 4px solid #9A7BFF !important;
    }

    /* Hover visible */
    .ag-theme-alpine-dark .ag-row:hover {
        background-color: rgba(45, 212, 191, 0.15) !important;
    }

    /* Celda activa */
    .ag-theme-alpine-dark .ag-cell-focus {
        border: 1px solid #2DD4BF !important;
    }

    /* Quitar ceguera en celdas */
    .ag-theme-alpine-dark .ag-cell {
        color: #e5e7eb !important;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_default_column(
        enableRowGroup=True,
        enableValue=True,
        sortable=True,
        filter=True,
        resizable=True
    )

    gb.configure_selection(
        selection_mode="single",
        use_checkbox=False
    )

    gb.configure_grid_options(
        domLayout='normal',
        rowHeight=32
    )

    gridOptions = gb.build()

    return AgGrid(
        df,
        gridOptions=gridOptions,
        theme='alpine-dark',   # 🎯 más personalizable que "dark"
        height=height,
        fit_columns_on_grid_load=True,
        key=key
    )
#------------------
#SELECCIONAR ESTADO
#-----------------
def seleccionar_estado():
    import pandas as pd
    import streamlit as st
    from db import get_connection, obtener_empresas

    # =========================
    # 🎨 ESTILO OPTIMIZADO
    # =========================
    st.markdown("""
    <style>

    .fs-card {
        background: #020617;
        border:1px solid #1e293b;
        border-radius:16px;
        padding:20px;
        margin-bottom:20px;
        box-shadow: 0 0 12px rgba(0,0,0,0.5);
    }

    .fs-title {
        color:#38BDF8;
        font-size:22px;
        font-weight:bold;
    }

    .fs-subtitle {
        color:#94a3b8;
        font-size:14px;
        margin-bottom:12px;
    }

    .summary-line {
        display:flex;
        justify-content:space-between;
        padding:10px 0;
        border-bottom:1px solid #1e293b;
        font-size:14px;
    }

    .summary-label {
        color:#94a3b8;
    }

    .summary-value {
        color:#38BDF8;
        font-weight:bold;
    }

    .fin-table {
        width: 100%;
        border-collapse: collapse;
        background:#020617;
        color:#e2e8f0;
        font-size:13px;
    }

    .fin-table th {
        background:#0f172a;
        padding:8px;
        border:1px solid #1e293b;
        color:#38BDF8;
        text-align:center;
    }

    .fin-table td {
        padding:8px;
        border:1px solid #1e293b;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #2563eb, #38BDF8);
        color:white;
        font-weight:bold;
        border:none;
        padding:0.6rem 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HEADER
    # =========================
    st.markdown("""
    <div class="fs-card">
        <div class="fs-title">📂 Gestión de Estados Financieros</div>
        <div class="fs-subtitle">Carga y visualización desde base de datos</div>
    </div>
    """, unsafe_allow_html=True)

    empresas = obtener_empresas()

    if not empresas:
        st.warning("⚠ No hay empresas registradas.")
        return

    empresa_id, empresa_nombre = st.selectbox(
        "🏢 Empresa",
        empresas,
        format_func=lambda x: x[1]
    )

    periodicidad = st.radio("📅 Periodicidad", ["Anual", "Mensual"], horizontal=True)
    periodicidad_db = periodicidad.lower()

    conn = get_connection()

    df_anios = pd.read_sql_query("""
        SELECT DISTINCT año 
        FROM estados_financieros 
        WHERE empresa_id = ?
        AND periodicidad = ?
        ORDER BY año ASC
    """, conn, params=(empresa_id, periodicidad_db))

    conn.close()

    if df_anios.empty:
        st.warning("Esta empresa no tiene datos financieros.")
        return

    anios = df_anios["año"].tolist()

    anios_seleccionados = st.multiselect(
        "📅 Años disponibles",
        anios,
        default=[max(anios)]
    )

    mes_num = None

    if periodicidad_db == "mensual":
        conn = get_connection()
        df_meses = pd.read_sql_query("""
            SELECT DISTINCT mes
            FROM estados_financieros
            WHERE empresa_id = ?
            AND periodicidad = 'mensual'
            AND mes IS NOT NULL
        """, conn, params=(empresa_id,))
        conn.close()

        meses_map = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                     7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}

        mes_dict = {meses_map[m]: m for m in df_meses["mes"].tolist()}
        mes_nombre = st.selectbox("📆 Mes", list(mes_dict.keys()))
        mes_num = mes_dict[mes_nombre]

    # =========================
    # CARGAR ESTADOS
    # =========================
    def cargar_estado(tipo_estado, anio):
        conn = get_connection()
        query = """
        SELECT cuenta, monto
        FROM estados_financieros
        WHERE empresa_id = ?
        AND tipo_estado = ?
        AND año = ?
        AND periodicidad = ?
        """
        params = [empresa_id, tipo_estado, anio, periodicidad_db]

        if periodicidad_db == "mensual":
            query += " AND mes = ?"
            params.append(mes_num)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # =========================
    # BOTÓN CARGAR
    # =========================
    if st.button("📂 Cargar Estados Financieros"):
        balances, resultados = [], []

        for anio in anios_seleccionados:
            bg_df = cargar_estado("BG", anio)
            er_df = cargar_estado("ER", anio)

            if not bg_df.empty:
                bg_df["Año"] = anio
                balances.append(bg_df)

            if not er_df.empty:
                er_df["Año"] = anio
                resultados.append(er_df)

        st.session_state.estado_seleccionado = {
            "empresa": empresa_nombre,
            "balances": balances,
            "resultados": resultados,
            "periodicidad": periodicidad_db,
            "mes": mes_num,
            "años": anios_seleccionados
        }

        st.success("✅ Estados financieros cargados correctamente")

    # =========================
    # RESUMEN + VISUAL LIMPIO
    # =========================
    if st.session_state.get("estado_seleccionado"):

        datos = st.session_state.estado_seleccionado

        st.markdown("""<div class="fs-card">""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-line">
            <div class="summary-label">Empresa</div>
            <div class="summary-value">{datos["empresa"]}</div>
        </div>
        <div class="summary-line">
            <div class="summary-label">Balances cargados</div>
            <div class="summary-value">{len(datos["balances"])}</div>
        </div>
        <div class="summary-line">
            <div class="summary-label">Resultados cargados</div>
            <div class="summary-value">{len(datos["resultados"])}</div>
        </div>
        <div class="summary-line">
            <div class="summary-label">Periodicidad</div>
            <div class="summary-value">{datos["periodicidad"].capitalize()}</div>
        </div>
        <div class="summary-line">
            <div class="summary-label">Años</div>
            <div class="summary-value">{", ".join(map(str, datos["años"]))}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # SELECTOR DE AÑO PARA PREVIEW
        # =========================
        anio_preview = st.selectbox(
            "Selecciona año para preview",
            datos["años"]
        )

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("📘 Balance General", expanded=True):
                for bg in datos["balances"]:
                    if bg["Año"].iloc[0] == anio_preview:
                        st.markdown(bg.to_html(index=False, classes="fin-table"), unsafe_allow_html=True)

        with col2:
            with st.expander("📗 Estado de Resultados", expanded=True):
                for er in datos["resultados"]:
                    if er["Año"].iloc[0] == anio_preview:
                        st.markdown(er.to_html(index=False, classes="fin-table"), unsafe_allow_html=True)
# ---------------------------
# Función robusta para lectura
# ---------------------------
def process_upload_list(uploaded):
    files = []
    if uploaded:
        for f in uploaded:
            try:
                if f.name.lower().endswith('.csv'):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)

                # Normalizar columnas
                cols = {c.lower().strip(): c for c in df.columns}

                cuenta_col = None
                monto_col = None

                for k, orig in cols.items():
                    if k in ["cuenta", "descripcion", "nombre", "concepto", "detalle"]:
                        cuenta_col = orig
                    if k in ["monto", "saldo", "valor", "importe", "total"]:
                        monto_col = orig

                if 'Cuenta' in df.columns and 'Monto' in df.columns:
                    cuenta_col = 'Cuenta'
                    monto_col = 'Monto'

                if cuenta_col and monto_col:
                    df2 = df[[cuenta_col, monto_col]].copy()
                    df2.columns = ["Cuenta", "Monto"]
                    df2['Cuenta'] = df2['Cuenta'].astype(str).str.strip()
                    df2['Monto'] = to_numeric_series(df2['Monto'])
                    files.append((f.name.replace('.xlsx','').replace('.csv',''), df2))
                else:
                    st.warning(f"⚠ Archivo {f.name} omitido: requiere columnas 'Cuenta' y 'Monto'.")

            except Exception as e:
                st.error(f"Error al procesar {f.name}: {e}")

    files.sort(key=lambda x: x[0])
    return files

# ----------------------------
# Export helpers: CSV / Excel / HTML
# ----------------------------
def df_to_excel_bytes(dfs: dict) -> bytes:
    # dfs: {"sheetname": dataframe}
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in dfs.items():
            try:
                df.to_excel(writer, sheet_name=name[:31], index=False)
            except Exception:
                # attempt to stringify problematic columns
                df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def df_to_html_report(title: str, tables: dict) -> str:
    # tables: {"title": df}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; background: #0b1020; color: #E6EDF3; padding: 20px; }}
          .card {{ background: #0f1724; border-radius: 8px; padding: 12px; margin-bottom: 18px; border:1px solid #1e293b }}
          h1 {{ color: {ACCENT_TITLE}; }}
          table {{ width:100%; border-collapse: collapse; margin-top:8px; }}
          th, td {{ padding:6px 8px; border:1px solid #333; font-size:12px; }}
          th {{ background: #111827; color: #E6EDF3; text-align:left; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <div class="small-muted">Generado: {now}</div>
    """
    for tname, df in tables.items():
        html += f"<div class='card'><h2 style='color:{ACCENT_SUBHEADER};'>{tname}</h2>"
        # safe to_html
        try:
            html += df.to_html(index=False, classes='table', border=0)
        except Exception:
            html += pd.DataFrame(df).to_html(index=False, classes='table', border=0)
        html += "</div>"
    html += "</body></html>"
    return html

def make_download_button_bytes(content_bytes: bytes, filename: str, mime: str):
    b64 = base64.b64encode(content_bytes).decode()
    href = f"data:{mime};base64,{b64}"
    return href


import streamlit as st



# ==================== ESTILOS LIMPIOS ====================
st.sidebar.markdown("""
<style>

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #020617, #0f172a);
}

/* Header */
.sidebar-container {
    padding: 20px 15px;
    text-align: center;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 10px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
    color: #38BDF8;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #94a3b8;
}

/* Badge base (modo) */
.badge-style {
    width: 100%;
    text-align: center;
    padding: 10px;
    margin: 10px 0;
    border-radius: 14px;
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.15);
    color: #8fe9ff;
    font-size: 13px;
    font-weight: bold;
}

/* Botones tipo badge */
div.stButton > button {
    all: unset;
    width: 100%;
    cursor: pointer;
    display: block;
    text-align: center;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 14px;
    background: rgba(56,189,248,0.04);
    border: 1px solid #1e293b;
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.25s ease;
}

/* Hover */
div.stButton > button:hover {
    background: rgba(56,189,248,0.12);
    border: 1px solid #38BDF8;
    box-shadow: 0 0 10px rgba(56,189,248,0.3);
}

/* Botón activo */
.nav-active {
    background: linear-gradient(135deg, #38BDF8, #2563eb) !important;
    color: #020617 !important;
    font-weight: bold !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(56,189,248,0.7);
}

/* Sesión activa */
.session-box {
    margin-top: 15px;
    padding: 12px;
    border-radius: 14px;
    background: rgba(56,189,248,0.03);
    border: 1px solid #1e293b;
    text-align: center;
    font-size: 13px;
    color:#e2e8f0;
}

/* Footer */
.footer-mini {
    text-align:center;
    font-size:11px;
    color:#64748b;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ==================== CABECERA ====================
st.sidebar.markdown("""
<div class="sidebar-container">
    <div class="sidebar-title">FinanSys</div>
    <div class="sidebar-subtitle">Sistema de Análisis Financiero</div>
</div>
""", unsafe_allow_html=True)

# ==================== MODO DE DATOS ====================
if "modo_datos" not in st.session_state:
    st.session_state.modo_datos = None

modo = st.session_state.modo_datos

modo_txt = {
    "usar": "📂 Estados guardados",
    "cargar": "📤 Subir estados",
    "crear": "📝 Crear manual"
}.get(modo, "🏁 Modo inicio")

# Badge de modo
st.sidebar.markdown(f"""
<div class="badge-style">{modo_txt}</div>
""", unsafe_allow_html=True)

# ==================== SECCIONES ====================
if modo == "usar":
    sections = [
        "Inicio",
        "Usar estados guardados",
        "Análisis vertical",
        "Análisis horizontal",
        "Razones financieras",
        "DuPont",
        "Flujo de efectivo",
        "Origen y Aplicación de Fondos",
        "Interpretación"
    ]
elif modo == "cargar":
    sections = ["Inicio", "Cargar archivos"]
elif modo == "crear":
    sections = ["Inicio", "Crear BG / ER"]
else:
    sections = ["Inicio"]

# Inicializar sección
if "section" not in st.session_state:
    st.session_state.section = sections[0]

# ==================== NAVEGACIÓN ====================
for sec in sections:

    clicked = st.sidebar.button(sec, key=f"btn_{sec}")

    # Aplicar clase activa al botón correspondiente
    st.sidebar.markdown(f"""
    <script>
    const buttons = window.parent.document.querySelectorAll("button");
    buttons.forEach(btn => {{
        if(btn.innerText.trim() === "{sec}") {{
            btn.classList.remove("nav-active");
            if("{st.session_state.section}" === "{sec}") {{
                btn.classList.add("nav-active");
            }}
        }}
    }});
    </script>
    """, unsafe_allow_html=True)

    if clicked:
        st.session_state.section = sec
        st.rerun()

# ==================== SESIÓN ACTIVA ====================
if st.session_state.get("estado_seleccionado"):
    datos = st.session_state.estado_seleccionado
    empresa = datos.get("empresa", "N/A")
    n_bg = len(datos.get("balances", []))
    n_er = len(datos.get("resultados", []))

    st.sidebar.markdown(f"""
    <div class="session-box">
        <b>Sesión activa</b><br><br>
        🏢 {empresa}<br>
        📘 BG: {n_bg}<br>
        📗 ER: {n_er}
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.sidebar.markdown("""
<div class="footer-mini">
    FinanSys © 2025  
</div>
""", unsafe_allow_html=True)
# ===============================
# INICIO PREMIUM FINANSYS
# ===============================

section = st.session_state.get("section", "Inicio")

if section == "Inicio":

    import base64

    # ===================== CSS PREMIUM =====================
    st.markdown("""
    <style>

    @keyframes neonMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-container {
        position: relative;
        padding: 60px 40px;
        border-radius: 25px;
        background: linear-gradient(135deg, #020617, #0f172a);
        border: 1px solid rgba(56,189,248,0.15);
        box-shadow: 0 0 55px rgba(0,255,255,0.15);
        overflow: hidden;
        text-align: center;
    }

    .neon-bar {
        position: absolute;
        top: 0;
        left: 0;
        height: 4px;
        width: 100%;
        background: linear-gradient(270deg, #00f6ff, #2563eb, #00f6ff);
        background-size: 400% 400%;
        animation: neonMove 6s linear infinite;
        box-shadow: 0 0 20px rgba(0,255,255,0.7);
    }

    .hero-content {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .logo-animated {
        width: 180px;
        margin-bottom: 18px;
        filter: drop-shadow(0px 0px 15px rgba(0,255,255,0.7));
        animation: floatLogo 4s ease-in-out infinite;
    }

    @keyframes floatLogo {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        color: white;
        margin-bottom: 6px;
    }

    .hero-title span {
        color: #38BDF8;
        text-shadow: 0 0 15px rgba(56,189,248,0.8);
    }

    .hero-subtitle {
        font-size: 17px;
        color: #94a3b8;
        margin-bottom: 45px;
    }

    .mode-buttons {
        margin-top: 30px;
        width: 85%;
        max-width: 700px;
    }

    .mode-btn {
        padding: 16px;
        border-radius: 16px;
        background: linear-gradient(145deg, #020617, #0f172a);
        border: 1px solid rgba(56,189,248,0.2);
        color: #e2e8f0;
        font-weight: bold;
        font-size: 15px;
        width: 100%;
        margin: 8px 0;
        transition: all 0.3s ease;
    }

    .mode-btn:hover {
        background: linear-gradient(145deg, #020617, #1e293b);
        transform: translateY(-4px);
        box-shadow: 0 0 18px rgba(56,189,248,0.5);
    }

    </style>
    """, unsafe_allow_html=True)

    # ===================== CONTENEDOR =====================
    st.markdown("<div class='hero-container'>", unsafe_allow_html=True)
    st.markdown("<div class='neon-bar'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-content'>", unsafe_allow_html=True)

    # LOGO CENTRADO
    st.markdown(
        f"<img src='data:image/png;base64,{base64.b64encode(logo_bytes).decode()}' class='logo-animated'>",
        unsafe_allow_html=True
    )

    # TITULOS
    st.markdown("""
        <div class="hero-title">
            Bienvenido a <span>FinanSys</span>
        </div>
        <div class="hero-subtitle">
            Plataforma avanzada de análisis y diagnóstico financiero
        </div>
    """, unsafe_allow_html=True)

    # ===================== BOTONES =====================
    st.markdown("<div class='mode-buttons'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📂 Usar estados guardados", use_container_width=True):
            st.session_state.modo_datos = "usar"
            st.session_state.section = "Usar estados guardados"
            st.rerun()

    with col2:
        if st.button("📤 Subir estados financieros", use_container_width=True):
            st.session_state.modo_datos = "cargar"
            st.session_state.section = "Cargar archivos"
            st.rerun()

    with col3:
        if st.button("📝 Crear estados manualmente", use_container_width=True):
            st.session_state.modo_datos = "crear"
            st.session_state.section = "Crear BG / ER"
            st.rerun()

    st.markdown("</div></div></div>", unsafe_allow_html=True)

elif section == "Crear BG / ER":

  
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Constructor de Balance General / Estado de Resultado</h3>", unsafe_allow_html=True)

    # =======================
    # Empresa
    # =======================
    st.markdown("### 🏢 Empresa")
    empresa_nombre = st.text_input("Nombre de la empresa")

    # =======================
    # Tipo / Periodicidad
    # =======================
    tipo = st.radio(
        "¿Qué deseas crear?",
        ["Balance General", "Estado de Resultado"],
        key="tipo_estado"
    )

    periodicidad = st.radio(
        "Periodicidad",
        ["Anual", "Mensual"],
        key="periodicidad_estado"
    )

    año = st.number_input(
        "Año:",
        min_value=1900,
        max_value=2100,
        value=2024,
        key="anio_estado"
    )

    if periodicidad == "Mensual":
        meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                 "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
        mes_nombre = st.selectbox("Mes:", meses, key="mes_estado")
        mes_num = meses.index(mes_nombre) + 1
    else:
        mes_num = None

    # =======================
    # Clasificaciones
    # =======================
    CLASIFICACIONES_BG = {
        "Activo Corriente": ["Caja", "Bancos", "Clientes", "Inventarios"],
        "Activo No Corriente": ["Activo fijo", "Propiedad, planta y equipo", "Intangibles"],
        "Pasivo Corriente": ["Proveedores", "Cuentas por pagar", "Deudas CP"],
        "Pasivo No Corriente": ["Deuda LP", "Obligaciones financieras"],
        "Capital Contable": ["Capital social", "Utilidades retenidas"]
    }

    CLASIFICACIONES_ER = {
        "Ingresos": ["Ventas", "Otros ingresos"],
        "Costos": ["Costo de ventas"],
        "Gastos": ["Gastos administrativos", "Gastos financieros", "Gastos operativos"]
    }

    clasificaciones = CLASIFICACIONES_BG if tipo == "Balance General" else CLASIFICACIONES_ER

    # =======================
    # Inicializaciones
    # =======================
    if "df_edit" not in st.session_state:
        st.session_state.df_edit = pd.DataFrame({"Cuenta": [], "Monto": []})

    if "confirm_vaciar" not in st.session_state:
        st.session_state.confirm_vaciar = False

    # =======================
    # Selección de cuentas
    # =======================
    clasificacion = st.selectbox(
        "Clasificación:",
        list(clasificaciones.keys())
    )

    cuenta_predet = st.selectbox(
        "Cuenta predeterminada:",
        clasificaciones[clasificacion]
    )

    nueva_cuenta = st.text_input("Cuenta nueva (opcional):")

    monto_input = st.text_input(
        "Monto:",
        placeholder="Ej: 1000, 1.000,50, -500, $2,000"
    )

    # =======================
    # Agregar cuenta
    # =======================
    if st.button("➕ Agregar cuenta"):

        cuenta_final = nueva_cuenta.strip() if nueva_cuenta.strip() else cuenta_predet
        monto_val = parse_number(monto_input)

        if cuenta_final == "":
            st.error("❌ Debes ingresar una cuenta.")
        elif np.isnan(monto_val):
            st.error("❌ Monto inválido.")
        else:
            st.session_state.df_edit.loc[len(st.session_state.df_edit)] = [
                cuenta_final,
                monto_val
            ]
            st.success(f"✅ Cuenta '{cuenta_final}' agregada")
            st.rerun()

    # =======================
    # Tabla preview
    # =======================
    st.markdown("### 📊 Estado actual")
    st.dataframe(st.session_state.df_edit, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Vaciar tabla"):
            st.session_state.confirm_vaciar = True

    with col2:
        guardar_pressed = st.button("💾 Guardar estado")

    # =======================
    # Confirmación vaciado
    # =======================
    if st.session_state.confirm_vaciar:
        st.warning("¿Seguro que deseas borrar todo?")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("✅ Sí, borrar"):
                st.session_state.df_edit = pd.DataFrame({"Cuenta": [], "Monto": []})
                st.session_state.confirm_vaciar = False
                st.rerun()

        with c2:
            if st.button("❌ Cancelar"):
                st.session_state.confirm_vaciar = False

    # =======================
    # GUARDAR EN BD
    # =======================
    if guardar_pressed:

        if not empresa_nombre.strip():
            st.error("❌ Debes indicar la empresa.")
            st.stop()

        if st.session_state.df_edit.empty:
            st.error("❌ La tabla está vacía.")
            st.stop()

        df_final = st.session_state.df_edit.copy()
        df_final["Monto"] = to_numeric_series(df_final["Monto"])

        if df_final["Monto"].isna().any():
            st.error("❌ Hay montos inválidos.")
            st.stop()

        # ==================================================
        # ✅ GENERAR TOTALES AUTOMÁTICOS
        # ==================================================
        df_temp = df_final.copy()
        df_temp["Cuenta_norm"] = df_temp["Cuenta"].astype(str).str.lower().str.strip()

        # Eliminar totales anteriores
        df_temp = df_temp[
            ~df_temp["Cuenta_norm"].str.contains(
                "total|utilidad neta|resultado del ejercicio",
                case=False,
                na=False
            )
        ]

        if tipo == "Balance General":

            total_activo = df_temp[df_temp["Cuenta_norm"].str.contains("activo")]["Monto"].sum()
            total_pasivo = df_temp[df_temp["Cuenta_norm"].str.contains("pasivo")]["Monto"].sum()
            total_capital = df_temp[df_temp["Cuenta_norm"].str.contains("capital")]["Monto"].sum()

            nuevos_totales = pd.DataFrame([
                {"Cuenta": "Total Activo", "Monto": total_activo},
                {"Cuenta": "Total Pasivo", "Monto": total_pasivo},
                {"Cuenta": "Total Capital Contable", "Monto": total_capital},
                {"Cuenta": "Total Pasivo + Capital", "Monto": total_pasivo + total_capital}
            ])

        else:  # Estado de Resultado

            ingresos = df_temp[df_temp["Cuenta_norm"].str.contains("venta|ingreso")]["Monto"].sum()
            costos = df_temp[df_temp["Cuenta_norm"].str.contains("costo")]["Monto"].sum()
            gastos = df_temp[df_temp["Cuenta_norm"].str.contains("gasto")]["Monto"].sum()

            utilidad_neta = ingresos - costos - gastos

            nuevos_totales = pd.DataFrame([
                {"Cuenta": "Total Ingresos", "Monto": ingresos},
                {"Cuenta": "Total Costos", "Monto": costos},
                {"Cuenta": "Total Gastos", "Monto": gastos},
                {"Cuenta": "Utilidad Neta", "Monto": utilidad_neta}
            ])

        df_final = pd.concat([
            df_temp[["Cuenta", "Monto"]],
            nuevos_totales
        ], ignore_index=True)

        # ==================================================
        # Buscar o crear empresa
        # ==================================================
        empresas = obtener_empresas()
        empresas_dict = {e[1]: e[0] for e in empresas}

        if empresa_nombre not in empresas_dict:
            crear_empresa(
                empresa_nombre,
                sector="No definido",
                fecha_registro=str(datetime.now().date())
            )
            empresas = obtener_empresas()
            empresas_dict = {e[1]: e[0] for e in empresas}

        empresa_id = empresas_dict[empresa_nombre]

        # Tipo estado BD
        tipo_estado = "BG" if tipo == "Balance General" else "ER"

        # Periodicidad
        if periodicidad == "Anual":
            periodicidad_bd = "anual"
            mes_bd = None
        else:
            periodicidad_bd = "mensual"
            mes_bd = mes_num

        # ==================================================
        # GUARDAR EN LA BD
        # ==================================================
        for _, row in df_final.iterrows():
            guardar_estado(
                empresa_id=empresa_id,
                tipo_estado=tipo_estado,
                periodicidad=periodicidad_bd,
                año=int(año),
                mes=mes_bd,
                cuenta=row["Cuenta"],
                monto=float(row["Monto"])
            )

        st.success("✅ Estado financiero guardado con totales automáticos")

        # Limpiar tabla
        st.session_state.df_edit = pd.DataFrame({"Cuenta": [], "Monto": []})

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

if section == "Usar estados guardados":
    seleccionar_estado()
    st.markdown("</div>", unsafe_allow_html=True)





# ----------------------------
# Sección: Cargar Archivos (SOLO SQLite)
# ----------------------------
elif section == "Cargar archivos":

    from db import crear_empresa, obtener_empresas, guardar_estado
    from datetime import datetime

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>📂 Carga de Estados Financieros (Base de Datos)</h3>", unsafe_allow_html=True)

    # =====================
    # EMPRESA
    # =====================
    empresa = st.text_input("🏢 Nombre de la empresa")

    if not empresa.strip():
        st.warning("Debes ingresar un nombre de empresa.")
        st.stop()

    # =====================
    # CONFIGURACIÓN
    # =====================
    periodicidad = st.radio("Periodicidad", ["Anual", "Mensual"])
    tipo_estado = st.radio("Tipo de estado", ["BG", "ER"])

    año = st.number_input(
        "Año",
        min_value=1900,
        max_value=2100,
        value=2024,
        step=1
    )

    mes = None
    mes_num = None

    if periodicidad == "Mensual":
        meses = [
            "Enero","Febrero","Marzo","Abril","Mayo","Junio",
            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
        ]
        mes = st.selectbox("Mes", meses)
        mes_num = meses.index(mes) + 1

    # =====================
    # SUBIDA DE ARCHIVO
    # =====================
    uploaded_file = st.file_uploader(
        f"Sube el archivo {tipo_estado} de {empresa}",
        type=["xlsx", "csv"]
    )

    if uploaded_file:

        if st.button("💾 Guardar en base de datos"):

            # Leer archivo
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Validar columnas mínimas
            if df.shape[1] < 2:
                st.error("❌ El archivo debe tener al menos 2 columnas: Cuenta | Monto")
                st.stop()

            df = df.iloc[:, :2]
            df.columns = ["Cuenta", "Monto"]

            # Limpiar datos
            df["Cuenta"] = df["Cuenta"].astype(str).str.strip()
            df["Monto"] = to_numeric_series(df["Monto"])

            if df["Monto"].isna().any():
                st.error("❌ Hay valores no numéricos en la columna Monto")
                st.stop()

            # =====================
            # CREAR EMPRESA SI NO EXISTE
            # =====================
            empresas = obtener_empresas()
            nombres = [e[1] for e in empresas]

            if empresa not in nombres:
                crear_empresa(
                    empresa,
                    sector="General",
                    fecha_registro=str(datetime.now().date())
                )
                empresas = obtener_empresas()

            empresa_id = [e[0] for e in empresas if e[1] == empresa][0]

            # =====================
            # VALIDAR QUE NO SE REPITA
            # =====================
            conn = db.get_connection()
            c = conn.cursor()

            if periodicidad == "Anual":
                c.execute("""
                    SELECT COUNT(*) 
                    FROM estados_financieros
                    WHERE empresa_id = ?
                    AND tipo_estado = ?
                    AND año = ?
                    AND mes IS NULL
                """, (empresa_id, tipo_estado, año))
            else:
                c.execute("""
                    SELECT COUNT(*) 
                    FROM estados_financieros
                    WHERE empresa_id = ?
                    AND tipo_estado = ?
                    AND año = ?
                    AND mes = ?
                """, (empresa_id, tipo_estado, año, mes_num))

            existe = c.fetchone()[0]
            conn.close()

            if existe > 0:
                st.error("❌ Este estado ya existe en la base de datos.")
                st.stop()

            # =====================
            # GUARDAR EN SQLITE
            # =====================
            for _, row in df.iterrows():
                guardar_estado(
                    empresa_id=empresa_id,
                    tipo_estado=tipo_estado,
                    periodicidad=periodicidad.lower(),
                    año=int(año),
                    mes=None if periodicidad == "Anual" else mes_num,
                    cuenta=row["Cuenta"],
                    monto=float(row["Monto"])
                )

            st.success(f"✅ {tipo_estado} de {empresa} guardado correctamente en la base de datos.")

    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Sección: Análisis vertical
# ----------------------------
elif section == "Análisis vertical":
    import plotly.express as px
    import streamlit as st

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0f172a,#1e293b);
        padding:20px;
        border-radius:18px;
        box-shadow:0 10px 35px rgba(0,0,0,0.7);
    ">
    <h2 style='color:#38bdf8'>📊 Análisis Vertical Financiero</h2>
    <p style='color:#94a3b8'>Análisis estructural de Estados Financieros seleccionados</p>
    </div>
    """, unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero debes seleccionar estados en: Usar estados guardados.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])

    def normalizar_df(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        return df

    # ================= BALANCES =====================
    st.markdown("---")
    st.subheader("📘 Balances Generales")

    if not balances:
        st.info("No hay balances cargados.")
    else:
        tabs = st.tabs([f"📅 {df['Año'].iloc[0]}" for df in balances])

        for tab, bg in zip(tabs, balances):

            with tab:
                anio = bg["Año"].iloc[0]
                bg = normalizar_df(bg)
                df_v = vertical_analysis(f"BG {anio}", bg, "BG")

                if df_v is None:
                    st.warning(f"⚠ No se pudo procesar Balance {anio}")
                    continue

                col1, col2 = st.columns([1.3, 1])

                with col1:
                    st.markdown(f"<h4 style='color:#93c5fd'>🧾 Estructura del Balance – {anio}</h4>", unsafe_allow_html=True)

                    display_df = df_v[["Cuenta", "Monto", "Porcentaje"]].copy()
                    display_df["Monto"] = display_df["Monto"].map(lambda x: f"${x:,.2f}")
                    display_df["Porcentaje"] = display_df["Porcentaje"].map(lambda x: f"{x:.2f}%")

                    st.markdown(
                        display_df.to_html(
                            index=False,
                            classes="table table-dark table-bordered",
                            justify="center"
                        ),
                        unsafe_allow_html=True
                    )

                with col2:
                    df_plot = df_v[~df_v["Cuenta_norm"].str.contains(
                        r"\btotal\b|\bsubtotal\b|\bsuma\b", regex=True, case=False, na=False)]
                    df_plot = df_plot.sort_values("Monto", ascending=False).head(8)

                    fig = px.pie(
                        df_plot,
                        names="Cuenta",
                        values="Monto",
                        hole=0.55,
                        color_discrete_sequence=px.colors.sequential.Plasma
                    )

                    fig.update_traces(
                        textinfo="percent+label",
                        pull=[0.05] * len(df_plot),
                        hovertemplate="<b>%{label}</b><br>Monto: $%{value:,.0f}<br>%{percent}",
                        marker=dict(line=dict(color="white", width=1))
                    )

                    fig.update_layout(
                        title=f"Estructura {empresa} ({anio})",
                        template="plotly_dark",
                        height=420,
                        transition=dict(duration=600),
                        margin=dict(t=60, b=0, l=0, r=0)
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ================= RESULTADOS =====================
    st.markdown("---")
    st.subheader("📗 Estados de Resultados")

    if not resultados:
        st.info("No hay estados de resultados cargados.")
    else:
        tabs_er = st.tabs([f"📅 {df['Año'].iloc[0]}" for df in resultados])

        for tab, er in zip(tabs_er, resultados):

            with tab:
                anio = er["Año"].iloc[0]
                er = normalizar_df(er)
                df_v = vertical_analysis(f"ER {anio}", er, "ER")

                if df_v is None:
                    st.warning(f"⚠ No se pudo procesar Estado {anio}")
                    continue

                col1, col2 = st.columns([1.3, 1])

                with col1:
                    st.markdown(f"<h4 style='color:#93c5fd'>🧾 Estructura ER – {anio}</h4>", unsafe_allow_html=True)

                    display_df = df_v[["Cuenta", "Monto", "Porcentaje"]].copy()
                    display_df["Monto"] = display_df["Monto"].map(lambda x: f"${x:,.2f}")
                    display_df["Porcentaje"] = display_df["Porcentaje"].map(lambda x: f"{x:.2f}%")

                    st.markdown(
                        display_df.to_html(index=False, justify="center"),
                        unsafe_allow_html=True
                    )

                with col2:
                    df_graf = df_v[df_v["Cuenta"].str.contains(
                        "venta|ingreso|costo|utilidad|gasto",
                        case=False,
                        na=False
                    )].sort_values("Porcentaje", ascending=False).head(10)

                    fig = px.bar(
                        df_graf,
                        x="Porcentaje",
                        y="Cuenta",
                        orientation="h",
                        text=df_graf["Porcentaje"].map(lambda x: f"{x:.2f}%"),
                        color="Porcentaje",
                        color_continuous_scale="Viridis"
                    )

                    fig.update_layout(
                        title=f"Contribución Estructural - {empresa} ({anio})",
                        template="plotly_dark",
                        height=420,
                        transition=dict(duration=700),
                        margin=dict(t=60, b=0, l=0, r=0)
                    )

                    fig.update_traces(textposition="outside")

                    st.plotly_chart(fig, use_container_width=True)

elif section == "Análisis horizontal":
    import plotly.express as px
    import numpy as np
    import streamlit as st

    st.markdown("""
    <div style="background:linear-gradient(135deg,#020617,#0f172a);
         padding:20px;
         border-radius:20px;
         box-shadow: 0 10px 40px rgba(0,0,0,0.7);">
    <h2 style="color:#38bdf8">📈 Análisis Horizontal Financiero</h2>
    <p style="color:#94a3b8">Comparación financiera entre períodos seleccionados</p>
    </div>
    """, unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona balances en 'Usar estados guardados'.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances")

    if balances is None:
        bg = datos.get("bg")
        anio = datos.get("anio")
        balances = [(str(anio), bg)] if bg is not None else []

    balances_limpios = []

    for item in balances:
        if isinstance(item, tuple):
            balances_limpios.append(item)
        else:
            anio = str(item["Año"].iloc[0]) if "Año" in item.columns else "Periodo"
            balances_limpios.append((anio, item))

    balances = balances_limpios

    if len(balances) < 2:
        st.warning("⚠ Necesitas al menos 2 balances cargados.")
        st.stop()

    nombres = [nombre for nombre, _ in balances]

    col1, col2 = st.columns(2)
    with col1:
        periodo_inicial = st.selectbox("📅 Periodo inicial", nombres, 0)
    with col2:
        periodo_final = st.selectbox("📅 Periodo final", nombres, len(nombres)-1)

    idx_i = nombres.index(periodo_inicial)
    idx_f = nombres.index(periodo_final)

    df_inicial = balances[idx_i][1]
    df_final = balances[idx_f][1]

    try:
        df_h = horizontal_analysis(df_inicial, df_final, periodo_inicial, periodo_final)
        df_h.columns = df_h.columns.str.strip()

        col_prev = f"{periodo_inicial} (Monto)"
        col_act = f"{periodo_final} (Monto)"

        total_inicial = df_h[col_prev].sum()
        total_final = df_h[col_act].sum()
        variacion_total = total_final - total_inicial
        porcentaje_total = (variacion_total / total_inicial * 100) if total_inicial != 0 else 0

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(
                f"<h4 style='color:#93c5fd'>💰 Total Inicial</h4>"
                f"<h2>${total_inicial:,.0f}</h2>", 
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"<h4 style='color:#93c5fd'>💰 Total Final</h4>"
                f"<h2>${total_final:,.0f}</h2>", 
                unsafe_allow_html=True
            )
        with k3:
            color = "#22c55e" if porcentaje_total > 0 else "#ef4444"
            st.markdown(
                f"<h4 style='color:#93c5fd'>📈 Variación Total</h4>"
                f"<h2 style='color:{color}'>{porcentaje_total:.2f}%</h2>", 
                unsafe_allow_html=True
            )

        st.markdown("---")

        info_balance = validar_balance(df_final)

        if info_balance["cuadra"]:
            st.success("✅ El balance CUADRA correctamente.")
        else:
            st.error(f"❌ El balance NO cuadra. Diferencia: ${info_balance['diferencia']:,.2f}")

        # ============================
        # TABLA HORIZONTAL MODERNA
        # ============================

        st.markdown("""
        <div style="
            background: linear-gradient(135deg,#0f172a,#1e293b);
            padding:20px;
            border-radius:18px;
            margin-top:15px;
            box-shadow:0 10px 40px rgba(0,0,0,0.7);">

        <h3 style="color:#38bdf8;margin-bottom:15px;">
        📋 Detalle de análisis horizontal
        </h3>
        """, unsafe_allow_html=True)

        df_tabla = df_h.copy()

        for col in df_tabla.columns:
            if "Monto" in col or "Variación" in col:
                df_tabla[col] = df_tabla[col].apply(
                    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
                )

        if "Variación (%)" in df_tabla.columns:
            df_tabla["Variación (%)"] = df_tabla["Variación (%)"].apply(
                lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
            )

        st.markdown(
            df_tabla.to_html(
                index=False,
                justify="center",
                classes="table table-dark table-hover table-striped table-bordered"
            ),
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================
        # GRÁFICO 1
        # ============================

        graf_df = df_h.sort_values("Variación", key=lambda x: abs(x), ascending=False).head(10)
        graf_df["Tendencia"] = graf_df["Variación"].apply(
            lambda x: "Incremento" if x >= 0 else "Disminución"
        )

        fig = px.bar(
            graf_df,
            x="Variación",
            y="Cuenta",
            orientation="h",
            color="Tendencia",
            text=graf_df["Variación"].map(lambda x: f"${x:,.0f}"),
            color_discrete_map={
                "Incremento": "#22c55e",
                "Disminución": "#ef4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            transition=dict(duration=700),
            title=f"Impacto por cuenta ({periodo_inicial} → {periodo_final})"
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

        # ============================
        # GRÁFICO 2
        # ============================

        df_h["Categoría"] = df_h["Cuenta"].apply(clasificar_categoria)
        df_cat = df_h.groupby("Categoría")["Variación"].sum().reset_index()

        fig_cat = px.bar(
            df_cat,
            x="Variación",
            y="Categoría",
            orientation="h",
            color="Variación",
            color_continuous_scale="Turbo",
            template="plotly_dark",
            title="📊 Impacto por categoría financiera"
        )

        st.plotly_chart(fig_cat, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error en análisis horizontal: {str(e)}")
# ----------------------------
# Sección: Razones financieras
# ----------------------------
elif section == "Razones financieras":

    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go

    # ================== ESTILO DUPONT ==================
    st.markdown("""
    <style>
    @keyframes slideFade {
        from {opacity: 0; transform: translateY(15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .dup-card {
        background: linear-gradient(145deg, #020617, #0f172a);
        padding: 22px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.5);
        animation: slideFade 0.7s ease;
    }

    .dup-subcard {
        background: #020617;
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #1e293b;
        margin-bottom: 16px;
        animation: slideFade 1s ease;
    }

    .dup-title {
        color: #38BDF8;
        font-size: 26px;
        font-weight: bold;
    }

    .dup-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .dup-metric {
        background:#020617;
        padding:12px;
        border-radius:12px;
        border:1px solid #1e293b;
        text-align:center;
        margin-bottom:10px;
    }

    .dup-metric h4 {
        color:#94a3b8;
        font-size:14px;
        margin-bottom:5px;
    }

    .dup-metric h2 {
        color:#38BDF8;
        font-size:22px;
        font-weight:bold;
    }

    table {
        width:100%;
        border-collapse: collapse;
        background:#020617;
        color:#e2e8f0;
        font-size:14px;
    }

    table th {
        background:#0f172a;
        padding:8px;
        border:1px solid #1e293b;
        color:#38BDF8;
        text-align:center;
    }

    table td {
        padding:8px;
        border:1px solid #1e293b;
        text-align:center;
    }

    table tr:hover {
        background:#1e293b;
    }
    </style>
    """, unsafe_allow_html=True)


    # ================= HEADER =================
    st.markdown("""
    <div class="dup-card">
        <div class="dup-title">📊 Razones Financieras</div>
        <div class="dup-subtitle">
            Análisis financiero comparativo multi-periodo
        </div>
    </div>
    """, unsafe_allow_html=True)


    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona estados financieros desde tu base.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])
    anios = datos.get("años", [])

    if not balances or not resultados:
        st.warning("⚠ Este módulo requiere Balance General y Estado de Resultados.")
        st.stop()

    todas_las_razones = []

    try:
        for i in range(min(len(balances), len(resultados))):
            bg = balances[i]
            er = resultados[i]
            anio = anios[i]

            razones = compute_ratios(bg, er)

            if not razones:
                continue

            for nombre, valor in razones.items():
                todas_las_razones.append({
                    "Empresa": empresa,
                    "Periodo": anio,
                    "Razón": nombre,
                    "Valor": round(float(valor), 2)
                })

        if len(todas_las_razones) == 0:
            st.error("❌ No se pudieron calcular razones financieras.")
            st.stop()

        df_r = pd.DataFrame(todas_las_razones)

        # ================= KPIS =================
        ultimo_anio = df_r["Periodo"].max()
        df_ultimo = df_r[df_r["Periodo"] == ultimo_anio]

        def obtener_valor(df, nombre):
            v = df[df["Razón"] == nombre]["Valor"]
            return float(v.iloc[0]) if not v.empty else 0

        liquidez = obtener_valor(df_ultimo, "Razón Circulante")
        endeudamiento = obtener_valor(df_ultimo, "Razón de Endeudamiento")
        roa = obtener_valor(df_ultimo, "ROA")
        roe = obtener_valor(df_ultimo, "ROE")

        st.markdown(f"""
        <div class="dup-subcard">
            📆 Último período analizado: <b>{ultimo_anio}</b>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="dup-metric">
                <h4>💧 Razón Circulante</h4>
                <h2>{liquidez:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="dup-metric">
                <h4>🏦 Endeudamiento</h4>
                <h2>{endeudamiento:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="dup-metric">
                <h4>🚀 ROA</h4>
                <h2>{roa:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="dup-metric">
                <h4>📈 ROE</h4>
                <h2>{roe:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # ================= TABLA ESTILADA =================
        pivot = df_r.pivot(
            index="Razón",
            columns="Periodo",
            values="Valor"
        ).round(2).reset_index()

        df_display = pivot.copy()

        for col in df_display.columns:
            if col != "Razón":
                df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")

        st.markdown("""
        <div class="dup-subcard">
            <h4 style="color:#38BDF8;">📋 Desglose de razones por período</h4>
            <p style="color:#94a3b8;font-size:14px;">
                Todas las razones redondeadas a 2 decimales
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            df_display.to_html(index=False, justify="center"),
            unsafe_allow_html=True
        )

        # ================= RADAR FINANCIERO =================
        st.markdown(f"""
        <div class="dup-subcard">
            <h4 style="color:#38BDF8;">🕸 Radar Financiero ({ultimo_anio})</h4>
            <p style="color:#94a3b8;font-size:14px;">Visualización financiera del último período</p>
        </div>
        """, unsafe_allow_html=True)

        labels = ["Razón Circulante", "Razón de Endeudamiento", "ROA", "ROE"]
        values = [liquidez, endeudamiento, roa, roe]

        radar_fig = go.Figure()

        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            marker=dict(size=6, color="#38BDF8"),
            line=dict(color="#38BDF8", width=3)
        ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor="#020617",
                radialaxis=dict(
                    visible=True,
                    gridcolor="#1e293b",
                    tickfont=dict(color="#94a3b8")
                ),
                angularaxis=dict(
                    tickfont=dict(color="#e2e8f0", size=12)
                )
            ),
            paper_bgcolor="#020617",
            font=dict(color="#e2e8f0", size=13),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False
        )

        st.plotly_chart(radar_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error al generar Razones financieras: {str(e)}")

# ----------------------------
# Sección: DuPont
# ----------------------------
elif section == "DuPont":

    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px

    # ================== ESTILO ==================
    st.markdown("""
    <style>
    @keyframes slideFade {
        from {opacity: 0; transform: translateY(15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .dup-card {
        background: linear-gradient(145deg, #020617, #0f172a);
        padding: 22px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.5);
        animation: slideFade 0.7s ease;
    }

    .dup-subcard {
        background: #020617;
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #1e293b;
        margin-bottom: 16px;
        animation: slideFade 1s ease;
    }

    .dup-title {
        color: #38BDF8;
        font-size: 26px;
        font-weight: bold;
    }

    .dup-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .dup-metric {
        background:#020617;
        padding:12px;
        border-radius:12px;
        border:1px solid #1e293b;
        text-align:center;
        margin-bottom:10px;
    }

    .dup-metric h4 {
        color:#94a3b8;
        font-size:14px;
        margin-bottom:5px;
    }

    .dup-metric h2 {
        color:#38BDF8;
        font-size:22px;
        font-weight:bold;
    }

    table {
        width:100%;
        border-collapse: collapse;
        background:#020617;
        color:#e2e8f0;
        font-size:14px;
    }

    table th {
        background:#0f172a;
        padding:8px;
        border:1px solid #1e293b;
        color:#38BDF8;
        text-align:center;
    }

    table td {
        padding:8px;
        border:1px solid #1e293b;
        text-align:center;
    }

    table tr:hover {
        background:#1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================= HEADER =================
    st.markdown("""
    <div class="dup-card">
        <div class="dup-title">📈 Análisis DuPont</div>
        <div class="dup-subtitle">
            Descomposición del ROE: Margen · Rotación · Apalancamiento
        </div>
    </div>
    """, unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona estados financieros desde tu base.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])

    if not balances or not resultados:
        st.warning("⚠ DuPont requiere Balance General y Estado de Resultados.")
        st.stop()

    # ================= EMPAREJAR POR AÑO =================

    resultados_dict = {er["Año"].iloc[0]: er for er in resultados}
    dfs_dupont = []

    for bg in balances:
        anio = bg["Año"].iloc[0]

        if anio not in resultados_dict:
            continue

        er = resultados_dict[anio]

        try:
            df_dup = compute_dupont(bg, er)
            df_dup["Periodo"] = anio
            dfs_dupont.append(df_dup)
        except:
            continue

    if not dfs_dupont:
        st.error("No se pudo calcular DuPont.")
        st.stop()

    df_total = pd.concat(dfs_dupont)

    # Redondeo global
    df_total["Valor"] = pd.to_numeric(df_total["Valor"], errors="coerce").round(2)

    # ================= KPIs ESTILO FLUJO =================

    ultimo_anio = df_total["Periodo"].max()
    df_ultimo = df_total[df_total["Periodo"] == ultimo_anio]

    def get_val(df, name):
        v = df[df["Componente"].str.contains(name, case=False)]["Valor"]
        return round(float(v.iloc[0]), 2) if not v.empty else np.nan

    roe = get_val(df_ultimo, "ROE")
    margen = get_val(df_ultimo, "Margen")
    rotacion = get_val(df_ultimo, "Rotación")
    apalancamiento = get_val(df_ultimo, "Apalancamiento")

    st.markdown("""
    <div class="dup-subcard">
        📆 Último período analizado: <b>{}</b>
    </div>
    """.format(ultimo_anio), unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="dup-metric">
            <h4>📌 ROE</h4>
            <h2>{roe:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="dup-metric">
            <h4>📊 Margen Neto</h4>
            <h2>{margen:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="dup-metric">
            <h4>🔁 Rotación de Activos</h4>
            <h2>{rotacion:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="dup-metric">
            <h4>🏦 Apalancamiento</h4>
            <h2>{apalancamiento:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ================= TABLA ESTILO ANALISIS VERTICAL =================

    pivot = df_total.pivot(
        index="Componente",
        columns="Periodo",
        values="Valor"
    ).reset_index()

    pivot = pivot.round(2)

    display_df = pivot.copy()

    for col in display_df.columns:
        if col != "Componente":
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

    st.markdown("""
    <div class="dup-subcard">
        <h4 style="color:#38BDF8;">📋 Desglose DuPont por Período</h4>
        <p style="color:#94a3b8;font-size:14px;">
            Valores redondeados a 2 decimales
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        display_df.to_html(index=False, justify="center"),
        unsafe_allow_html=True
    )

    # ================= GRÁFICO PRINCIPAL =================

    COLORS = ["#38BDF8", "#6366F1", "#22C55E", "#FACC15"]

    fig = px.bar(
        df_total,
        x="Periodo",
        y="Valor",
        color="Componente",
        barmode="group",
        text="Valor",
        color_discrete_sequence=COLORS
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_dark",
        height=540,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(size=14),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Valor", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=40, r=40, t=50, b=80),
        uniformtext_minsize=9,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= GRÁFICO SECUNDARIO =================

    fig2 = px.line(
        df_total,
        x="Periodo",
        y="Valor",
        color="Componente",
        markers=True,
        color_discrete_sequence=COLORS
    )

    fig2.update_layout(
        template="plotly_dark",
        height=520,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=70)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ================= EXPORTACIÓN =================

    excel_bytes = df_to_excel_bytes({
        f"DuPont_{empresa}": pivot
    })

    st.download_button(
        "📥 Exportar DuPont (Excel)",
        data=excel_bytes,
        file_name=f"DuPont_{empresa}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
#-----------------------------
# Sección: Flujo de Efectivo
# ----------------------------
elif section == "Flujo de efectivo":

    import streamlit as st
    import plotly.express as px
    import pandas as pd
    import numpy as np

    # ================== ESTILO (MISMO LOOK DUPONT) ==================
    st.markdown("""
    <style>
    @keyframes slideFade {
        from {opacity: 0; transform: translateY(15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .efe-card {
        background: linear-gradient(145deg, #020617, #0f172a);
        padding: 22px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.5);
        animation: slideFade 0.7s ease;
    }

    .efe-subcard {
        background: #020617;
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #1e293b;
        margin-bottom: 16px;
        animation: slideFade 1s ease;
    }

    .efe-title {
        color: #38BDF8;
        font-size: 26px;
        font-weight: bold;
    }

    .efe-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .efe-metric {
        background:#020617;
        padding:12px;
        border-radius:12px;
        border:1px solid #1e293b;
        text-align:center;
        margin-bottom:10px;
    }

    .efe-metric h4 {
        color:#94a3b8;
        font-size:14px;
        margin-bottom:5px;
    }

    .efe-metric h2 {
        color:#38BDF8;
        font-size:22px;
        font-weight:bold;
    }

    table {
        width:100%;
        border-collapse: collapse;
        background:#020617;
        color:#e2e8f0;
        font-size:14px;
    }

    table th {
        background:#0f172a;
        padding:8px;
        border:1px solid #1e293b;
        color:#38BDF8;
        text-align:center;
    }

    table td {
        padding:8px;
        border:1px solid #1e293b;
        text-align:center;
    }

    table tr:hover {
        background:#1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

    COLORS = ["#38BDF8", "#6366F1", "#22C55E"]

    # ================= HEADER =================
    st.markdown("""
    <div class="efe-card">
        <div class="efe-title">💧 Estado de Flujo de Efectivo</div>
        <div class="efe-subtitle">
            Método Directo e Indirecto · Comparativo por período
        </div>
    </div>
    """, unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona estados financieros desde tu base.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])
    anios = datos.get("años", [])

    if not balances or not resultados:
        st.warning("⚠ El flujo requiere BG y ER cargados.")
        st.stop()

    if len(balances) < 2:
        st.warning("⚠ Necesitas mínimo 2 balances para comparar.")
        st.stop()

    bg_actual = balances[-1]
    er_actual = resultados[-1]
    bg_anterior = balances[-2]

    anio = anios[-1]
    anio_anterior = anios[-2]

    st.markdown(f"""
    <div class="efe-subcard">
        📆 Comparación: <b>{anio_anterior}</b> → <b>{anio}</b>
    </div>
    """, unsafe_allow_html=True)

    # ================= CÁLCULO =================
    with st.spinner("💧 Calculando flujo de efectivo..."):
        df_efe_ind = compute_cashflow_indirect(bg_actual, er_actual, bg_anterior)
        df_efe_dir = compute_cashflow_direct(bg_actual, er_actual)

    df_efe_ind["Monto"] = pd.to_numeric(df_efe_ind["Monto"], errors="coerce").round(2)
    df_efe_dir["Monto"] = pd.to_numeric(df_efe_dir["Monto"], errors="coerce").round(2)

    # ================= KPIs (SE QUEDAN IGUAL) =================
    A = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Operación (A)","Monto"].sum())
    B = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Inversión (B)","Monto"].sum())
    C = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Financiamiento (C)","Monto"].sum())
    N = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo Neto (A+B+C)","Monto"].sum())

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="efe-metric">
            <h4>💼 Operación</h4>
            <h2>${A:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="efe-metric">
            <h4>🏗 Inversión</h4>
            <h2>${B:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="efe-metric">
            <h4>🏦 Financiamiento</h4>
            <h2>${C:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="efe-metric">
            <h4>💰 Flujo Neto</h4>
            <h2>${N:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ================= TABLAS FORMATO DUPONT =================

    def pretty_table(df):
        temp = df.copy()
        temp["Monto"] = temp["Monto"].map(lambda x: f"{x:,.2f}")
        return temp.to_html(index=False, justify="center")

    st.markdown("""
    <div class="efe-subcard">
        <h4 style="color:#38BDF8;">📋 Método Indirecto</h4>
        <p style="color:#94a3b8;font-size:14px;">
        Flujo generado a partir del resultado y variaciones operativas
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(pretty_table(df_efe_ind), unsafe_allow_html=True)

    st.markdown("""
    <div class="efe-subcard">
        <h4 style="color:#38BDF8;">📋 Método Directo</h4>
        <p style="color:#94a3b8;font-size:14px;">
        Entradas y salidas reales de efectivo
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(pretty_table(df_efe_dir), unsafe_allow_html=True)

    # ================= GRÁFICO PRINCIPAL =================
    chart_df = pd.DataFrame({
        "Actividad": ["Operación", "Inversión", "Financiamiento"],
        "Monto": [A, B, C]
    })

    st.markdown("""
    <div class="efe-subcard">
        <h4 style="color:#38BDF8;">📊 Distribución del Flujo</h4>
    </div>
    """, unsafe_allow_html=True)

    fig = px.bar(
        chart_df,
        x="Actividad",
        y="Monto",
        text="Monto",
        color="Actividad",
        color_discrete_sequence=COLORS
    )

    fig.update_traces(
        texttemplate="%{text:$,.2f}",
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(size=14),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(
            title="Monto ($)",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=50, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= GRÁFICO SECUNDARIO =================

    df_comp = chart_df.copy()

    fig2 = px.line(
        df_comp,
        x="Actividad",
        y="Monto",
        markers=True,
        color="Actividad",
        color_discrete_sequence=COLORS
    )

    fig2.update_layout(
        template="plotly_dark",
        height=480,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=70)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ================= EXPORTACIÓN =================
    excel_bytes = df_to_excel_bytes({
        "Flujo_Indirecto": df_efe_ind.round(2),
        "Flujo_Directo": df_efe_dir.round(2)
    })

    st.download_button(
        "📥 Exportar Flujo de Efectivo (Excel)",
        data=excel_bytes,
        file_name=f"Flujo_{empresa}_{anio_anterior}_{anio}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


 
# ----------------------------
# Sección: Estado de Origen y Aplicación
# ----------------------------
elif section == "Origen y Aplicación de Fondos":

    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import traceback

    # ======================================================
    # 🔒 ESTILO DARK TOTAL
    # ======================================================
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"]  {
        background-color: #020617 !important;
        color: #f1f5f9 !important;
    }

    .block-container {
        background-color: #020617 !important;
    }

    .eoaf-card {
        background: linear-gradient(145deg, #020617, #0f172a);
        padding: 22px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.7);
    }

    .eoaf-box {
        background:#020617;
        border:1px solid #1e293b;
        padding:15px;
        border-radius:14px;
        margin-bottom:15px;
    }

    /* ---- TABLAS ESTILO DUPONT ---- */
    .eoaf-table {
        width: 100%;
        border-collapse: collapse;
        background:#020617;
        color:#e2e8f0;
        font-size:14px;
        border-radius:12px;
        overflow:hidden;
    }

    .eoaf-table th {
        background:#0f172a;
        padding:10px;
        border:1px solid #1e293b;
        color:#38BDF8;
        text-align:center;
        font-size:15px;
    }

    .eoaf-table td {
        padding:10px;
        border:1px solid #1e293b;
        text-align:center;
    }

    .eoaf-table tr:hover {
        background:#1e293b;
    }

    .eoaf-title {
        color:#38BDF8;
        font-size:18px;
        margin:15px 0 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ======================================================
    # ETIQUETA
    # ======================================================
    st.markdown("""
    <div class="eoaf-card">
        <h2 style="color:#38BDF8;">📊 Estado de Origen y Aplicación de Fondos</h2>
        <p style="color:#94a3b8;">Comparación entre periodos</p>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # OBTENER DATOS
    # ======================================================
    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.error("❌ No hay estado financiero seleccionado desde la BD.")
        st.stop()

    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])
    anios = datos.get("años", [])

    if len(balances) < 2 or len(resultados) < 1:
        st.error("⚠ Necesitas mínimo 2 Balances y 1 Estado de Resultados.")
        st.stop()

    bg_anterior = balances[-2]
    bg_actual = balances[-1]
    er_actual = resultados[-1]

    anio_actual = anios[-1]
    anio_anterior = anios[-2]

    st.success(f"📆 Comparando {anio_anterior} → {anio_actual}")

    # ======================================================
    # PROCESO
    # ======================================================
    try:
        df_origen, df_aplic, df_resumen = compute_eoaf(
            bg_anterior,
            bg_actual,
            er_actual
        )
    except Exception:
        st.error("❌ Error en compute_eoaf")
        st.code(traceback.format_exc())
        st.stop()

    for df in [df_origen, df_aplic, df_resumen]:
        if not df.empty:
            df["Monto"] = pd.to_numeric(df["Monto"], errors="coerce").fillna(0).round(2)

    total_origen = df_origen["Monto"].sum() if not df_origen.empty else 0
    total_aplic = df_aplic["Monto"].sum() if not df_aplic.empty else 0
    diferencia = total_origen - total_aplic

    color_diff = "#22c55e" if diferencia >= 0 else "#ef4444"

    # ======================================================
    # TARJETAS KPI
    # ======================================================
    st.markdown(f"""
    <div class="eoaf-box" style="display:flex; justify-content:space-around; text-align:center;">
        <div>
            <h2 style="color:#22c55e;">${total_origen:,.2f}</h2>
            <span>Orígenes</span>
        </div>
        <div>
            <h2 style="color:#f59e0b;">${total_aplic:,.2f}</h2>
            <span>Aplicaciones</span>
        </div>
        <div>
            <h2 style="color:{color_diff};">${diferencia:,.2f}</h2>
            <span>Diferencia</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # FUNCIÓN PARA TABLAS ESTILO ELEGANTE
    # ======================================================
    def render_table(df):
        if df.empty:
            return "<p style='color:#94a3b8'>Sin datos</p>"

        df_html = df.copy()

        if "Monto" in df_html.columns:
            df_html["Monto"] = df_html["Monto"].map(lambda x: f"${x:,.2f}")

        return df_html.to_html(
            index=False,
            justify="center",
            classes="eoaf-table",
            border=0,
            escape=False
        )

    # ======================================================
    # TABLAS ESTILO DUPONT
    # ======================================================
    st.markdown("<div class='eoaf-title'>📥 Orígenes de Fondos</div>", unsafe_allow_html=True)
    st.markdown(render_table(df_origen), unsafe_allow_html=True)

    st.markdown("<div class='eoaf-title'>📤 Aplicaciones de Fondos</div>", unsafe_allow_html=True)
    st.markdown(render_table(df_aplic), unsafe_allow_html=True)

    st.markdown("<div class='eoaf-title'>📌 Resumen EOAF</div>", unsafe_allow_html=True)
    st.markdown(render_table(df_resumen), unsafe_allow_html=True)

    # ======================================================
    # 📊 GRÁFICO
    # ======================================================
    chart_df = pd.DataFrame({
        "Tipo": ["Orígenes", "Aplicaciones"],
        "Monto": [total_origen, total_aplic]
    })

    fig = px.bar(
        chart_df,
        x="Tipo",
        y="Monto",
        text_auto=True
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(size=14),
        height=480,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(title="Monto", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title=None)
    )

    fig.update_traces(
        texttemplate="%{y:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

#-----------------------------
# Sección: Interpretación IA
# ----------------------------
elif section == "Interpretación":

    import pandas as pd
    import streamlit as st
    import plotly.express as px

    # ================== ESTILO VISUAL ==================
    st.markdown("""
    <style>
    .interp-card {
        background: linear-gradient(145deg, #020617, #0f172a);
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0px 0px 22px rgba(0,0,0,0.6);
        margin-bottom: 20px;
        animation: slideFade 0.6s ease;
    }

    .interp-title {
        color:#38BDF8;
        font-size:26px;
        font-weight:bold;
    }

    .interp-subtitle {
        color:#94a3b8;
        font-size:15px;
        margin-bottom:8px;
    }

    .interp-box {
        background:#020617;
        border:1px solid #1e293b;
        padding:15px;
        border-radius:14px;
        margin-top:15px;
        margin-bottom:15px;
    }

    .interp-result {
        background:#0f172a;
        border-left:6px solid #38BDF8;
        padding:22px;
        border-radius:14px;
        margin-top:18px;
        line-height:1.8;
        font-size:15px;
        color:#e2e8f0;
    }

    .metric-box {
        background:#020617;
        padding:16px;
        border-radius:14px;
        border:1px solid #1e293b;
        text-align:center;
    }

    .metric-box h4 {
        color:#94a3b8;
        font-size:14px;
    }

    .metric-box h2 {
        color:#38BDF8;
        font-size:20px;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================== HEADER ==================
    st.markdown("""
    <div class="interp-card">
        <div class="interp-title">📊 Interpretación Financiera</div>
        <div class="interp-subtitle">
            Análisis general basado en los estados financieros seleccionados
        </div>
    </div>
    """, unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("Primero selecciona un estado financiero en 'Usar estados guardados'.")
        st.stop()

    empresa = datos.get("empresa")
    anios = datos.get("años", [])
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])

    if not balances or not resultados:
        st.warning("El estado seleccionado debe contener Balance General y Estado de Resultados.")
        st.stop()

    # ✅ Ahora trabajamos con TODOS los años
    summary = f"""
    Análisis financiero general de la empresa {empresa}
    Periodos analizados: {min(anios)} a {max(anios)}
    Cantidad de estados revisados: {len(balances)} balances y {len(resultados)} resultados.
    """

    for i in range(len(balances)):
        bg = balances[i]
        er = resultados[i]
        anio = anios[i]

        summary += f"\n--- PERIODO {anio} ---\n"

        # === Análisis vertical BG ===
        try:
            dv_bg = vertical_analysis(empresa, bg, "BG")

            resumen_bg = dv_bg[['Cuenta', 'Porcentaje']]\
                .dropna()\
                .sort_values("Porcentaje", ascending=False)\
                .head(5)

            summary += "\nBalance General - principales cuentas:\n"
            summary += resumen_bg.to_string(index=False) + "\n"

        except Exception as e:
            summary += f"\n(No se pudo generar análisis vertical BG: {e})\n"

        # === Análisis vertical ER ===
        try:
            dv_er = vertical_analysis(empresa, er, "ER")

            resumen_er = dv_er[['Cuenta', 'Porcentaje']]\
                .dropna()\
                .sort_values("Porcentaje", ascending=False)\
                .head(5)

            summary += "\nEstado de Resultados - principales cuentas:\n"
            summary += resumen_er.to_string(index=False) + "\n"

        except Exception as e:
            summary += f"\n(No se pudo generar análisis vertical ER: {e})\n"

        # === KPIs ===
        try:
            ratios = compute_ratios(bg, er)
            df_kpis = pd.DataFrame.from_dict(
                ratios, orient="index", columns=["Valor"]
            ).reset_index()

            summary += "\nIndicadores financieros del periodo:\n"
            summary += df_kpis.to_string(index=False)
            summary += "\n"

        except Exception as e:
            summary += f"\n(No se pudieron generar indicadores: {e})\n"

    # ========================== PREVIEW TÉCNICO ==========================
    with st.expander("📄 Ver resumen técnico completo"):
        st.code(summary[:4000])

    # ========================== BOTÓN DE GENERACIÓN ==========================
    st.markdown("### 📝 Generar interpretación automática")

    if st.button("Generar informe completo"):
        with st.spinner("Analizando información financiera..."):
            texto = generate_interpretation_gemini(summary)

            if texto.lower().startswith("error"):
                st.error("No se pudo generar la interpretación.")
            else:
                st.session_state.interpretacion_general = texto
                st.success("Informe generado correctamente.")

    # ========================== MOSTRAR RESULTADO ==========================
    if st.session_state.get("interpretacion_general"):

        st.markdown("""
        <div class="interp-result">
            <h3 style="color:#38BDF8;">📘 Informe Financiero General</h3>
        """, unsafe_allow_html=True)

        # ⚠️ Filtro para quitar cualquier mención a "IA"
        texto_limpio = st.session_state.interpretacion_general.replace("IA", "").replace("inteligencia artificial", "")

        st.markdown(texto_limpio, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr class='st-sep'/>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 FinanSys. Todos los derechos reservados.</div>", unsafe_allow_html=True)
