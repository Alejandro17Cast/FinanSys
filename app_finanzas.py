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

# ----------------------------
# Utilidades robustas
# ----------------------------
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

def map_accounts(df: pd.DataFrame, account_map: dict) -> dict:
    if df is None or "Cuenta" not in df.columns or "Monto" not in df.columns:
        return {k: 0.0 for k in account_map.keys()}
    d = df.copy()
    d["Cuenta"] = d["Cuenta"].astype(str)
    d["Cuenta_norm"] = d["Cuenta"].apply(norm_account_name)
    d["Monto"] = to_numeric_series(d["Monto"])
    result = {}
    for key, patterns in account_map.items():
        result[key] = find_account_amount(d, patterns)
    return result

def safe_get(map_obj: dict, key: str) -> float:
    try:
        return float(map_obj.get(key, 0.0))
    except Exception:
        return 0.0

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

    # Normalizar nombres de cuentas
    dfv["cuenta_norm"] = dfv["cuenta"].astype(str).str.lower().str.strip()

    # =====================================================
    # CÁLCULO AUTOMÁTICO DE TOTALES
    # =====================================================
    total = None

    # ---------------------------------------------
    # 1️⃣ BALANCE GENERAL (BG)
    # ---------------------------------------------
    if tipo == "BG":

        # A. Buscar TOTAL ACTIVO explícito
        if dfv["cuenta_norm"].str.contains("total activo").any():
            total = dfv.loc[
                dfv["cuenta_norm"].str.contains("total activo"),
                "monto"
            ].sum()

        # B. Si no existe, inferir activos por nombres comunes
        if total is None or total == 0:

            activos_mask = dfv["cuenta_norm"].str.contains(
                "activo|caja|bancos|cliente|clientes|inventario|propiedad|planta|equipo|maquinaria|terreno|edificio|intangible",
                case=False,
                na=False
            )

            total = dfv.loc[activos_mask, "monto"].sum()

        # C. Validación final
        if total == 0:
            st.error("❌ No se pudo calcular Total Activos. Revisa la estructura del Balance.")
            return None

    # ---------------------------------------------
    # 2️⃣ ESTADO DE RESULTADOS (ER)
    # ---------------------------------------------
    elif tipo == "ER":

        # A. Detectar ventas totales explícitas
        if dfv["cuenta_norm"].str.contains("total ingreso|total ventas").any():
            total = dfv.loc[
                dfv["cuenta_norm"].str.contains("total ingreso|total ventas"),
                "monto"
            ].sum()

        # B. Si no existe, inferir por palabras clave
        if total is None or total == 0:

            ventas_mask = dfv["cuenta_norm"].str.contains(
                "venta|ventas|ingreso|ingresos",
                case=False,
                na=False
            )

            total = dfv.loc[ventas_mask, "monto"].sum()

        # C. Validación final
        if total == 0:
            st.error("❌ No se pudo calcular Ventas Totales. Revisa si existe 'Ventas' o 'Ingresos'.")
            return None

    else:
        st.error("Tipo inválido.")
        return None

    # =====================================================
    # PORCENTAJE
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

    bg["cuenta"] = bg["cuenta"].str.lower()
    er["cuenta"] = er["cuenta"].str.lower()

    bg["monto"] = pd.to_numeric(bg["monto"], errors="coerce").fillna(0)
    er["monto"] = pd.to_numeric(er["monto"], errors="coerce").fillna(0)

    # ========================
    # Extraer valores base
    # ========================

    activos_corr = bg[bg["cuenta"].str.contains("activo circulante|activo corriente|efectivo|banco|caja", na=False)]["monto"].sum()
    activos = bg[bg["cuenta"].str.contains("activo", na=False)]["monto"].sum()
    activos_fijos = bg[bg["cuenta"].str.contains("activo fijo|propiedad|planta|equipo", na=False)]["monto"].sum()

    pasivos_corr = bg[bg["cuenta"].str.contains("pasivo circulante|pasivo corriente|cuentas por pagar|proveedores", na=False)]["monto"].sum()
    pasivos = bg[bg["cuenta"].str.contains("pasivo", na=False)]["monto"].sum()

    capital = bg[bg["cuenta"].str.contains("capital|patrimonio", na=False)]["monto"].sum()

    inventarios = bg[bg["cuenta"].str.contains("inventario", na=False)]["monto"].sum()
    cuentas_cobrar = bg[bg["cuenta"].str.contains("cuentas por cobrar|clientes", na=False)]["monto"].sum()

    ventas = er[er["cuenta"].str.contains("venta|ingreso", na=False)]["monto"].sum()
    costo_ventas = er[er["cuenta"].str.contains("costo", na=False)]["monto"].sum()

    utilidad_bruta = ventas - costo_ventas

    utilidad_operativa = er[er["cuenta"].str.contains("operativa|utilidad operativa", na=False)]["monto"].sum()
    utilidad_neta = er[er["cuenta"].str.contains("utilidad neta|resultado neto", na=False)]["monto"].sum()

    intereses = er[er["cuenta"].str.contains("intereses|gasto financiero", na=False)]["monto"].sum()

    # ========================
    # ✅ RAZONES DE LIQUIDEZ
    # ========================
    razon_circulante = activos_corr / pasivos_corr if pasivos_corr != 0 else np.nan
    razon_rapida = (activos_corr - inventarios) / pasivos_corr if pasivos_corr != 0 else np.nan

    # ========================
    # ✅ RAZONES DE ACTIVIDAD
    # ========================
    rotacion_inventarios = costo_ventas / inventarios if inventarios != 0 else np.nan
    rotacion_cxc = ventas / cuentas_cobrar if cuentas_cobrar != 0 else np.nan
    periodo_cobro = 360 / rotacion_cxc if rotacion_cxc and rotacion_cxc != 0 else np.nan
    rotacion_activos_fijos = ventas / activos_fijos if activos_fijos != 0 else np.nan
    rotacion_activos_totales = ventas / activos if activos != 0 else np.nan

    # ========================
    # ✅ RAZONES DE ENDEUDAMIENTO
    # ========================
    razon_endeudamiento = pasivos / activos if activos != 0 else np.nan
    pasivo_capital = pasivos / capital if capital != 0 else np.nan
    cobertura_intereses = utilidad_operativa / intereses if intereses != 0 else np.nan

    # ========================
    # ✅ RAZONES DE RENTABILIDAD
    # ========================
    margen_bruto = utilidad_bruta / ventas if ventas != 0 else np.nan
    margen_operativo = utilidad_operativa / ventas if ventas != 0 else np.nan
    margen_neto = utilidad_neta / ventas if ventas != 0 else np.nan
    roa = utilidad_neta / activos if activos != 0 else np.nan

    # ========================
    # 📦 RESULTADO FINAL
    # ========================
    razones = {

        # Liquidez
        "Razón Circulante": razon_circulante,
        "Razón Rápida": razon_rapida,

        # Actividad
        "Rotación de Inventarios": rotacion_inventarios,
        "Rotación Cuentas por Cobrar": rotacion_cxc,
        "Periodo Promedio de Cobro (días)": periodo_cobro,
        "Rotación de Activos Fijos": rotacion_activos_fijos,
        "Rotación de Activos Totales": rotacion_activos_totales,

        # Endeudamiento
        "Razón de Endeudamiento": razon_endeudamiento,
        "Pasivo / Capital": pasivo_capital,
        "Cobertura de Intereses": cobertura_intereses,

        # Rentabilidad
        "Margen Bruto": margen_bruto,
        "Margen Operativo": margen_operativo,
        "Margen Neto": margen_neto,
        "ROA": roa
    }

    return razones


def compute_dupont(balance_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    bal = map_accounts(balance_df, ACCOUNT_MAP_BALANCE)
    res = map_accounts(results_df, ACCOUNT_MAP_RESULTS)

    ventas = safe_get(res, "VENTAS_NETAS")
    utilidad_neta = safe_get(res, "UTILIDAD_NETA")
    activo_total = safe_get(bal, "ACTIVO_TOTAL")
    patrimonio = safe_get(bal, "PATRIMONIO")

    eps = 1e-9

    # Componentes DuPont
    margen_neto = utilidad_neta / max(ventas, eps)
    rotacion_activos = ventas / max(activo_total, eps)
    apalancamiento = activo_total / max(patrimonio, eps)

    # ROE final
    roe = margen_neto * rotacion_activos * apalancamiento

    dupont_df = pd.DataFrame({
        "Componente": [
            "Margen Neto (Utilidad Neta / Ventas)",
            "Rotación de Activos (Ventas / Activo Total)",
            "Apalancamiento Financiero (Activo Total / Patrimonio)",
            "ROE (Retorno sobre el Patrimonio)"
        ],
        "Valor": [
            round(margen_neto, 4),
            round(rotacion_activos, 4),
            round(apalancamiento, 4),
            round(roe, 4)
        ]
    })

    return dupont_df



# ----------------------------
# Flujo de efectivo (Indirecto + Directo)
# ----------------------------

def remove_totals(df):
    """
    Elimina todas las filas que sean totales:
    total, subtotal, suma, activo total, pasivo total, patrimonio total, etc.
    """
    d = df.copy()
    d["Cuenta_norm"] = d["Cuenta"].apply(norm_account_name)

    d = d[~d["Cuenta_norm"].str.contains(
        r"\btotal\b|\bsubtotal\b|\bsuma\b|activo total|pasivo total|patrimonio total|activos totales|pasivos totales",
        regex=True,
        na=False
    )]

    d["Monto"] = to_numeric_series(d["Monto"])
    return d



def compute_cashflow_direct(balance_df_act: pd.DataFrame, results_df_act: pd.DataFrame) -> pd.DataFrame:
    """
    Flujo de Efectivo Método DIRECTO — versión SIN totales.
    Solo usa cuentas operativas reales del BG y ER.
    """

    # -----------------------------
    # LIMPIAR: remover totales
    # -----------------------------
    bal = remove_totals(balance_df_act)
    res = remove_totals(results_df_act)

    # -----------------------------
    # COBROS A CLIENTES
    # -----------------------------
    ventas = res.loc[
        res["Cuenta_norm"].str.contains(r"venta|ingreso", na=False),
        "Monto"
    ].sum()

    cxc = bal.loc[
        bal["Cuenta_norm"].str.contains(r"cobrar|cliente|deudor", na=False),
        "Monto"
    ].sum()

    cobros_clientes = ventas - cxc

    # -----------------------------
    # PAGOS A PROVEEDORES
    # -----------------------------
    compras = res.loc[
        res["Cuenta_norm"].str.contains(r"costo", na=False),
        "Monto"
    ].sum()

    cxp = bal.loc[
        bal["Cuenta_norm"].str.contains(r"pagar|proveedor", na=False),
        "Monto"
    ].sum()

    pagos_proveedores = -(compras - cxp)

    # -----------------------------
    # GASTOS OPERATIVOS PAGADOS
    # -----------------------------
    gastos_operativos = res.loc[
        res["Cuenta_norm"].str.contains(r"gasto|operaci|servicio", na=False),
        "Monto"
    ].sum()

    pagos_operativos = -gastos_operativos

    # -----------------------------
    # IMPUESTOS & INTERESES
    # -----------------------------
    impuestos = res.loc[
        res["Cuenta_norm"].str.contains(r"impuesto|isr|iva|tribut", na=False),
        "Monto"
    ].sum()

    intereses = res.loc[
        res["Cuenta_norm"].str.contains(r"interes|financ", na=False),
        "Monto"
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
def compute_cashflow_indirect(balance_df_act: pd.DataFrame, results_df_act: pd.DataFrame, balance_df_prev: pd.DataFrame) -> pd.DataFrame:
    """
    Flujo de Efectivo Método INDIRECTO.
    Usa variaciones del BG real (cuentas operativas solamente).
    """

    bal_a = remove_totals(balance_df_act)
    bal_p = remove_totals(balance_df_prev)
    res_a = remove_totals(results_df_act)

    # --- Utilidad Neta ---
    utilidad_neta = res_a.loc[
        res_a["Cuenta_norm"].str.contains(r"utilidad neta|resultado neto|ganancia", na=False),
        "Monto"
    ].sum()

    # --- Depreciación ---
    depreciacion = res_a.loc[
        res_a["Cuenta_norm"].str.contains(r"deprecia|amortiz", na=False),
        "Monto"
    ].sum()

    # --- Variaciones operativas ---
    def get_val(df, pattern):
        return df.loc[df["Cuenta_norm"].str.contains(pattern, na=False), "Monto"].sum()

    cxc_a = get_val(bal_a, r"cobrar|cliente|deudor")
    cxc_p = get_val(bal_p, r"cobrar|cliente|deudor")

    inv_a = get_val(bal_a, r"invent")
    inv_p = get_val(bal_p, r"invent")

    prov_a = get_val(bal_a, r"pagar|proveedor")
    prov_p = get_val(bal_p, r"pagar|proveedor")

    # Variaciones
    cambio_cxc = cxc_a - cxc_p
    cambio_inv = inv_a - inv_p
    cambio_prov = prov_a - prov_p

    flujo_operativo = utilidad_neta + depreciacion - cambio_cxc - cambio_inv + cambio_prov

    # --- Inversión: PPE (cambia el activo fijo real, no totales) ---
    af_a = get_val(bal_a, r"propiedad|activo fijo|equipo|ppe")
    af_p = get_val(bal_p, r"propiedad|activo fijo|equipo|ppe")

    flujo_inversion = -(af_a - af_p)

    # --- Financiamiento: cambios en deudas y capital (sin totales) ---
    deuda_a = get_val(bal_a, r"deuda|obligacion|prestamo|credito")
    deuda_p = get_val(bal_p, r"deuda|obligacion|prestamo|credito")

    capital_a = get_val(bal_a, r"patrimonio|capital social")
    capital_p = get_val(bal_p, r"patrimonio|capital social")

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



# ----------------------------
# Estado de Origen y Aplicación de Fondos (EOAF)
# ----------------------------

def compute_eoaf(balance_prev: pd.DataFrame, balance_act: pd.DataFrame, results_act: pd.DataFrame):
    """
    Calcula el Estado de Origen y Aplicación de Fondos usando:
    - BG anterior
    - BG actual
    - ER del periodo actual (para utilidad neta y depreciación)
    """

    # Mapear cuentas
    bal_p = map_accounts(balance_prev, ACCOUNT_MAP_BALANCE)
    bal_a = map_accounts(balance_act, ACCOUNT_MAP_BALANCE)
    res_a = map_accounts(results_act, ACCOUNT_MAP_RESULTS)

    # Copias del balance general
    df_prev = balance_prev.copy()
    df_act = balance_act.copy()

    df_prev["Cuenta_norm"] = df_prev["Cuenta"].apply(norm_account_name)
    df_act["Cuenta_norm"] = df_act["Cuenta"].apply(norm_account_name)

    df_prev["Monto"] = to_numeric_series(df_prev["Monto"])
    df_act["Monto"] = to_numeric_series(df_act["Monto"])

    # ⭐ NUEVO: remover filas que contengan "total" en cualquier forma
    df_prev = df_prev[~df_prev["Cuenta_norm"].str.contains("total", case=False, na=False)]
    df_act = df_act[~df_act["Cuenta_norm"].str.contains("total", case=False, na=False)]

    # Merge para comparar variaciones del BG
    merged = df_act.merge(
        df_prev[["Cuenta_norm", "Monto"]].rename(columns={"Monto": "Monto_Anterior"}),
        on="Cuenta_norm",
        how="left"
    )

    merged["Monto_Anterior"] = merged["Monto_Anterior"].fillna(0.0)
    merged["Variación"] = merged["Monto"] - merged["Monto_Anterior"]

    origenes = []
    aplicaciones = []

    def add_origen(cuenta, monto):
        if abs(monto) > 1e-6:
            origenes.append({"Concepto": cuenta, "Monto": round(monto, 2)})

    def add_aplicacion(cuenta, monto):
        if abs(monto) > 1e-6:
            aplicaciones.append({"Concepto": cuenta, "Monto": round(monto, 2)})

    # Clasificación EOAF
    for _, row in merged.iterrows():
        cuenta = row["Cuenta"]
        variacion = row["Variación"]
        c_norm = row["Cuenta_norm"]

        if variacion == 0:
            continue

        # Activos
        if ("activo" in c_norm or "invent" in c_norm or "caja" in c_norm or
            "banco" in c_norm or "cobrar" in c_norm):
            if variacion > 0:
                add_aplicacion(f"Aumento de {cuenta}", variacion)
            else:
                add_origen(f"Disminución de {cuenta}", abs(variacion))

        # Pasivos
        elif ("pasivo" in c_norm or "pagar" in c_norm or "proveedor" in c_norm or
              "obligacion" in c_norm or "deuda" in c_norm):
            if variacion > 0:
                add_origen(f"Aumento de {cuenta}", variacion)
            else:
                add_aplicacion(f"Disminución de {cuenta}", abs(variacion))

        # Patrimonio
        elif "patrimonio" in c_norm or "capital" in c_norm:
            if variacion > 0:
                add_origen(f"Aumento de {cuenta}", variacion)
            else:
                add_aplicacion(f"Disminución de {cuenta}", abs(variacion))

    # Orígenes desde Estado de Resultados
    utilidad = safe_get(res_a, "UTILIDAD_NETA")
    depreciacion = safe_get(res_a, "DEPRECIACION")

    add_origen("Utilidad Neta del Periodo", utilidad)
    add_origen("Depreciación y Amortización", depreciacion)

    df_origen = pd.DataFrame(origenes)
    df_aplic = pd.DataFrame(aplicaciones)

    total_origen = df_origen["Monto"].sum()
    total_aplic = df_aplic["Monto"].sum()

    resumen = pd.DataFrame({
        "Tipo": ["Total Orígenes", "Total Aplicaciones", "Diferencia (O - A)"],
        "Monto": [total_origen, total_aplic, round(total_origen - total_aplic, 2)]
    })

    return df_origen, df_aplic, resumen


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

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        enableRowGroup=True,
        enableValue=True,
        sortable=True,
        filter=True,
        resizable=True
    )
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()

    return AgGrid(
        df,
        gridOptions=gridOptions,
        theme='dark',
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

    st.markdown("### 📂 Usar estados financieros guardados (SQLite)")

    empresas = obtener_empresas()

    if not empresas:
        st.warning("No hay empresas registradas en la base de datos.")
        return

    # =========================
    # SELECCIONAR EMPRESA
    # =========================
    empresa_id, empresa_nombre = st.selectbox(
        "🏢 Empresa",
        empresas,
        format_func=lambda x: x[1]
    )

    # =========================
    # PERIODICIDAD
    # =========================
    periodicidad = st.radio("📅 Periodicidad", ["Anual", "Mensual"])
    periodicidad_db = periodicidad.lower()

    # =========================
    # AÑOS DISPONIBLES
    # =========================
    conn = get_connection()

    query_anios = """
        SELECT DISTINCT año 
        FROM estados_financieros 
        WHERE empresa_id = ?
        AND periodicidad = ?
        ORDER BY año ASC
    """

    df_anios = pd.read_sql_query(
        query_anios,
        conn,
        params=(empresa_id, periodicidad_db)
    )

    conn.close()

    if df_anios.empty:
        st.warning("Esta empresa no tiene datos para esta periodicidad.")
        return

    anios = df_anios["año"].tolist()

    anios_seleccionados = st.multiselect(
        "📅 Selecciona uno o varios años",
        anios,
        default=[max(anios)]
    )

    if not anios_seleccionados:
        st.warning("Debes seleccionar al menos un año.")
        return

    # =========================
    # MES SI ES MENSUAL
    # =========================
    mes_num = None

    if periodicidad_db == "mensual":

        conn = get_connection()

        query_meses = """
            SELECT DISTINCT mes
            FROM estados_financieros
            WHERE empresa_id = ?
            AND periodicidad = 'mensual'
            AND mes IS NOT NULL
            ORDER BY mes ASC
        """

        df_meses = pd.read_sql_query(query_meses, conn, params=(empresa_id,))
        conn.close()

        if df_meses.empty:
            st.warning("Esta empresa no tiene registros mensuales.")
            return

        meses = df_meses["mes"].tolist()

        meses_map = {
            1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
            5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto",
            9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"
        }

        mes_dict = {meses_map[m]: m for m in meses if m in meses_map}
        mes_nombre = st.selectbox("📆 Mes", list(mes_dict.keys()))
        mes_num = mes_dict[mes_nombre]

    # ==================================================
    # FUNCIÓN PARA CARGAR BG / ER
    # ==================================================
    def cargar_estado(tipo_estado, añoSeleccionado):

        conn = get_connection()

        query = """
            SELECT cuenta, monto
            FROM estados_financieros
            WHERE empresa_id = ?
            AND tipo_estado = ?
            AND año = ?
            AND periodicidad = ?
        """

        params = [empresa_id, tipo_estado, añoSeleccionado, periodicidad_db]

        if periodicidad_db == "mensual":
            query += " AND mes = ?"
            params.append(mes_num)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    # =========================
    # BOTÓN CARGAR
    # =========================
    if st.button("📂 Cargar estados financieros"):

        balances = []
        resultados = []

        for anio in anios_seleccionados:

            bg_df = cargar_estado("BG", anio)
            er_df = cargar_estado("ER", anio)

            if not bg_df.empty:
                bg_df["Año"] = anio
                balances.append(bg_df)

            if not er_df.empty:
                er_df["Año"] = anio
                resultados.append(er_df)

        if not balances and not resultados:
            st.error("❌ No existen BG ni ER para los periodos seleccionados.")
            return

        # ✅ GUARDAMOS BIEN TODO
        st.session_state.estado_seleccionado = {
            "empresa": empresa_nombre,
            "balances": balances,
            "resultados": resultados,   # 👈 ESTO ES CLAVE
            "periodicidad": periodicidad_db,
            "mes": mes_num,
            "años": anios_seleccionados
        }

        st.success(
            f"✅ Se cargaron {len(balances)} BG y {len(resultados)} ER"
        )

        # =========================
        # PREVIEW
        # =========================
        for bg in balances:
            st.markdown(f"### 📘 Balance General {bg['Año'].iloc[0]}")
            st.dataframe(bg, use_container_width=True)

        for er in resultados:
            st.markdown(f"### 📗 Estado de Resultados {er['Año'].iloc[0]}")
            st.dataframe(er, use_container_width=True)

        # Guardar estado global
        st.session_state.empresa_activa = empresa_nombre
        st.session_state.periodicidad_activa = periodicidad_db
        st.session_state.mes_activo = mes_num

        st.rerun()

    # =========================
    # ESTADO ACTUAL
    # =========================
    if st.session_state.get("estado_seleccionado"):

        datos = st.session_state.estado_seleccionado

        st.info(
            f"📌 Estados activos:\n"
            f"Empresa: {datos['empresa']} | "
            f"BG: {len(datos.get('balances', []))} | "
            f"ER: {len(datos.get('resultados', []))} | "
            f"Periodicidad: {datos['periodicidad']}"
        )
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

# ----------------------------
# Sidebar - navegación
# ----------------------------
# ----------------------------
# Sidebar - navegación dinámica FinanSys
# ----------------------------

# Estilos animados para sidebar
st.sidebar.markdown("""
<style>
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #00eaff;
        text-align: center;
        margin-bottom: 15px;
        text-shadow: 0 0 8px rgba(0,234,255,0.6);
        animation: pulseGlow 3s infinite ease-in-out;
    }

    @keyframes pulseGlow {
        0% { text-shadow: 0 0 6px rgba(0,234,255,0.4); }
        50% { text-shadow: 0 0 14px rgba(0,234,255,0.9); }
        100% { text-shadow: 0 0 6px rgba(0,234,255,0.4); }
    }

    .mode-chip {
        text-align:center;
        padding:8px;
        margin-bottom:10px;
        border-radius:10px;
        background: rgba(0,255,255,0.08);
        border: 1px solid rgba(0,255,255,0.15);
        font-size: 14px;
        color: #8fe9ff;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>📊 FinanSys</div>", unsafe_allow_html=True)

# Inicializar modo
if "modo_datos" not in st.session_state:
    st.session_state.modo_datos = None

modo = st.session_state.modo_datos

# Mostrar modo activo
if modo == "usar":
    st.sidebar.markdown("<div class='mode-chip'>📂 Modo: Estados guardados</div>", unsafe_allow_html=True)
elif modo == "cargar":
    st.sidebar.markdown("<div class='mode-chip'>📤 Modo: Subir estados</div>", unsafe_allow_html=True)
elif modo == "crear":
    st.sidebar.markdown("<div class='mode-chip'>📝 Modo: Crear manualmente</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div class='mode-chip'>🏁 Modo: Inicio</div>", unsafe_allow_html=True)

# Menú dinámico
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
        "KPIs",
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
    st.session_state.section = "Inicio"

# Resetear si cambia de modo
if st.session_state.section not in sections:
    st.session_state.section = sections[0]

# Navegación
section = st.sidebar.radio(
    "📌 Navegación",
    sections,
    index=sections.index(st.session_state.section),
    key="section_sidebar"
)

st.session_state.section = section
# Sección: Inicio
# ----------------------------
import base64
import streamlit as st

# ============================
#   SECCIÓN INICIO
# ============================
if section == "Inicio":


    # ===== CSS ÉPICO CON FONDO DE PARTICULAS =====
    st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
    }

    /* ===== CONTENEDOR HERO ===== */
    .hero-container {
        position: relative;
        padding: 50px;
        border-radius: 25px;
        background: rgba(0, 8, 18, 0.85);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(0,255,255,0.1);
        box-shadow: 0 0 40px rgba(0,255,255,0.12);
        overflow: hidden;
        z-index: 0;
    }

    /* ===== PARTICULAS DE FONDO ===== */
  .hero-container::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 25px;
    background: radial-gradient(circle, rgba(0,255,255,0.2) 0%, transparent 70%);
    box-shadow: 0 0 50px rgba(0,255,255,0.3) inset;
    z-index: -1;
    animation: floatParticles 6s linear infinite;
    background-size: 50% 50%;
    pointer-events: none;  /* ✅ Importante */
}

    @keyframes floatParticles {
        0% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
        100% { background-position: 0% 0%; }
    }

    /* ===== LOGO ANIMADO ===== */
    .logo-animated {
        width: 190px;
        display: block;
        margin: auto;
        margin-bottom: 12px;
        filter: drop-shadow(0px 0px 10px rgba(0,255,255,0.55));
        transition: transform 0.4s ease, filter 0.4s ease;
        animation: float 4s ease-in-out infinite;
    }
    .logo-animated:hover {
        transform: scale(1.10);
        filter: drop-shadow(0px 0px 18px rgba(0,255,255,0.9));
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    /* ===== TITULOS ===== */
    .title-hero {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 1px;
        animation: fadeInUp 1.5s ease-out;
    }
    .title-glow {
        color: #00eaff;
        text-shadow: 0px 0px 18px rgba(0,230,255,0.9);
    }
    .subtitle-hero {
        text-align: center;
        color: #b8dcff;
        font-size: 20px;
        margin-top: -8px;
        opacity: 0.85;
        animation: fadeInUp 1.7s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(25px) scale(0.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ===== MÉTRICAS ===== */
    .data-box {
        margin-top: 35px;
        text-align: center;
        animation: fadeInUp 1.9s ease-out;
    }
    .metric-box {
        display: inline-block;
        padding: 20px 35px;
        background: rgba(0,255,255,0.07);
        border-radius: 18px;
        border: 1px solid rgba(0,255,255,0.15);
        backdrop-filter: blur(6px);
        box-shadow: 0 0 20px rgba(0,255,255,0.15);
        animation: pulseGlow 4s infinite ease-in-out;
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 12px rgba(0,255,255,0.4); }
        50% { box-shadow: 0 0 25px rgba(0,255,255,0.7); }
        100% { box-shadow: 0 0 12px rgba(0,255,255,0.4); }
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== CONTENIDO HERO =====
    st.markdown("<div class='hero-container'>", unsafe_allow_html=True)

    # ---- LOGO ----
    st.markdown(
        f"<img src='data:image/png;base64,{base64.b64encode(logo_bytes).decode()}' class='logo-animated'>",
        unsafe_allow_html=True
    )

    # ---- TITULOS ----
    st.markdown(
        "<h1 class='title-hero'>Bienvenido a <span class='title-glow'>FinanSys</span></h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle-hero'>Tu plataforma profesional de análisis financiero</div>",
        unsafe_allow_html=True
    )

    # ---- MÉTRICAS ----
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)

    if st.session_state.get("balances"):
        name, df = st.session_state.balances[-1]
        bal_map = map_accounts(df, ACCOUNT_MAP_BALANCE)
        ta = safe_get(bal_map, "ACTIVO_TOTAL")

        st.markdown(
            f"<div class='metric-box'>"
            f"<h3 style='color:white; margin:0;'>Activo Total ({name})</h3>"
            f"<h2 style='color:#00f6ff; margin:0; font-size:30px;'>${ta:,.2f}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='color:#7f8fa6;'>Aún no has cargado archivos financieros</div>",
            unsafe_allow_html=True
        )
    st.markdown("<div style='margin-top:40px; text-align:center;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00eaff;'>¿Cómo deseas trabajar?</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📂 Usar estados guardados", use_container_width=True):
           st.session_state.modo_datos = "usar"
           st.session_state.section = "Usar estados guardados"
           st.rerun()

    with col2:
        if st.button("📤 Subir estados nuevos", use_container_width=True):
           st.session_state.modo_datos = "cargar"
           st.session_state.section = "Cargar archivos"
           st.rerun()

    with col3:
        if st.button("📝 Crear estados manualmente", use_container_width=True):
           st.session_state.modo_datos = "crear"
           st.session_state.section = "Crear estados"
           st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
   

    st.markdown("</div></div>", unsafe_allow_html=True)

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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>📂 Consultar Estados Financieros Guardados</h3>", unsafe_allow_html=True)

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

    st.markdown("## 📊 Análisis Vertical Financiero")
    st.markdown("Análisis estructural de Estados Financieros seleccionados")

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero debes seleccionar estados en: Usar estados guardados.")
        st.stop()

    empresa = datos.get("empresa")

    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])

    # ======================================================
    # FUNCIÓN NORMALIZADORA
    # ======================================================
    def normalizar_df(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        return df

    # ======================================================
    #  SECCIÓN BALANCES GENERALES
    # ======================================================
    st.markdown("---")
    st.markdown("## 📘 Balances Generales seleccionados")

    if not balances:
        st.info("No hay balances cargados.")
    else:

        tabs = st.tabs([f"📅 {df['Año'].iloc[0]}" for df in balances])

        for i, (tab, bg) in enumerate(zip(tabs, balances)):

            with tab:

                anio = bg["Año"].iloc[0]
                bg = normalizar_df(bg)

                df_v = vertical_analysis(f"BG {anio}", bg, "BG")

                if df_v is None:
                    st.warning(f"⚠ No se pudo procesar Balance {anio}")
                    continue

                col1, col2 = st.columns([1.3, 1])

                # ===========================
                # TABLA PROFESIONAL
                # ===========================
                with col1:
                    st.markdown(f"### 🧾 Estructura del Balance – {anio}")

                    display_df = df_v[["Cuenta", "Monto", "Porcentaje"]].copy()

                    display_df["Monto"] = display_df["Monto"].map(lambda x: f"${x:,.2f}")
                    display_df["Porcentaje"] = display_df["Porcentaje"].map(lambda x: f"{x:.2f}%")

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=350
                    )

                # ===========================
                # GRÁFICO PIE MODE PREMIUM
                # ===========================
                with col2:

                    df_plot = df_v[
                        ~df_v["Cuenta_norm"].str.contains(
                            r"\btotal\b|\bsubtotal\b|\bsuma\b",
                            regex=True,
                            case=False,
                            na=False
                        )
                    ]

                    df_plot = df_plot.sort_values("Monto", ascending=False).head(10)

                    fig = px.pie(
                        df_plot,
                        names="Cuenta",
                        values="Monto",
                        hole=0.45,
                        title=f"Estructura de Activos {empresa} ({anio})"
                    )

                    fig.update_traces(
                        textinfo="percent+label",
                        pull=[0.04] * len(df_plot),
                        insidetextorientation='radial'
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=380,
                        showlegend=True,
                        margin=dict(t=60, b=0, l=0, r=0)
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    #  SECCIÓN ESTADOS DE RESULTADOS
    # ======================================================
    st.markdown("---")
    st.markdown("## 📗 Estados de Resultados seleccionados")

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

                # ===========================
                # TABLA PRO
                # ===========================
                with col1:
                    st.markdown(f"### 🧾 Estructura del ER – {anio}")

                    display_df = df_v[["Cuenta", "Monto", "Porcentaje"]].copy()

                    display_df["Monto"] = display_df["Monto"].map(lambda x: f"${x:,.2f}")
                    display_df["Porcentaje"] = display_df["Porcentaje"].map(lambda x: f"{x:.2f}%")

                    st.dataframe(display_df, use_container_width=True, height=350)

                # ===========================
                # GRÁFICO BARRAS HORIZONTAL
                # ===========================
                with col2:

                    df_graf = df_v[
                        df_v["Cuenta"].str.contains(
                            "venta|ingreso|costo|utilidad|gasto",
                            case=False,
                            na=False
                        )
                    ].sort_values("Porcentaje", ascending=False).head(10)

                    fig = px.bar(
                        df_graf,
                        x="Porcentaje",
                        y="Cuenta",
                        orientation="h",
                        text=df_graf["Porcentaje"].map(lambda x: f"{x:.2f}%"),
                        title=f"Contribución Estructural - {empresa} ({anio})"
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=380,
                        xaxis_title="%",
                        yaxis_title="Cuenta",
                        margin=dict(t=60, b=0, l=0, r=0)
                    )

                    fig.update_traces(textposition="outside")

                    st.plotly_chart(fig, use_container_width=True)

elif section == "Análisis horizontal":

    import plotly.express as px
    import numpy as np
    import streamlit as st

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>📈 Análisis Horizontal</h2>", unsafe_allow_html=True)
    st.markdown("Comparación financiera entre períodos seleccionados")

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona balances en 'Usar estados guardados'.")
        st.stop()

    empresa = datos.get("empresa")

    # ---------------------------
    # 📦 OBTENER BALANCES
    # ---------------------------
    balances = datos.get("balances")

    # Compatibilidad con versión antigua
    if balances is None:
        bg = datos.get("bg")
        anio = datos.get("anio")

        if bg is not None:
            balances = [(str(anio), bg)]
        else:
            balances = []

    # Normalizar formato [(periodo, dataframe)]
    balances_limpios = []

    for item in balances:
        if isinstance(item, tuple):
            balances_limpios.append(item)
        else:
            if "Año" in item.columns:
                anio = str(item["Año"].iloc[0])
            else:
                anio = "Periodo"
            balances_limpios.append((anio, item))

    balances = balances_limpios

    if len(balances) < 2:
        st.warning("⚠ Necesitas al menos 2 balances cargados.")
        st.stop()

    # ---------------------------
    # 🎯 SELECTORES
    # ---------------------------
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

    # ---------------------------
    # ⚙ ANÁLISIS
    # ---------------------------
    try:
        df_h = horizontal_analysis(df_inicial, df_final, periodo_inicial, periodo_final)
        df_h.columns = df_h.columns.str.strip()

        # Columnas dinámicas
        col_prev = f"{periodo_inicial} (Monto)"
        col_act = f"{periodo_final} (Monto)"

        # Validación de columnas esenciales
        required_cols = ["Cuenta", col_prev, col_act, "Variación", "Variación (%)"]

        for col in required_cols:
            if col not in df_h.columns:
                st.error(f"❌ Falta la columna requerida: {col}")
                st.write("Columnas detectadas:", df_h.columns.tolist())
                st.stop()

        # ---------------------------
        # 🧮 KPIs SUPERIORES
        # ---------------------------
        total_inicial = df_h[col_prev].sum()
        total_final = df_h[col_act].sum()
        variacion_total = total_final - total_inicial
        porcentaje_total = (variacion_total / total_inicial * 100) if total_inicial != 0 else 0

        k1, k2, k3 = st.columns(3)

        k1.metric("💰 Total Inicial", f"${total_inicial:,.0f}")
        k2.metric("💰 Total Final", f"${total_final:,.0f}")
        k3.metric("📈 Variación Total", f"{porcentaje_total:.2f}%", delta=f"${variacion_total:,.0f}")

        st.markdown("---")

        # ---------------------------
        # ✅ VALIDACIÓN CONTABLE
        # ---------------------------
        info_balance = validar_balance(df_final)

        if info_balance["cuadra"]:
            st.success("✅ El balance CUADRA correctamente.")
        else:
            st.error(f"❌ El balance NO cuadra. Diferencia: ${info_balance['diferencia']:,.2f}")

        st.markdown("---")

        # ---------------------------
        # 📋 TABLA PRINCIPAL
        # ---------------------------
        aggrid_dark(
            df_h.fillna(""),
            height=420,
            key=f"ah_{empresa}_{periodo_inicial}_{periodo_final}"
        )

        # ---------------------------
        # 📊 GRÁFICO DE IMPACTO
        # ---------------------------
        graf_df = df_h.sort_values(
            "Variación",
            key=lambda x: abs(x),
            ascending=False
        ).head(10)

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
            title=f"🔥 Impacto por cuenta ({periodo_inicial} → {periodo_final})"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Variación monetaria",
            yaxis_title="Cuenta",
            showlegend=True
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ---------------------------
        # 🚨 CRECIMIENTO PELIGROSO
        # ---------------------------
        peligrosas = detectar_cuentas_peligrosas(df_h)

        if not peligrosas.empty:
            st.warning("🚨 Cuentas con crecimiento peligroso detectadas:")
            aggrid_dark(peligrosas, height=260, key="riesgo_crecimiento")

        st.markdown("---")

        # ---------------------------
        # 📊 ANÁLISIS POR CATEGORÍA
        # ---------------------------
        df_h["Categoría"] = df_h["Cuenta"].apply(clasificar_categoria)

        df_cat = df_h.groupby("Categoría")["Variación"].sum().reset_index()

        fig_cat = px.bar(
            df_cat,
            x="Variación",
            y="Categoría",
            orientation="h",
            title="📊 Impacto por categoría financiera",
            template="plotly_dark"
        )

        st.plotly_chart(fig_cat, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error en análisis horizontal: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Sección: Razones financieras
# ----------------------------
elif section == "Razones financieras":

    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import numpy as np

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>📊 Razones Financieras</h2>", unsafe_allow_html=True)
    st.markdown("Evaluación del desempeño financiero basada en BG + ER (todos los periodos seleccionados)")

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona estados financieros en 'Usar estados guardados'.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])
    anios = datos.get("años", [])

    if not balances or not resultados:
        st.warning("⚠ El estado seleccionado debe tener BG y ER para calcular razones.")
        st.stop()

    # ============================
    # ⚙️ CÁLCULO MULTI-AÑO
    # ============================
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
                    "Valor": round(valor, 2)  # ✅ redondeo a 2 decimales
                })

        if len(todas_las_razones) == 0:
            st.error("❌ No se pudo calcular ninguna razón financiera.")
            st.stop()

        df_r = pd.DataFrame(todas_las_razones)
        st.session_state.ratios = df_r

        # ============================
        # 💧 KPIs SOLO DEL AÑO MÁS RECIENTE
        # ============================
        ultimo_anio = max(anios)
        df_ultimo = df_r[df_r["Periodo"] == ultimo_anio]

        col1, col2, col3 = st.columns(3)

        def obtener_valor(df, nombre):
            v = df[df["Razón"] == nombre]["Valor"]
            return float(v.iloc[0]) if not v.empty else np.nan

        liquidez = obtener_valor(df_ultimo, "Razón Circulante")
        endeudamiento = obtener_valor(df_ultimo, "Razón de Endeudamiento")
        roa = obtener_valor(df_ultimo, "ROA")

        col1.metric("💧 Razón Circulante", f"{liquidez:.2f}" if not np.isnan(liquidez) else "N/D")
        col2.metric("🏦 Endeudamiento", f"{endeudamiento:.2f}" if not np.isnan(endeudamiento) else "N/D")
        col3.metric("🚀 ROA", f"{roa*100:.2f}%" if not np.isnan(roa) else "N/D")

        st.markdown("---")

        # ============================
        # 📋 TABLA OSCURA MULTI-AÑO
        # ============================
        pivoted = df_r.pivot(
            index="Razón",
            columns="Periodo",
            values="Valor"
        )

        pivoted = pivoted.round(2)

        aggrid_dark(
            pivoted.reset_index(),
            height=450,
            key=f"rf_{empresa}_multi"
        )

        # ============================
        # 📊 GRÁFICO COMPARATIVO POR AÑO
        # ============================
        fig = px.line(
            df_r,
            x="Periodo",
            y="Valor",
            color="Razón",
            markers=True,
            title=f"📈 Evolución de Razones Financieras - {empresa}"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Año",
            yaxis_title="Valor de la razón",
            legend_title="Razones"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error al calcular razones financieras: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Sección: DuPont
# ----------------------------
elif section == "DuPont":

    import pandas as pd
    import numpy as np
    import streamlit as st

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>📊 Análisis DuPont</h2>", unsafe_allow_html=True)
    st.markdown("Desglose del ROE por periodo usando BG + ER")

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("⚠ Primero selecciona estados financieros en 'Usar estados guardados'.")
        st.stop()

    empresa = datos.get("empresa")
    balances = datos.get("balances", [])
    resultados = datos.get("resultados", [])

    if not balances or not resultados:
        st.warning("⚠ DuPont requiere BG y ER cargados.")
        st.stop()

    # ============================
    # Emparejar BG y ER por año
    # ============================
    resultados_dict = {er["Año"].iloc[0]: er for er in resultados}

    dfs_dupont = []

    for bg in balances:

        anio = bg["Año"].iloc[0]

        if anio not in resultados_dict:
            st.warning(f"⚠ El año {anio} no tiene estado de resultados, se omitió.")
            continue

        er = resultados_dict[anio]

        try:
            df_dup = compute_dupont(bg, er)

            df_dup["Periodo"] = anio
            dfs_dupont.append(df_dup)

        except Exception as e:
            st.error(f"❌ Error calculando DuPont para {anio}: {str(e)}")

    if not dfs_dupont:
        st.error("❌ No se pudo calcular DuPont para ningún periodo.")
        st.stop()

    # ============================
    # Consolidar todos los años
    # ============================
    df_total = pd.concat(dfs_dupont)

    # Redondeo a 2 decimales
    df_total["Valor"] = df_total["Valor"].astype(float).round(2)

    # ============================
    # 📌 KPIs SUPERIORES (último año)
    # ============================
    ultimo_anio = df_total["Periodo"].max()
    df_ultimo = df_total[df_total["Periodo"] == ultimo_anio]

    roe = df_ultimo[df_ultimo["Componente"].str.contains("ROE")]["Valor"].values[0]
    margen = df_ultimo[df_ultimo["Componente"].str.contains("Margen Neto")]["Valor"].values[0]
    rotacion = df_ultimo[df_ultimo["Componente"].str.contains("Rotación")]["Valor"].values[0]
    apalancamiento = df_ultimo[df_ultimo["Componente"].str.contains("Apalancamiento")]["Valor"].values[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📈 ROE", f"{roe:.2f}")
    c2.metric("💰 Margen Neto", f"{margen:.2f}")
    c3.metric("🔁 Rotación Activos", f"{rotacion:.2f}")
    c4.metric("🏦 Apalancamiento", f"{apalancamiento:.2f}")

    st.markdown("---")

    # ============================
    # 🎨 Tabla oscura y elegante
    # ============================
    pivot = df_total.pivot(
        index="Componente",
        columns="Periodo",
        values="Valor"
    ).reset_index()

    aggrid_dark(
        pivot,
        height=350,
        key=f"dupont_{empresa}"
    )

    # ============================
    # 📊 Gráfico DuPont por año
    # ============================
    import plotly.express as px

    fig = px.line(
        df_total,
        x="Periodo",
        y="Valor",
        color="Componente",
        markers=True,
        title=f"📊 Evolución DuPont - {empresa}"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        yaxis_title="Valor",
        xaxis_title="Periodo"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 📥 Exportar DuPont
    # ============================
    excel_bytes = df_to_excel_bytes({
        f"DuPont_{empresa}": pivot
    })

    st.download_button(
        label="📥 Exportar DuPont (Excel)",
        data=excel_bytes,
        file_name=f"DuPont_{empresa}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("</div>", unsafe_allow_html=True)
# Sección: Flujo de Efectivo
# ----------------------------
elif section == "Flujo de efectivo":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Flujo de Efectivo - Indirecto y Directo</h3>", unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("Primero selecciona un estado financiero en 'Usar estados guardados'.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    empresa = datos.get("empresa")
    anio = datos.get("anio")
    bg_actual = datos.get("bg")
    er_actual = datos.get("er")

    if bg_actual is None or er_actual is None:
        st.warning("El estado seleccionado necesita BG y ER para el flujo.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ==========================
    # Buscar periodo anterior
    # ==========================
    db = st.session_state.db_estados

    anios_empresa = sorted(db[empresa].keys())

    if anio not in anios_empresa:
        st.error("El año seleccionado no está correctamente registrado.")
        st.stop()

    idx_actual = anios_empresa.index(anio)

    if idx_actual == 0:
        st.warning("No hay periodo anterior para calcular flujo de efectivo.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    anio_anterior = anios_empresa[idx_actual - 1]

    # Tomar BG anterior (Anual)
    bg_anterior = db[empresa][anio_anterior]["Anual"].get("BG")

    if bg_anterior is None:
        st.warning(f"No hay Balance General para {anio_anterior}.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.success(f"Comparando periodos: {anio_anterior} → {anio}")

    # ==========================
    # CÁLCULO FLUJO INDIRECTO
    # ==========================
    df_efe_ind = compute_cashflow_indirect(
        bg_actual,
        er_actual,
        bg_anterior
    )

    st.markdown("### Método Indirecto")
    aggrid_dark(df_efe_ind, height=260, key=f"efe_ind_{empresa}_{anio}")

    # ==========================
    # CÁLCULO FLUJO DIRECTO
    # ==========================
    df_efe_dir = compute_cashflow_direct(bg_actual, er_actual)

    st.markdown("### Método Directo")
    aggrid_dark(df_efe_dir, height=220, key=f"efe_dir_{empresa}_{anio}")

    # ==========================
    # KPIs del Flujo
    # ==========================
    try:
        A = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Operación (A)","Monto"].sum())
        B = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Inversión (B)","Monto"].sum())
        C = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Financiamiento (C)","Monto"].sum())
        N = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo Neto (A+B+C)","Monto"].sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flujo Operativo (A)", f"${A:,.2f}")
        k2.metric("Flujo Inversión (B)", f"${B:,.2f}")
        k3.metric("Flujo Financiamiento (C)", f"${C:,.2f}")
        k4.metric("Flujo Neto", f"${N:,.2f}")

    except Exception as e:
        st.error(f"Error calculando KPIs: {e}")

    # ==========================
    # Gráfico
    # ==========================
    chart_df = pd.DataFrame({
        "Actividad": ["Operación", "Inversión", "Financiamiento"],
        "Monto": [A, B, C]
    })

    fig = px.bar(
        chart_df,
        x="Actividad",
        y="Monto",
        text="Monto",
        color="Actividad",
        color_discrete_sequence=COLORS,
        title=f"Flujos — {empresa} {anio_anterior} → {anio}"
    )

    fig.update_layout(template='plotly_dark', height=420)
    fig.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Exportaciones
    # ==========================
    export_name_base = f"Flujo_{empresa}_{anio_anterior}_to_{anio}"

    excel_bytes = df_to_excel_bytes({
        "Flujo_Indirecto": df_efe_ind,
        "Flujo_Directo": df_efe_dir
    })

    st.download_button(
        "Exportar Flujo (Excel)",
        data=excel_bytes,
        file_name=f"{export_name_base}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    csv_ind = df_to_csv_bytes(df_efe_ind)
    csv_dir = df_to_csv_bytes(df_efe_dir)

    st.download_button(
        "Exportar Indirecto (CSV)",
        data=csv_ind,
        file_name=f"{export_name_base}_indirecto.csv",
        mime="text/csv"
    )

    st.download_button(
        "Exportar Directo (CSV)",
        data=csv_dir,
        file_name=f"{export_name_base}_directo.csv",
        mime="text/csv"
    )

    # ==========================
    # HTML Reporte
    # ==========================
    html_report = df_to_html_report(
        f"Estado de Flujo - {empresa} ({anio_anterior} → {anio})",
        {
            "Flujo Indirecto": df_efe_ind,
            "Flujo Directo": df_efe_dir,
            "KPIs": pd.DataFrame([
                {"Concepto":"Flujo Operativo (A)","Monto":A},
                {"Concepto":"Flujo Inversión (B)","Monto":B},
                {"Concepto":"Flujo Financiamiento (C)","Monto":C},
                {"Concepto":"Flujo Neto","Monto":N}
            ])
        }
    )

    st.download_button(
        "Exportar Informe (HTML, imprimir a PDF)",
        data=html_report.encode("utf-8"),
        file_name=f"{export_name_base}.html",
        mime="text/html"
    )

    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Sección: Estado de Origen y Aplicación
# ----------------------------

elif section == "Origen y Aplicación de Fondos":
    import traceback

    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader'>Estado de Origen y Aplicación de Fondos (EOAF)</h3>", unsafe_allow_html=True)

        datos = st.session_state.get("estado_seleccionado")

        if not datos:
            st.warning("Primero selecciona un estado financiero en 'Usar estados guardados'.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        empresa = datos.get("empresa")
        anio = datos.get("anio")
        bg_actual = datos.get("bg")
        er_actual = datos.get("er")

        if bg_actual is None:
            st.warning("El estado seleccionado no tiene Balance General.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # ==========================
        # Buscar periodo anterior
        # ==========================
        db = st.session_state.db_estados
        anios_empresa = sorted(db[empresa].keys())

        if anio not in anios_empresa:
            st.error("El año seleccionado no es válido.")
            st.stop()

        idx_actual = anios_empresa.index(anio)

        if idx_actual == 0:
            st.warning("No existe periodo anterior para esta empresa.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        anio_anterior = anios_empresa[idx_actual - 1]

        bg_anterior = db[empresa][anio_anterior]["Anual"].get("BG")

        if bg_anterior is None:
            st.warning(f"No existe Balance General para {anio_anterior}.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.success(f"Comparando {empresa}: {anio_anterior} → {anio}")

        # ==========================
        # Estado de resultados opcional
        # ==========================
        if er_actual is None:
            st.warning("No hay Estado de Resultados, se calculará solo con variaciones del BG.")
            er_df = None
        else:
            er_df = er_actual

        # ==========================
        # Cálculo EOAF
        # ==========================
        try:
            df_origen, df_aplic, df_resumen = compute_eoaf(
                bg_anterior,
                bg_actual,
                er_df
            )
        except Exception:
            st.error("Error en compute_eoaf(). Traza:")
            st.code(traceback.format_exc())
            st.stop()

        # ==========================
        # Mostrar tablas
        # ==========================
        st.markdown("### Orígenes de Fondos")
        aggrid_dark(
            df_origen if df_origen is not None and not df_origen.empty 
            else pd.DataFrame(columns=["Concepto", "Monto"]),
            height=260,
            key=f"eoaf_origen_{empresa}_{anio}"
        )

        st.markdown("### Aplicaciones de Fondos")
        aggrid_dark(
            df_aplic if df_aplic is not None and not df_aplic.empty 
            else pd.DataFrame(columns=["Concepto", "Monto"]),
            height=260,
            key=f"eoaf_aplic_{empresa}_{anio}"
        )

        st.markdown("### Resumen EOAF")
        aggrid_dark(
            df_resumen if df_resumen is not None and not df_resumen.empty 
            else pd.DataFrame(columns=["Tipo", "Monto"]),
            height=140,
            key=f"eoaf_resumen_{empresa}_{anio}"
        )

        # ==========================
        # Exportaciones
        # ==========================
        export_name = f"EOAF_{empresa}_{anio_anterior}_to_{anio}"

        excel_bytes = df_to_excel_bytes({
            "Orígenes": df_origen if df_origen is not None else pd.DataFrame(columns=["Concepto", "Monto"]),
            "Aplicaciones": df_aplic if df_aplic is not None else pd.DataFrame(columns=["Concepto", "Monto"]),
            "Resumen": df_resumen if df_resumen is not None else pd.DataFrame(columns=["Tipo", "Monto"])
        })

        st.download_button(
            "Exportar EOAF (Excel)",
            data=excel_bytes,
            file_name=f"{export_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            "Exportar Orígenes (CSV)",
            data=df_to_csv_bytes(
                df_origen if df_origen is not None 
                else pd.DataFrame(columns=["Concepto","Monto"])
            ),
            file_name=f"{export_name}_Origenes.csv"
        )

        st.download_button(
            "Exportar Aplicaciones (CSV)",
            data=df_to_csv_bytes(
                df_aplic if df_aplic is not None 
                else pd.DataFrame(columns=["Concepto","Monto"])
            ),
            file_name=f"{export_name}_Aplicaciones.csv"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception:
        st.error("⚠ Ocurrió un error inesperado en la sección EOAF.")
        st.code(traceback.format_exc())
# ----------------------------
# Sección: KPIs (Dashboard de indicadores)
# ----------------------------
elif section == "KPIs":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Panel de Indicadores (KPIs)</h3>", unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("Primero selecciona un estado financiero en 'Usar estados guardados'.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    empresa = datos.get("empresa")
    anio = datos.get("anio")
    bg_actual = datos.get("bg")
    er_actual = datos.get("er")

    if bg_actual is None or er_actual is None:
        st.warning("El estado seleccionado debe tener Balance General y Estado de Resultados.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    db = st.session_state.db_estados
    anios_empresa = sorted(db[empresa].keys())

    if anio not in anios_empresa:
        st.error("El año seleccionado no es válido.")
        st.stop()

    idx = anios_empresa.index(anio)

    if idx == 0:
        st.warning("No hay periodo anterior para comparar KPIs.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ==========================
    # Periodo anterior
    # ==========================
    anio_prev = anios_empresa[idx - 1]
    bg_prev = db[empresa][anio_prev]["Anual"].get("BG")
    er_prev = db[empresa][anio_prev]["Anual"].get("ER")

    if bg_prev is None or er_prev is None:
        st.warning(f"No hay BG o ER completo para {anio_prev}.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.success(f"Comparando KPIs: {empresa} {anio_prev} → {anio}")

    # ==========================
    # Cálculo de ratios
    # ==========================
    ratios_actual = compute_ratios(bg_actual, er_actual)
    ratios_prev = compute_ratios(bg_prev, er_prev)

    # ==========================
    # Tarjetas principales
    # ==========================
    c1, c2, c3, c4 = st.columns(4)

    def delta(metric):
        return round(ratios_actual.get(metric, 0) - ratios_prev.get(metric, 0), 2)

    c1.metric(
        "Razón Corriente",
        f"{ratios_actual.get('Razón Circulante','-')}",
        f"Δ {delta('Razón Circulante')}"
    )

    c2.metric(
        "Razón Rápida",
        f"{ratios_actual.get('Razón Rápida','-')}",
        f"Δ {delta('Razón Rápida')}"
    )

    c3.metric(
        "Capital Neto de Trabajo",
        f"{ratios_actual.get('Capital Neto de Trabajo','-')}",
        f"Δ {delta('Capital Neto de Trabajo')}"
    )

    c4.metric(
        "ROA (%)",
        f"{ratios_actual.get('ROA (%)','-')}",
        f"Δ {delta('ROA (%)')}"
    )

    # ==========================
    # Gráfico márgenes
    # ==========================
    st.markdown("#### Márgenes del periodo seleccionado")

    chart = pd.DataFrame({
        "Métrica": ["Margen Bruto", "Margen Operativo", "Margen Neto"],
        "Valor (%)": [
            ratios_actual.get("Margen Utilidad Bruta (%)", 0),
            ratios_actual.get("Margen Utilidad Operativa (%)", 0),
            ratios_actual.get("Margen Utilidad Neta (%)", 0)
        ]
    })

    fig = px.bar(
        chart,
        x="Métrica",
        y="Valor (%)",
        text="Valor (%)",
        color="Métrica",
        color_discrete_sequence=COLORS
    )

    fig.update_layout(template='plotly_dark', height=380)
    fig.update_traces(textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Export KPIs
    # ==========================
    df_kpis = pd.DataFrame.from_dict(
        ratios_actual,
        orient="index",
        columns=["Valor"]
    ).reset_index().rename(columns={"index": "Indicador"})

    st.download_button(
        "Exportar KPIs (CSV)",
        data=df_to_csv_bytes(df_kpis),
        file_name=f"KPIs_{empresa}_{anio}.csv",
        mime="text/csv"
    )

    st.download_button(
        "Exportar KPIs (Excel)",
        data=df_to_excel_bytes({"KPIs": df_kpis}),
        file_name=f"KPIs_{empresa}_{anio}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Sección: Interpretación IA
# ----------------------------
elif section == "Interpretación":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Interpretación Ejecutiva</h3>", unsafe_allow_html=True)

    datos = st.session_state.get("estado_seleccionado")

    if not datos:
        st.warning("Primero selecciona un estado financiero en 'Usar estados guardados'.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    empresa = datos.get("empresa")
    anio = datos.get("anio")
    bg = datos.get("bg")
    er = datos.get("er")

    if bg is None or er is None:
        st.warning("El estado seleccionado debe tener BG y ER para generar interpretación.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ==========================
    # Resumen base
    # ==========================
    summary = f"Resumen financiero de {empresa} para el periodo {anio}:\n\n"

    # ==========================
    # Análisis vertical BG
    # ==========================
    try:
        dv_bg = vertical_analysis(empresa, bg, 'BG')
        summary += f"Análisis vertical del Balance General ({anio}) - principales cuentas:\n"

        resumen_bg = dv_bg[['Cuenta', 'Porcentaje']]\
            .dropna()\
            .sort_values("Porcentaje", ascending=False)\
            .head(6)

        summary += resumen_bg.to_string(index=False) + "\n\n"

    except Exception as e:
        summary += f"Error en análisis vertical BG: {e}\n\n"

    # ==========================
    # Análisis vertical ER
    # ==========================
    try:
        dv_er = vertical_analysis(empresa, er, 'ER')
        summary += f"Análisis vertical del Estado de Resultados ({anio}) - principales cuentas:\n"

        resumen_er = dv_er[['Cuenta', 'Porcentaje']]\
            .dropna()\
            .sort_values("Porcentaje", ascending=False)\
            .head(6)

        summary += resumen_er.to_string(index=False) + "\n\n"

    except Exception as e:
        summary += f"Error en análisis vertical ER: {e}\n\n"

    # ==========================
    # KPIs del periodo seleccionado
    # ==========================
    try:
        ratios_actual = compute_ratios(bg, er)
        df_kpis = pd.DataFrame.from_dict(
            ratios_actual,
            orient="index",
            columns=["Valor"]
        ).reset_index().rename(columns={"index": "Indicador"})

        summary += "Indicadores financieros clave:\n"
        summary += df_kpis.head(12).to_string(index=False)

    except Exception as e:
        summary += f"\nError obteniendo KPIs: {e}"

    # ==========================
    # Mostrar preview
    # ==========================
    st.code(summary[:3500], language="")

    # ==========================
    # Generar con IA
    # ==========================
    if st.button("Generar interpretación"):
        with st.spinner("Generando interpretación ..."):
            text = generate_interpretation_gemini(summary)

            if text.lower().startswith("error") or "no se" in text.lower():
                st.error(text)
            else:
                st.session_state.ia_interpretation = text
                st.success("Interpretación generada correctamente.")

    if st.session_state.get("ia_interpretation"):
        st.markdown("#### 📊 Informe de Interpretación")
        st.markdown(st.session_state.ia_interpretation)

    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr class='st-sep'/>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 FinanSys. Todos los derechos reservados.</div>", unsafe_allow_html=True)
