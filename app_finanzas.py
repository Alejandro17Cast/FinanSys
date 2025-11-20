# app.py - FinanSys (versión completa con export y flujo directo/indirecto y KPIs)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
from google import genai
import io 
import os
import re
import base64
import traceback
import db
from datetime import datetime
db.init_db()

# ----------------------------
# Config página y paleta
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

def vertical_analysis(name: str, df: pd.DataFrame, ttype: str) -> pd.DataFrame:
    dfv = df.copy()
    dfv["Cuenta_norm"] = dfv["Cuenta"].apply(norm_account_name)
    dfv["Monto"] = to_numeric_series(dfv["Monto"])
    if ttype == "BG":
        base = detect_base_balance(dfv)
    else:
        base = detect_base_results(dfv)
    exclude_mask = dfv["Cuenta_norm"].str.contains(r"\btotal\b|\bsubtotal\b|\bsuma\b", regex=True, na=False)
    dfv["Porcentaje"] = np.nan
    if base == 0 or np.isnan(base):
        dfv["Porcentaje"] = np.nan
    else:
        # compute percentages only for non-total rows
        dfv.loc[~exclude_mask, "Porcentaje"] = (dfv.loc[~exclude_mask, "Monto"] / base * 100).round(2)
        dfv.loc[exclude_mask, "Porcentaje"] = np.nan
    dfv["Monto"] = dfv["Monto"].round(2)
    # order by absolute magnitude
    try:
        dfv = dfv.reindex(dfv["Monto"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    except Exception:
        dfv = dfv.reset_index(drop=True)
    return dfv

def horizontal_analysis(df_prev: pd.DataFrame, df_act: pd.DataFrame, period_prev: str, period_act: str) -> pd.DataFrame:
    dp = df_prev.copy(); da = df_act.copy()
    dp["Cuenta_norm"] = dp["Cuenta"].apply(norm_account_name); da["Cuenta_norm"] = da["Cuenta"].apply(norm_account_name)
    dp["Monto"] = to_numeric_series(dp["Monto"]); da["Monto"] = to_numeric_series(da["Monto"])
    merged = da.merge(dp[["Cuenta_norm", "Monto"]].rename(columns={"Monto": "Monto_Anterior"}), on="Cuenta_norm", how="left")
    merged["Monto_Anterior"] = merged["Monto_Anterior"].fillna(0.0)
    merged[f"{period_act} (Monto)"] = merged["Monto"].round(2)
    merged[f"{period_prev} (Monto)"] = merged["Monto_Anterior"].round(2)
    merged["Variación Absoluta"] = (merged[f"{period_act} (Monto)"] - merged[f"{period_prev} (Monto)"]).round(2)
    prev_series = merged[f"{period_prev} (Monto)"].replace({0.0: np.nan})
    merged["Variación Relativa (%)"] = ((merged[f"{period_act} (Monto)"] / prev_series - 1) * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0).round(2)
    return merged[["Cuenta", f"{period_prev} (Monto)", f"{period_act} (Monto)", "Variación Absoluta", "Variación Relativa (%)"]]

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

def compute_ratios(balance_df: pd.DataFrame, results_df: pd.DataFrame) -> dict:
    bal = map_accounts(balance_df, ACCOUNT_MAP_BALANCE)
    res = map_accounts(results_df, ACCOUNT_MAP_RESULTS)
    r = {}
    ac = safe_get(bal, "ACTIVO_CORRIENTE"); pc = safe_get(bal, "PASIVO_CORRIENTE"); inv = safe_get(bal, "INVENTARIO")
    ventas = safe_get(res, "VENTAS_NETAS"); cv = safe_get(res, "COSTO_VENTAS"); cc = safe_get(bal, "CUENTAS_POR_COBRAR")
    at = safe_get(bal, "ACTIVO_TOTAL"); af = safe_get(bal, "ACTIVO_FIJO"); pt = safe_get(bal, "PASIVO_TOTAL"); pat = safe_get(bal, "PATRIMONIO")
    uo = safe_get(res, "UTILIDAD_OPERATIVA"); gi = safe_get(res, "GASTO_INTERESES"); ub = safe_get(res, "UTILIDAD_BRUTA"); un = safe_get(res, "UTILIDAD_NETA")
    eps = 1e-9
    # safe divisions - using max with eps
    r["Razón Circulante"] = round(ac / max(pc, eps), 2)
    r["Razón Rápida"] = round((ac - inv) / max(pc, eps), 2)
    CNT, CNO = calc_cnt_cno(balance_df)
    r["Capital Neto de Trabajo"] = CNT
    r["Capital Neto Operativo"] = CNO
    r["Rotación Inventarios (veces)"] = round(cv / max(inv, eps), 2)
    r["Rotación Cuentas por Cobrar (veces)"] = round(ventas / max(cc, eps), 2)
    r["Periodo Promedio Cobro (días)"] = round(360 / max(r["Rotación Cuentas por Cobrar (veces)"], eps), 2)
    r["Rotación Activos Fijos (veces)"] = round(ventas / max(af, eps), 2)
    r["Rotación Activos Totales (veces)"] = round(ventas / max(at, eps), 2)
    r["Razón de Endeudamiento (Pasivo/Activo)"] = round(pt / max(at, eps), 2)
    r["Razón Pasivo / Capital"] = round(pt / max(pat, eps), 2)
    r["Cobertura Intereses (veces)"] = round(uo / max(gi, eps), 2)
    r["Margen Utilidad Bruta (%)"] = round((ub / max(ventas, eps)) * 100, 2)
    r["Margen Utilidad Operativa (%)"] = round((uo / max(ventas, eps)) * 100, 2)
    r["Margen Utilidad Neta (%)"] = round((un / max(ventas, eps)) * 100, 2)
    r["ROA (%)"] = round((un / max(at, eps)) * 100, 2)
    return r

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

def compute_cashflow_indirect(balance_df_act: pd.DataFrame, results_df_act: pd.DataFrame, balance_df_prev: pd.DataFrame) -> pd.DataFrame:
    bal_a = map_accounts(balance_df_act, ACCOUNT_MAP_BALANCE); res_a = map_accounts(results_df_act, ACCOUNT_MAP_RESULTS); bal_p = map_accounts(balance_df_prev, ACCOUNT_MAP_BALANCE)
    un = safe_get(res_a, "UTILIDAD_NETA"); depr = safe_get(res_a, "DEPRECIACION")
    cc_a = safe_get(bal_a, "CUENTAS_POR_COBRAR"); cc_p = safe_get(bal_p, "CUENTAS_POR_COBRAR")
    inv_a = safe_get(bal_a, "INVENTARIO"); inv_p = safe_get(bal_p, "INVENTARIO")
    prov_a = safe_get(bal_a, "PROVEEDORES"); prov_p = safe_get(bal_p, "PROVEEDORES")
    cambio_cc = cc_a - cc_p; cambio_inv = inv_a - inv_p; cambio_prov = prov_a - prov_p
    flujo_operativo = round(un + depr - cambio_cc - cambio_inv + cambio_prov, 2)
    af_a = safe_get(bal_a, "ACTIVO_FIJO"); af_p = safe_get(bal_p, "ACTIVO_FIJO")
    flujo_inversion = round(- (af_a - af_p), 2)
    pt_no_op_a = safe_get(bal_a, "PASIVO_TOTAL") - safe_get(bal_a, "PROVEEDORES")
    pt_no_op_p = safe_get(bal_p, "PASIVO_TOTAL") - safe_get(bal_p, "PROVEEDORES")
    cambio_deuda_neta = round(pt_no_op_a - pt_no_op_p, 2)
    cambio_patrimonio = round(safe_get(bal_a, "PATRIMONIO") - safe_get(bal_p, "PATRIMONIO"), 2)
    flujo_financiamiento = round(cambio_deuda_neta + cambio_patrimonio, 2)
    flujo_total = round(flujo_operativo + flujo_inversion + flujo_financiamiento, 2)
    data = {
        "Concepto": ["Utilidad Neta", "Depreciación y Amortización", "Ajuste Cuentas por Cobrar", "Ajuste Inventarios", "Ajuste Proveedores",
                     "Flujo de Operación (A)", "Flujo de Inversión (B)", "Flujo de Financiamiento (C)", "Flujo Neto (A+B+C)"],
        "Monto": [round(un,2), round(depr,2), round(-cambio_cc,2), round(-cambio_inv,2), round(cambio_prov,2), flujo_operativo, flujo_inversion, flujo_financiamiento, flujo_total]
    }
    return pd.DataFrame(data)

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
def aggrid_dark(df: pd.DataFrame, height: int = 300):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(enableRowGroup=True, enableValue=True, sortable=True, filter=True, resizable=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    return AgGrid(df, gridOptions=gridOptions, theme='dark', height=height, fit_columns_on_grid_load=True)

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
# DEBUG: comprobar etiquetas exactas (ver si hay espacios o caracteres invisibles)
st.sidebar.header("Secciones - FinanSys")
sections = ["Inicio", "Crear BG / ER", "Cargar archivos", "Análisis vertical", "Análisis horizontal",
            "Razones financieras", "DuPont", "Flujo de efectivo", "Origen y Aplicación de Fondos", "KPIs", "Interpretación"]



section = st.sidebar.radio("Selecciona la sección:", sections)

# ----------------------------
# Sección: Inicio
# ----------------------------
if section == "Inicio":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns([2,1])
    with cols[0]:
        st.markdown("<h2 class='subheader'>Bienvenido a FinanSys</h2>", unsafe_allow_html=True)
        st.write("Dashboard financiero profesional")
        st.markdown("<div class='small-muted'>Consejo: sube archivos con columnas 'Cuenta' y 'Monto' (CSV/XLSX). Usa nombres claros: 'Total de Activos', 'Ventas'.</div>", unsafe_allow_html=True)
    with cols[1]:
        if st.session_state.balances:
            name, df = st.session_state.balances[-1]
            bal_map = map_accounts(df, ACCOUNT_MAP_BALANCE)
            ta = safe_get(bal_map, "ACTIVO_TOTAL")
            st.metric(label=f"Activo Total ({name})", value=f"${ta:,.2f}")
        else:
            st.markdown("<div class='card'><div class='small-muted'>Sin datos cargados aún</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Crear Estado de Resultado/ Balance General
# ----------------------------
elif section == "Crear BG / ER":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Constructor de Balance General / Estado de Resultado</h3>", unsafe_allow_html=True)

    # Selección de tipo
    tipo = st.radio("¿Qué deseas crear?", ["Balance General", "Estado de Resultado"], key="tipo_crear")

    # Selección de periodicidad
    periodicidad = st.radio("Periodicidad del estado financiero", ["Anual", "Mensual"], key="periodicidad_crear")

    # Selección de año
    año = st.number_input("Año del estado financiero:", min_value=1900, max_value=2100, value=2024, key="anio_crear")

    # Si es mensual → Seleccionar mes
    if periodicidad == "Mensual":
        meses = [
            "Enero","Febrero","Marzo","Abril","Mayo","Junio",
            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
        ]
        mes_nombre = st.selectbox("Selecciona el mes:", meses)
        mes_num = meses.index(mes_nombre) + 1
    else:
        mes_num = None

    # --------------------------------------------------------------------
    # 🔥 SELECCIONAR O CREAR EMPRESA
    # --------------------------------------------------------------------
    st.write("### Empresa")

    modo = st.radio(
        "¿Qué deseas hacer?",
        ["Seleccionar empresa existente", "Crear nueva empresa"],
        horizontal=True,
        key="seleccion_empresa"
    )

    empresas = db.obtener_empresas()
    empresas_dict = {nombre: emp_id for emp_id, nombre in empresas}

    empresa_id = None

    if modo == "Seleccionar empresa existente":
        if not empresas:
            st.warning("No hay empresas registradas. Crea una nueva.")
        else:
            empresa_nombre = st.selectbox("Selecciona la empresa:", list(empresas_dict.keys()))
            empresa_id = empresas_dict[empresa_nombre]

    else:
        st.markdown("#### Registrar nueva empresa")
        nueva_empresa = st.text_input("Nombre de la empresa", key="nueva_empresa")
        nuevo_sector = st.text_input("Sector (opcional)", key="nuevo_sector")

        if st.button("Registrar empresa", key="registrar_empresa_btn"):
            if nueva_empresa.strip() == "":
                st.error("El nombre es obligatorio.")
            else:
                from datetime import date
                db.crear_empresa(nueva_empresa.strip(), nuevo_sector.strip(), str(date.today()))
                st.success(f"Empresa '{nueva_empresa}' creada correctamente. Ya puedes seleccionarla.")
                st.rerun()


    st.write("### Cuentas predeterminadas")

    # Cuentas estándar según tipo
    cuentas_bg = {
        "Activo Corriente": ["Caja", "Bancos", "Clientes", "Inventarios", "Otros Activos Corrientes"],
        "Activo No Corriente": ["Propiedad, Planta y Equipo", "Intangibles", "Inversiones a Largo Plazo"],
        "Pasivo Corriente": ["Proveedores", "Acreedores", "Obligaciones Bancarias CP"],
        "Pasivo No Corriente": ["Préstamos LP", "Obligaciones Financieras LP"],
        "Capital Contable": ["Capital Social", "Utilidades Retenidas"]
    }

    cuentas_er = {
        "Ingresos": ["Ventas", "Ingresos Operativos", "Otros Ingresos"],
        "Costos": ["Costo de Ventas", "Costos Operativos"],
        "Gastos": ["Gastos de Administración", "Gastos de Venta", "Gastos Financieros"]
    }

    clasificaciones = cuentas_bg if tipo == "Balance General" else cuentas_er
    clasificacion = st.selectbox("Selecciona la clasificación:", list(clasificaciones.keys()), key="clasificacion_crear")

    cuenta_seleccionada = st.selectbox(
        "Selecciona una cuenta predeterminada:",
        clasificaciones[clasificacion],
        key="cuenta_predet"
    )

    nueva_cuenta = st.text_input("O escribe una cuenta nueva:", key="nueva_cuenta")

    # Monto
    st.markdown("#### Monto (al agregar la cuenta)")
    col_m1, col_m2 = st.columns([2,1])
    with col_m1:
        monto_input = st.text_input("Monto (ej.: 1200.50 o -500)", value="0", key="monto_input")

    with col_m2:
        add_pressed = st.button("➕ Agregar cuenta a la tabla", key="add_account")

    # Preparar df
    if "df_edit" not in st.session_state:
        st.session_state.df_edit = pd.DataFrame({"Cuenta": [], "Monto": []})

    # Agregar cuenta
    if add_pressed:
        cuenta_final = nueva_cuenta.strip() if nueva_cuenta.strip() else cuenta_seleccionada

        try:
            monto_val = parse_number(monto_input)
        except:
            monto_val = np.nan

        if monto_val is None or (isinstance(monto_val, float) and np.isnan(monto_val)):
            st.warning("Monto inválido.")
        else:
            st.session_state.df_edit.loc[len(st.session_state.df_edit)] = [cuenta_final, monto_val]
            st.success(f"Cuenta '{cuenta_final}' agregada.")

    # Tabla editable
    st.write("### Edita tu tabla (como Excel):")
    gb = GridOptionsBuilder.from_dataframe(st.session_state.df_edit)
    gb.configure_default_column(editable=True, resizable=True)
    gb.configure_grid_options(enableRangeSelection=True)

    grid = AgGrid(
        st.session_state.df_edit,
        gridOptions=gb.build(),
        theme="dark",
        height=300,
        update_mode="MODEL_CHANGED",
        allow_unsafe_jscode=True,
        key="aggrid_creador"
    )

    st.session_state.df_edit = grid["data"]

    if st.button("🧹 Limpiar tabla", key="limpiar_tabla"):
        st.session_state.df_edit = pd.DataFrame({"Cuenta": [], "Monto": []})
        st.success("Tabla limpiada.")

    # GUARDAR BG / ER
    if st.button("💾 Guardar estado financiero", key="guardar_estado"):

        if empresa_id is None:
            st.error("Debes seleccionar o crear una empresa antes de guardar.")
            st.stop()

        df_final = pd.DataFrame(st.session_state.df_edit).copy()

        if df_final.empty:
            st.error("La tabla no puede estar vacía.")
            st.stop()

        df_final["Cuenta"] = df_final["Cuenta"].astype(str).str.strip()
        df_final["Monto"] = to_numeric_series(df_final["Monto"])

        df_final = df_final[df_final["Cuenta"] != ""].reset_index(drop=True)

        if df_final.empty:
            st.error("No quedan cuentas válidas.")
            st.stop()

        if df_final["Monto"].isna().any():
            st.error("Hay montos inválidos.")
            st.stop()

        # Guardar en BD (con periodicidad mensual o anual)
        for _, row in df_final.iterrows():
            db.guardar_estado(
                empresa_id=empresa_id,
                tipo_estado="BG" if tipo == "Balance General" else "ER",
                periodicidad="mensual" if periodicidad == "Mensual" else "anual",
                año=año,
                mes=mes_num,
                cuenta=row["Cuenta"],
                monto=float(row["Monto"])
            )

        st.success(f"{tipo} guardado exitosamente en la base de datos.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: Cargar archivos
# ----------------------------
elif section == "Cargar archivos":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Carga tus archivos (BG y ER)</h3>", unsafe_allow_html=True)

    uploaded_bg = st.file_uploader(
        "Balances Generales (BG) - acepta múltiples",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        key="u_bg"
    )

    uploaded_er = st.file_uploader(
        "Estados de Resultados (ER) - acepta múltiples",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        key="u_er"
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

                    # Detección flexible
                    for k, orig in cols.items():
                        if k in ["cuenta", "descripcion", "nombre", "concepto", "detalle"]:
                            cuenta_col = orig
                        if k in ["monto", "saldo", "valor", "importe", "total"]:
                            monto_col = orig

                    # Detección exacta
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
                        st.warning(
                            f"⚠ Archivo {f.name} omitido: requiere columnas 'Cuenta' y 'Monto' o equivalentes."
                        )

                except Exception as e:
                    st.error(f"Error al procesar {f.name}: {e}")

        files.sort(key=lambda x: x[0])
        return files

    # -----------------------------------------------------------------
    # ❗ RESTRICCIÓN: solo mostrar botón si subieron al menos un archivo
    # -----------------------------------------------------------------
    if uploaded_bg or uploaded_er:

        # ⭐ BOTÓN ADENTRO DEL ELIF (ya no sale en otras secciones)
        if st.button("Procesar archivos", type='primary', key="procesar_archivos"):

            import time
            progress = st.progress(0)
            status = st.empty()
            status.write("Procesando archivos...")

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            # Procesar
            st.session_state.balances = process_upload_list(uploaded_bg)
            st.session_state.resultados = process_upload_list(uploaded_er)

            status.write("✔ ¡Archivos procesados!")

            # Verificaciones
            if not st.session_state.balances and not st.session_state.resultados:
                st.error("❌ Ningún archivo válido fue procesado.")
                st.stop()

            if st.session_state.balances or st.session_state.resultados:
                st.success("Archivos procesados y guardados en session_state.")

            if len(st.session_state.balances) != len(st.session_state.resultados):
                st.warning("⚠ Se recomienda igual número de BG y ER por período.")

            # Previews
            if st.session_state.balances:
                st.markdown("#### Previews - Balances")
                for name, df in st.session_state.balances[:3]:
                    st.markdown(f"**{name}**")
                    aggrid_dark(df.head(8), height=180)

            if st.session_state.resultados:
                st.markdown("#### Previews - Resultados")
                for name, df in st.session_state.resultados[:3]:
                    st.markdown(f"**{name}**")
                    aggrid_dark(df.head(8), height=180)

    else:
        st.info("Sube al menos un archivo para habilitar el botón 'Procesar archivos'.")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Sección: Análisis vertical
# ----------------------------
elif section == "Análisis vertical":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Análisis vertical (Estructura)</h3>", unsafe_allow_html=True)
    if not st.session_state.balances and not st.session_state.resultados:
        st.warning("Carga archivos en 'Cargar archivos' primero.")
    else:
        if st.session_state.balances:
            st.markdown("## Balances Generales")
            for name, df in st.session_state.balances:
                st.markdown(f"### {name}")
                df_v = vertical_analysis(name, df, 'BG')
                cols = st.columns([1,1.3])
                with cols[0]:
                    st.markdown("Tabla (montos y % — 2 decimales)")
                    display_df = df_v[["Cuenta", "Monto", "Porcentaje"]].copy()
                    display_df = display_df.rename(columns={"Monto":"Monto (USD)", "Porcentaje":"% sobre Total Activo"})
                    aggrid_dark(display_df.fillna(""), height=300)
                with cols[1]:
                    try:
                        df_plot = df_v[~df_v["Cuenta_norm"].str.contains(r"\btotal\b|\bsubtotal\b|\bsuma\b", regex=True, na=False)]
                        if not df_plot.empty:
                            fig = px.pie(df_plot.head(12), names='Cuenta', values='Monto', hole=0.35,
                                         color_discrete_sequence=COLORS, title=f'Distribución (sin totales) - {name}')
                            fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.02]*len(df_plot.head(12)))
                            fig.update_layout(template='plotly_dark', height=380, legend=dict(orientation="h",yanchor="bottom",y=-0.2))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No hay cuentas adecuadas para graficar (filtro totales/subtotales).")
                    except Exception as e:
                        st.error(f"Error gráfico: {e}")

        if st.session_state.resultados:
            st.markdown("## Estados de Resultados")
            for name, df in st.session_state.resultados:
                st.markdown(f"### {name}")
                df_v = vertical_analysis(name, df, 'ER')
                cols = st.columns([1,1.3])
                with cols[0]:
                    st.markdown("Tabla (montos y % — 2 decimales respecto a Ventas)")
                    display_df = df_v[["Cuenta","Monto","Porcentaje"]].copy()
                    display_df = display_df.rename(columns={"Monto":"Monto (USD)","Porcentaje":"% sobre Ventas"})
                    aggrid_dark(display_df.fillna(""), height=300)
                with cols[1]:
                    try:
                        df_m = df_v[~df_v["Cuenta_norm"].str.contains(r"\btotal\b|\bsubtotal\b|\bsuma\b", regex=True, na=False)]
                        df_m = df_m[df_m["Cuenta"].str.contains('venta|utilidad|costo|margen|gasto', case=False, na=False)]
                        if not df_m.empty:
                            fig = px.bar(df_m.head(12).sort_values("Porcentaje", ascending=False), x="Porcentaje", y="Cuenta", orientation='h',
                                         text=df_m["Porcentaje"].map(lambda x: f"{x:.2f}%"), title=f'Contribución (%) - {name}', color_discrete_sequence=COLORS)
                            fig.update_layout(template='plotly_dark', height=380)
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No hay métricas ER destacadas para graficar (filtros aplicados).")
                    except Exception as e:
                        st.error(f"Error gráfico: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: Análisis horizontal
# ----------------------------
elif section == "Análisis horizontal":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Análisis horizontal (Comparación entre períodos)</h3>", unsafe_allow_html=True)
    if len(st.session_state.balances) < 2:
        st.warning("Se necesitan al menos dos BG para análisis horizontal.")
    else:
        names = [n for n, _ in st.session_state.balances]
        act = st.selectbox('Periodo final', names, index=len(names)-1)
        prev = st.selectbox('Periodo inicial', names, index=max(0, len(names)-2))
        idx_act = names.index(act); idx_prev = names.index(prev)
        df_h = horizontal_analysis(st.session_state.balances[idx_prev][1], st.session_state.balances[idx_act][1], prev, act)
        st.markdown('#### Variaciones en Balance General')
        aggrid_dark(df_h.fillna(""), height=380)
        # Export horizontal
        csv_bytes = df_to_csv_bytes(df_h.fillna(""))
        st.download_button(label="Exportar comparación (CSV)", data=csv_bytes, file_name=f"Comparacion_{prev}_vs_{act}.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: Razones financieras
# ----------------------------
elif section == "Razones financieras":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Razones financieras (valores con 2 decimales)</h3>", unsafe_allow_html=True)
    if not st.session_state.balances or not st.session_state.resultados:
        st.warning("Carga al menos un BG y un ER.")
    else:
        all_ratios = []
        n = min(len(st.session_state.balances), len(st.session_state.resultados))
        for i in range(n):
            bal_name, bal_df = st.session_state.balances[i]
            res_name, res_df = st.session_state.resultados[i]
            r = compute_ratios(bal_df, res_df)
            df_r = pd.DataFrame(list(r.items()), columns=["Razón", "Valor"])
            df_r['Periodo'] = bal_name
            all_ratios.append(df_r)
        st.session_state.ratios = all_ratios
        df_ratios_all = pd.concat(all_ratios, ignore_index=True)
        pivoted = df_ratios_all.pivot(index='Razón', columns='Periodo', values='Valor').reset_index().fillna("")
        aggrid_dark(pivoted, height=420)
        # Export ratios
        excel_bytes = df_to_excel_bytes({"Razones": pivoted})
        st.download_button(label="Exportar razones (Excel)", data=excel_bytes, file_name="Razones_FinanSys.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: DuPont
# ----------------------------
elif section == "DuPont":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>DuPont - Desglose del ROE</h3>", unsafe_allow_html=True)
    if not st.session_state.balances or not st.session_state.resultados:
        st.warning("Carga archivos primero.")
    else:
        periods = [n for n, _ in st.session_state.balances]
        sel = st.selectbox('Periodo', periods)
        idx = periods.index(sel)
        df_dup = compute_dupont(st.session_state.balances[idx][1], st.session_state.resultados[idx][1])
        aggrid_dark(df_dup, height=220)
        csvb = df_to_csv_bytes(df_dup)
        st.download_button("Exportar DuPont (CSV)", csvb, file_name=f"DuPont_{sel}.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: Flujo de Efectivo
# ----------------------------
elif section == "Flujo de efectivo":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Flujo de Efectivo - Indirecto y Directo</h3>", unsafe_allow_html=True)
    if len(st.session_state.balances) < 2 or len(st.session_state.resultados) < 1:
        st.warning("Se necesitan al menos dos BG y un ER para calcular el flujo completo (indirecto necesita ER del periodo final).")
    else:
        names = [n for n, _ in st.session_state.balances]
        act = st.selectbox('Periodo final (para ER y BG final)', names, index=len(names)-1, key='efe_act_master')
        prev = st.selectbox('Periodo inicial (BG anterior)', names, index=max(0, len(names)-2), key='efe_prev_master')
        idx_act = names.index(act); idx_prev = names.index(prev)

        # Indirecto
        # pick ER for the final period if exists, otherwise last ER available
        idx_er_for_act = min(idx_act, len(st.session_state.resultados)-1)
        df_efe_ind = compute_cashflow_indirect(st.session_state.balances[idx_act][1], st.session_state.resultados[idx_er_for_act][1], st.session_state.balances[idx_prev][1])
        st.markdown("### Método Indirecto (estimación automática)")
        aggrid_dark(df_efe_ind, height=320)

        # Directo
        df_efe_dir = compute_cashflow_direct(st.session_state.balances[idx_act][1], st.session_state.resultados[idx_er_for_act][1])
        st.markdown("### Método Directo (estimado)")
        aggrid_dark(df_efe_dir, height=220)

        # KPIs
        st.markdown("### KPIs del Flujo")
        A = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Operación (A)","Monto"].sum())
        B = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Inversión (B)","Monto"].sum())
        C = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo de Financiamiento (C)","Monto"].sum())
        N = float(df_efe_ind.loc[df_efe_ind["Concepto"]=="Flujo Neto (A+B+C)","Monto"].sum())
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Flujo Operativo (A)", f"${A:,.2f}")
        k2.metric("Flujo Inversión (B)", f"${B:,.2f}")
        k3.metric("Flujo Financiamiento (C)", f"${C:,.2f}")
        k4.metric("Flujo Neto", f"${N:,.2f}")

        # Gráfico
        chart_df = pd.DataFrame({"Actividad":["Operación","Inversión","Financiamiento"], "Monto":[A,B,C]})
        fig = px.bar(chart_df, x="Actividad", y="Monto", text="Monto", color="Actividad", color_discrete_sequence=COLORS, title=f"Flujos — {prev} → {act}")
        fig.update_layout(template='plotly_dark', height=420)
        fig.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Exports: Excel (multiple sheets), CSV, HTML
        export_name_base = f"Flujo_{prev}_to_{act}"
        excel_bytes = df_to_excel_bytes({"Flujo_Indirecto": df_efe_ind, "Flujo_Directo": df_efe_dir})
        st.download_button("Exportar Flujo (Excel)", data=excel_bytes, file_name=f"{export_name_base}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        csv_ind = df_to_csv_bytes(df_efe_ind); csv_dir = df_to_csv_bytes(df_efe_dir)
        st.download_button("Exportar Indirecto (CSV)", data=csv_ind, file_name=f"{export_name_base}_indirecto.csv", mime="text/csv")
        st.download_button("Exportar Directo (CSV)", data=csv_dir, file_name=f"{export_name_base}_directo.csv", mime="text/csv")

        # HTML report (imprimible a PDF desde navegador)
        html_report = df_to_html_report(f"Estado de Flujo - {prev} → {act}", {"Flujo Indirecto": df_efe_ind, "Flujo Directo": df_efe_dir, "KPIs": pd.DataFrame([{"Concepto":"Flujo Operativo (A)","Monto":A},{"Concepto":"Flujo Inversión (B)","Monto":B},{"Concepto":"Flujo Financiamiento (C)","Monto":C},{"Concepto":"Flujo Neto","Monto":N}])})
        html_bytes = html_report.encode('utf-8')
        st.download_button("Exportar Informe (HTML, imprimir a PDF)", data=html_bytes, file_name=f"{export_name_base}.html", mime="text/html")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: Estado de Origen y Aplicación
# ----------------------------

elif section == "Origen y Aplicación de Fondos":
    import traceback

    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader'>Estado de Origen y Aplicación de Fondos (EOAF)</h3>", unsafe_allow_html=True)

        # Verificar si hay al menos 2 BG
        if len(st.session_state.balances) < 2:
            st.warning("Necesitas al menos dos Balances Generales consecutivos para elaborar el EOAF.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # ---- Selección de períodos ----
        names = [n for n, _ in st.session_state.balances]
        act = st.selectbox("Periodo Actual", names, index=len(names) - 1)
        prev = st.selectbox("Periodo Anterior", names, index=max(0, len(names) - 2))

        idx_act = names.index(act)
        idx_prev = names.index(prev)

        bal_prev_df = st.session_state.balances[idx_prev][1]
        bal_act_df = st.session_state.balances[idx_act][1]

        # ---- ER del periodo actual ----
        if not st.session_state.resultados:
            st.warning("No hay Estados de Resultado cargados — solo se usarán variaciones de BG.")
            er_df = None
        else:
            # Emparejar el ER más cercano al BG actual
            idx_er = min(idx_act, len(st.session_state.resultados) - 1)
            er_df = st.session_state.resultados[idx_er][1]

        # ---- Calcular EOAF ----
        try:
            df_origen, df_aplic, df_resumen = compute_eoaf(bal_prev_df, bal_act_df, er_df)
        except Exception:
            st.error("Error en compute_eoaf(). Traza:")
            st.code(traceback.format_exc())
            st.stop()

        # ---- Mostrar tablas ----
        st.markdown("### Orígenes de Fondos")
        aggrid_dark(df_origen if df_origen is not None and not df_origen.empty 
                    else pd.DataFrame(columns=["Concepto", "Monto"]), height=260)

        st.markdown("### Aplicaciones de Fondos")
        aggrid_dark(df_aplic if df_aplic is not None and not df_aplic.empty 
                    else pd.DataFrame(columns=["Concepto", "Monto"]), height=260)

        st.markdown("### Resumen EOAF")
        aggrid_dark(df_resumen if df_resumen is not None and not df_resumen.empty 
                    else pd.DataFrame(columns=["Tipo", "Monto"]), height=140)

        # ---- Exports ----
        excel_bytes = df_to_excel_bytes({
            "Orígenes": df_origen if df_origen is not None else pd.DataFrame(columns=["Concepto", "Monto"]),
            "Aplicaciones": df_aplic if df_aplic is not None else pd.DataFrame(columns=["Concepto", "Monto"]),
            "Resumen": df_resumen if df_resumen is not None else pd.DataFrame(columns=["Tipo", "Monto"])
        })

        st.download_button(
            "Exportar EOAF (Excel)", 
            data=excel_bytes,
            file_name=f"EOAF_{prev}_to_{act}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            "Exportar Orígenes (CSV)",
            data=df_to_csv_bytes(df_origen if df_origen is not None else pd.DataFrame(columns=["Concepto","Monto"])),
            file_name=f"EOAF_Origenes_{prev}_to_{act}.csv"
        )

        st.download_button(
            "Exportar Aplicaciones (CSV)",
            data=df_to_csv_bytes(df_aplic if df_aplic is not None else pd.DataFrame(columns=["Concepto","Monto"])),
            file_name=f"EOAF_Aplicaciones_{prev}_to_{act}.csv"
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
    if not st.session_state.balances or not st.session_state.resultados:
        st.warning("Carga al menos un BG y un ER.")
    else:
        # Usar primer periodo disponible como base y último como comparación
        n_periods = len(st.session_state.balances)
        idx_latest = n_periods - 1
        idx_first = 0
        bal_latest = st.session_state.balances[idx_latest][1]
        res_latest = st.session_state.resultados[min(idx_latest, len(st.session_state.resultados)-1)][1]
        bal_first = st.session_state.balances[idx_first][1]
        res_first = st.session_state.resultados[min(idx_first, len(st.session_state.resultados)-1)][1]

        ratios_latest = compute_ratios(bal_latest, res_latest)
        ratios_first = compute_ratios(bal_first, res_first)

        # Tarjetas principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Razón Corriente", f"{ratios_latest.get('Razón Circulante','-')}", f"Δ {round(ratios_latest.get('Razón Circulante',0)-ratios_first.get('Razón Circulante',0),2)}")
        c2.metric("Razón Rápida", f"{ratios_latest.get('Razón Rápida','-')}", f"Δ {round(ratios_latest.get('Razón Rápida',0)-ratios_first.get('Razón Rápida',0),2)}")
        c3.metric("Capital Neto de Trabajo", f"{ratios_latest.get('Capital Neto de Trabajo','-')}", f"Δ {round(ratios_latest.get('Capital Neto de Trabajo',0)-ratios_first.get('Capital Neto de Trabajo',0),2)}")
        c4.metric("ROA (%)", f"{ratios_latest.get('ROA (%)','-')}", f"Δ {round(ratios_latest.get('ROA (%)',0)-ratios_first.get('ROA (%)',0),2)}")

        st.markdown("#### Gráfico: Margen y Rotación")
        # Build simple chart data
        chart = pd.DataFrame({
            "Métrica": ["Margen Bruto","Margen Operativo","Margen Neto"],
            "Valor (%)": [ratios_latest.get("Margen Utilidad Bruta (%)",0), ratios_latest.get("Margen Utilidad Operativa (%)",0), ratios_latest.get("Margen Utilidad Neta (%)",0)]
        })
        fig = px.bar(chart, x="Métrica", y="Valor (%)", text="Valor (%)", color="Métrica", color_discrete_sequence=COLORS)
        fig.update_layout(template='plotly_dark', height=380)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Export KPIs
        df_kpis = pd.DataFrame.from_dict(ratios_latest, orient='index', columns=["Valor"]).reset_index().rename(columns={"index":"Indicador"})
        st.download_button("Exportar KPIs (CSV)", data=df_to_csv_bytes(df_kpis), file_name="KPIs.csv", mime="text/csv")
        st.download_button("Exportar KPIs (Excel)", data=df_to_excel_bytes({"KPIs": df_kpis}), file_name="KPIs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sección: Interpretación IA
# ----------------------------
elif section == "Interpretación":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Interpretación Ejecutiva </h3>", unsafe_allow_html=True)
    if not st.session_state.ratios:
        st.warning("Calcula las razones primero en 'Razones financieras'.")
    else:
        # Construir resumen
        summary = "Resumen FinanSys para interpretación:\n\n"
        if st.session_state.balances:
            name, df = st.session_state.balances[0]
            dv = vertical_analysis(name, df, 'BG')
            summary += f"BG {name} - Top cuentas (por %):\n"
            try:
                summary += dv[['Cuenta','Porcentaje']].dropna().head(6).to_string(index=False) + "\n\n"
            except Exception:
                summary += dv.head(6).to_string(index=False) + "\n\n"
        if st.session_state.resultados:
            name, df = st.session_state.resultados[0]
            dv = vertical_analysis(name, df, 'ER')
            summary += f"ER {name} - Top cuentas (por %):\n"
            try:
                summary += dv[['Cuenta','Porcentaje']].dropna().head(6).to_string(index=False) + "\n\n"
            except Exception:
                summary += dv.head(6).to_string(index=False) + "\n\n"
        try:
            df_ratios_all = pd.concat(st.session_state.ratios, ignore_index=True)
            summary += "Razones (muestra):\n" + df_ratios_all.head(12).to_string(index=False)
        except Exception:
            summary += "Razones: (no disponibles)"

        st.code(summary[:3500], language='')

        if st.button('Generar interpretación '):
            with st.spinner('Generando interpretación...'):
                text = generate_interpretation_gemini(summary)
                if text.lower().startswith('error') or 'no se' in text.lower():
                    st.error(text)
                else:
                    st.session_state.ia_interpretation = text
                    st.success("Interpretación generada.")
        if st.session_state.ia_interpretation:
            st.markdown("#### Informe ")
            st.markdown(st.session_state.ia_interpretation)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr class='st-sep'/>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 FinanSys. Todos los derechos reservados.</div>", unsafe_allow_html=True)
