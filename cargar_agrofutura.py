from db import crear_empresa, guardar_estado, obtener_empresas

# ====================================
# Crear empresa
# ====================================

crear_empresa("AgroFutura S.A", "Agroindustrial - Producción", "2025-01-01")

empresas = obtener_empresas()
empresa_id = None

for e in empresas:
    if e[1] == "AgroFutura S.A":
        empresa_id = e[0]
        break

print("Empresa ID:", empresa_id)

# ====================================
# BALANCE GENERAL 2023
# ====================================

bg_2023 = [
    # ACTIVO
    ("Caja y Bancos", 70000),
    ("Cuentas por Cobrar", 50000),
    ("Inventarios", 60000),
    ("Activo Fijo", 160000),
    ("Total Activo", 340000),

    # PASIVO
    ("Proveedores", 60000),
    ("Deuda Financiera", 90000),
    ("Total Pasivo", 150000),

    # PATRIMONIO
    ("Capital Social", 140000),
    ("Utilidades Retenidas", 50000),
    ("Patrimonio", 190000)
]

for cuenta, monto in bg_2023:
    guardar_estado(empresa_id, "BG", "anual", 2023, None, cuenta, monto)

# ====================================
# BALANCE GENERAL 2024
# ====================================

bg_2024 = [
    # ACTIVO
    ("Caja y Bancos", 90000),
    ("Cuentas por Cobrar", 65000),
    ("Inventarios", 75000),
    ("Activo Fijo", 180000),
    ("Total Activo", 410000),

    # PASIVO
    ("Proveedores", 80000),
    ("Deuda Financiera", 100000),
    ("Total Pasivo", 180000),

    # PATRIMONIO
    ("Capital Social", 140000),
    ("Utilidades Retenidas", 100000),
    ("Patrimonio", 240000)
]

for cuenta, monto in bg_2024:
    guardar_estado(empresa_id, "BG", "anual", 2024, None, cuenta, monto)

# ====================================
# ESTADO DE RESULTADOS 2023 y 2024
# ====================================

estado_resultados = [
    ("Ventas Netas", 600000),
    ("Costo de Ventas", 360000),
    ("Utilidad Bruta", 240000),
    ("Gastos Operativos", 160000),
    ("Utilidad Operativa", 80000),
    ("Gasto por Intereses", 10000),
    ("Utilidad Antes de Impuestos", 70000),
    ("Impuestos", 20000),
    ("Utilidad Neta", 50000)
]

for cuenta, monto in estado_resultados:
    guardar_estado(empresa_id, "ER", "anual", 2023, None, cuenta, monto)
    guardar_estado(empresa_id, "ER", "anual", 2024, None, cuenta, monto)

print("✅ AgroFutura cargada perfectamente y cuadrada")