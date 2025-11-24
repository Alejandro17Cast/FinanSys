from db import crear_empresa, guardar_estado, obtener_empresas

# Crear empresa
crear_empresa("Café Diriangén", "Agroindustrial - Café", "2025-01-01")

# Obtener el ID real (por si no es 1)
empresas = obtener_empresas()
for e in empresas:
    if e[1] == "Café Diriangén":
        empresa_id = e[0]
        break

print("Empresa ID:", empresa_id)

# ======================
# BALANCE GENERAL 2023
# ======================
bg_2023 = [
    ("Caja y Bancos", 100000),
    ("Cuentas por Cobrar", 60000),
    ("Inventarios", 80000),
    ("Activo Fijo", 180000),
    ("Total Activo", 420000),

    ("Proveedores", 70000),
    ("Deuda Financiera", 100000),
    ("Total Pasivo", 170000),

    ("Capital Social", 200000),
    ("Utilidades Retenidas", 50000),
    ("Patrimonio", 250000),
]

for cuenta, monto in bg_2023:
    guardar_estado(empresa_id, "BG", "anual", 2023, None, cuenta, monto)

# ======================
# BALANCE GENERAL 2024
# ======================
bg_2024 = [
    ("Caja y Bancos", 120000),
    ("Cuentas por Cobrar", 80000),
    ("Inventarios", 100000),
    ("Activo Fijo", 200000),
    ("Total Activo", 500000),

    ("Proveedores", 90000),
    ("Deuda Financiera", 110000),
    ("Total Pasivo", 200000),

    ("Capital Social", 200000),
    ("Utilidades Retenidas", 100000),
    ("Patrimonio", 300000),
]

for cuenta, monto in bg_2024:
    guardar_estado(empresa_id, "BG", "anual", 2024, None, cuenta, monto)

# ======================
# ESTADO DE RESULTADOS
# ======================
estado_resultados = [
    ("Ventas Netas", 900000),
    ("Costo de Ventas", 540000),
    ("Utilidad Bruta", 360000),
    ("Gastos Operativos", 230000),
    ("Utilidad Operativa", 130000),
    ("Gasto por Intereses", 30000),
    ("Utilidad Antes de Impuestos", 100000),
    ("Impuestos", 30000),
    ("Utilidad Neta", 70000),
]

for cuenta, monto in estado_resultados:
    guardar_estado(empresa_id, "ER", "anual", 2023, None, cuenta, monto)
    guardar_estado(empresa_id, "ER", "anual", 2024, None, cuenta, monto)

print("✅ Datos cargados exitosamente.")
