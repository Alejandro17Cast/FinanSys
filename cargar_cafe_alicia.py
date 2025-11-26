from db import crear_empresa, guardar_estado, obtener_empresas

# Crear empresa
crear_empresa("Café Alicia", "Agroindustria-Café", "2025-02-01")

# Obtener el ID real (por si no es 1)
empresas = obtener_empresas()
for e in empresas:
    if e[1] == "Café Alicia":
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

    # Total Activo = 420,000 ✅

    ("Proveedores", 70000),
    ("Deuda Financiera", 100000),

    # Total Pasivo = 170,000 ✅

    ("Capital Social", 200000),
    ("Utilidades Retenidas", 50000),

    # Total Patrimonio = 250,000 ✅
    # Pasivo + Patrimonio = 420,000 ✅
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

    # Total Activo = 500,000 ✅

    ("Proveedores", 90000),
    ("Deuda Financiera", 110000),

    # Total Pasivo = 200,000 ✅

    ("Capital Social", 200000),
    ("Utilidades Retenidas", 120000),

    # Total Patrimonio = 300,000 ✅
    # Pasivo + Patrimonio = 500,000 ✅
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

print("✅ Datos de Cafetería Alicia cargados correctamente")
print("✅ BG 2023 y 2024 insertados")
print("✅ ER 2024 listo")
print("✅ Todo bien cuadrado para tu EOAF y AOA")