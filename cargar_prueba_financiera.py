from db import crear_empresa, obtener_empresas, guardar_estado

# ============================================
# 1. Crear Empresa: PRUEBA DE FINANZAS
# ============================================
crear_empresa("Prueba de Finanzas", "Pruebas del Sistema", "2025-01-01")

# Obtener ID real
empresas = obtener_empresas()
for e in empresas:
    if e[1] == "Prueba de Finanzas":
        empresa_id = e[0]
        break

print("Empresa ID:", empresa_id)


# ============================================
# 2. BALANCE GENERAL 2019 (CUADRADO)
# ============================================
bg_2019 = [
    ("Efectivo", 5000),
    ("Cuentas por Cobrar", 12000),
    ("Inventarios", 25000),
    ("Activo Fijo", 150000),
    ("Total Activo", 192000),

    ("Proveedores", 30000),
    ("Deudas", 40000),
    ("Total Pasivo", 70000),

    ("Capital Social", 100000),
    ("Utilidades Retenidas", 22000),
    ("Patrimonio", 122000),
]

for cuenta, monto in bg_2019:
    guardar_estado(empresa_id, "BG", "anual", 2019, None, cuenta, monto)


# ============================================
# 3. BALANCE GENERAL 2020 (CUADRADO)
# ============================================
# Nota: Ajustamos Utilidades Retenidas 2020 a 0 y agregamos Dividendos Pagados = 46,000
# Para cuadrar el EOAF perfectamente.

bg_2020 = [
    ("Efectivo", 8000),
    ("Cuentas por Cobrar", 15000),
    ("Inventarios", 25000),        # BAJÓ 5,000
    ("Activo Fijo", 170000),       # SUBIÓ 10,000 extra
    ("Total Activo", 218000),      # Ajustado

    # PASIVOS (Proveedores y Deudas bajan 8,000)
    ("Proveedores", 33000),
    ("Deudas", 42000),
    ("Total Pasivo", 75000),

    # PATRIMONIO
    ("Capital Social", 100000),
    ("Utilidades Retenidas", 33000 - 20000),  # menos los dividendos
    ("Dividendos Pagados", 20000),            # para EOAF
    ("Patrimonio", 143000),                   # Ajustado
]


for cuenta, monto in bg_2020:
    guardar_estado(empresa_id, "BG", "anual", 2020, None, cuenta, monto)


# ============================================
# 4. ESTADO DE RESULTADOS 2019
# ============================================
er_2019 = [
    ("Ventas Netas", 300000),
    ("Costo de Ventas", 180000),
    ("Utilidad Bruta", 120000),

    ("Gastos Operativos", 70000),
    ("Utilidad Operativa", 50000),

    ("Gasto por Intereses", 10000),
    ("Utilidad Antes de Impuestos", 40000),
    ("Impuestos", 12000),
    ("Utilidad Neta", 28000),
]

for cuenta, monto in er_2019:
    guardar_estado(empresa_id, "ER", "anual", 2019, None, cuenta, monto)


# ============================================
# 5. ESTADO DE RESULTADOS 2020
# ============================================
er_2020 = [
    ("Ventas Netas", 350000),
    ("Costo de Ventas", 200000),
    ("Utilidad Bruta", 150000),

    ("Gastos Operativos", 80000),
    ("Utilidad Operativa", 70000),

    ("Gasto por Intereses", 12000),
    ("Utilidad Antes de Impuestos", 58000),
    ("Impuestos", 17000),
    ("Utilidad Neta", 41000),
]

for cuenta, monto in er_2020:
    guardar_estado(empresa_id, "ER", "anual", 2020, None, cuenta, monto)


print("✅ Datos de PRUEBA DE FINANZAS cuadrado perfectamente cargados.")