import sqlite3

conn = sqlite3.connect("finansys.db")
cursor = conn.cursor()

# Dejar solo Café Diriangén y AgroFutura S.A
cursor.execute("""
DELETE FROM empresas
WHERE nombre NOT IN ('Café Diriangén', 'AgroFutura S.A')
""")

# Borrar estados financieros de las demás empresas
cursor.execute("""
DELETE FROM estados_financieros
WHERE empresa_id NOT IN (
    SELECT id FROM empresas 
    WHERE nombre IN ('Café Diriangén', 'AgroFutura S.A')
)
""")

conn.commit()
conn.close()

print("✅ Solo quedaron Café Diriangén y AgroFutura S.A en la base de datos")
