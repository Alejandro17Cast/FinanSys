import sqlite3

conn = sqlite3.connect("finansys.db")
cursor = conn.cursor()

# Dejar solo Café Diriangén
cursor.execute("""
DELETE FROM empresas
WHERE nombre != 'Café Diriangén'
""")

# También borramos sus estados financieros asociados
cursor.execute("""
DELETE FROM estados_financieros
WHERE empresa_id NOT IN (
    SELECT id FROM empresas WHERE nombre = 'Café Diriangén'
)
""")

conn.commit()
conn.close()

print("✅ Solo quedó Café Diriangén en la base de datos")