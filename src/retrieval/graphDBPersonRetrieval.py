from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")


def find_companies_by_person(tx, person_name):
    # Cypher-Query: Findet die Person, die Beziehungen und die Ziel-Firmen
    query = """
    MATCH (p:Person {name: $name})-[r:WORKS_AS]->(c:Company)
    RETURN c.name AS company, r.role AS role, c.industry AS industry
    """
    result = tx.run(query, name=person_name)
    return [record for record in result]

def main():
    search_name = "Elon Musk"
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            # Lese-Transaktion ausführen
            records = session.execute_read(find_companies_by_person, search_name)

            if not records:
                print(f"Keine Einträge für '{search_name}' gefunden.")
                return

            print(f"Ergebnisse für '{search_name}':")
            print("-" * 40)
            for record in records:
                print(f"• Firma: {record['company']} | Rolle: {record['role']} | Branche: {record['industry']}")


if __name__ == "__main__":
    main()
