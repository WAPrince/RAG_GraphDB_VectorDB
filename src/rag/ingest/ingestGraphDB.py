from pathlib import Path
import json
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
DATA_FILE = Path("/data/input/neo4j_semiconductor_graph.txt")


class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def _close(self):
        self.driver.close()

    def _import_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        nodes_to_create = []
        rels_to_create = []

        for line in lines:
            line = line.strip()
            if line.startswith("NODE;"):
                parts = line.split(";", 3)
                label = parts[1]
                node_id = parts[2]
                props = json.loads(parts[3])
                nodes_to_create.append((label, node_id, props))
            elif line.startswith("REL;"):
                parts = line.split(";", 3)
                start_id = parts[1]
                rel_type = parts[2]
                end_id = parts[3]
                rels_to_create.append((start_id, rel_type, end_id))

        with self.driver.session() as session:
            # 1. Knoten erstellen
            for label, node_id, props in nodes_to_create:
                query = f"MERGE (n:{label} {{id: $node_id}}) SET n += $props"
                session.run(query, node_id=node_id, props=props)
            print(f"{len(nodes_to_create)} Knoten erfolgreich geladen.")

            # 2. Beziehungen erstellen
            for start_id, rel_type, end_id in rels_to_create:
                # Da die Endknoten (Länder/Techs) evtl. implizit sind, erstellen wir sie generisch falls nicht vorhanden
                query = (
                    f"MERGE (a {{id: $start_id}}) "
                    f"MERGE (b {{id: $end_id}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b)"
                )
                session.run(query, start_id=start_id, end_id=end_id)
            print(f"{len(rels_to_create)} Beziehungen erfolgreich verknüpft.")

if __name__ == "__main__":

    importer = Neo4jImporter("bolt://localhost:7687", "neo4j", "password")
    importer._import_from_file(DATA_FILE)
    importer._close()