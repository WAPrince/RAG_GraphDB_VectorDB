from neo4j import GraphDatabase
import json
from neo4j.graph import Node, Relationship

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

def _convert_record_element(element):
    """Konvertiert Neo4j-spezifische Graph-Objekte in serialisierbare Python-Dicts."""
    if isinstance(element, Node):
        return {
            "identity": element.id,  # Interne Neo4j-ID
            "labels": list(element.labels),
            "properties": dict(element.items())
        }
    elif isinstance(element, Relationship):
        return {
            "identity": element.id,
            "start": element.start_node.id,
            "end": element.end_node.id,
            "type": element.type,
            "properties": dict(element.items())
        }
    elif isinstance(element, dict):
        return {k: _convert_record_element(v) for k, v in element.items()}
    elif isinstance(element, list):
        return [_convert_record_element(v) for v in element]
    else:
        # Standarddatentypen (Strings, Zahlen, Booleans) direkt zurückgeben
        return element

def _format_records_to_json(records, summary=None):
    """Überführt die Records-Liste in die finale JSON-Struktur."""
    formatted_records = []

    for record in records:
        # Ein Record verhält sich wie ein Dict oder eine Sequenz
        record_data = {
            "keys": list(record.keys()),
            "_fields": [_convert_record_element(val) for val in record.values()]
        }
        formatted_records.append(record_data)

    output = {"records": formatted_records}

    # Optional: Summary-Daten anhängen, falls vorhanden
    if summary:
        output["summary"] = {
            "counters": {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created
            },
            "result_available_after": summary.result_available_after,
            "result_consumed_after": summary.result_consumed_after
        }

    # In echten JSON-String umwandeln (formatiert mit Einrückung)
    return json.dumps(output, indent=2, ensure_ascii=False)


def search_documents_in_graphdb(generated_cypher,parameters):

    print("search_documents_in_graphdb START")

    if parameters is None:
        parameters = {}

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, _ = driver.execute_query(
                        query_=generated_cypher,
                        parameters_=parameters,
                        database_="neo4j"
        )
        json_output = _format_records_to_json(records, summary)
        print("search_documents_in_graphdb END")

        return json_output