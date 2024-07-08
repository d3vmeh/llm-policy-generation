from neo4j import GraphDatabase

class GraphUploader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upload_csv(self, nodes_csv_path, relationships_csv_path):
        with self.driver.session() as session:
            # Upload nodes from CSV
            session.write_transaction(self._upload_nodes, nodes_csv_path)
            # Upload relationships from CSV
            session.write_transaction(self._upload_relationships, relationships_csv_path)

    @staticmethod
    def _upload_nodes(tx, csv_path):
        query = """
        LOAD CSV WITH HEADERS FROM $csv_path AS row
        CREATE (:Node {id: row.id, name: row.name})
        """
        tx.run(query, csv_path=csv_path)

    @staticmethod
    def _upload_relationships(tx, csv_path):
        query = """
        LOAD CSV WITH HEADERS FROM $csv_path AS row
        MATCH (start:Node {id: row.startId}), (end:Node {id: row.endId})
        CREATE (start)-[:RELATES_TO]->(end)
        """
        tx.run(query, csv_path=csv_path)
        
class GraphExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def export_to_csv(self, nodes_file, relationships_file):
        with self.driver.session() as session:
            # Export nodes
            session.write_transaction(self._export_nodes, nodes_file)
            # Export relationships
            session.write_transaction(self._export_relationships, relationships_file)

    @staticmethod
    def _export_nodes(tx, file_path):
        query = """
        CALL apoc.export.csv.query(
            "MATCH (n) RETURN n",
            $file_path,
            {}
        )
        """
        tx.run(query, file_path=file_path)

    @staticmethod
    def _export_relationships(tx, file_path):
        query = """
        CALL apoc.export.csv.query(
            "MATCH ()-[r]->() RETURN r",
            $file_path,
            {}
        )
        """
        tx.run(query, file_path=file_path)

class GraphImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_from_csv(self, nodes_file, relationships_file):
        with self.driver.session() as session:
            # Import nodes
            session.write_transaction(self._import_nodes, nodes_file)
            # Import relationships
            session.write_transaction(self._import_relationships, relationships_file)

    @staticmethod
    def _import_nodes(tx, file_path):
        query = """
        LOAD CSV WITH HEADERS FROM $file_path AS row
        CREATE (n:Node {props})
        SET n += row
        """
        tx.run(query, file_path="file:///" + file_path)

    @staticmethod
    def _import_relationships(tx, file_path):
        query = """
        LOAD CSV WITH HEADERS FROM $file_path AS row
        MATCH (start:Node {id: row.startId}), (end:Node {id: row.endId})
        CREATE (start)-[r:RELATES {type: row.type}]->(end)
        """
        tx.run(query, file_path="file:///" + file_path)

