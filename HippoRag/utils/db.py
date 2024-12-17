from neo4j import GraphDatabase

URI = "neo4j://6c9a589a.databases.neo4j.io/"
AUTH = ("neo4j", "v4wCSEYLavHxIKMMgnWhSAVI61tTqOcRZTXoUSxxNLc")

driver = GraphDatabase.driver(URI, auth=AUTH, encrypted=True, trust="TRUST_ALL_CERTIFICATES")

with driver.session() as session:
    result = session.run("RETURN 'Connection successful!' AS message")
    print(result.single()["message"])


def save_graph_to_neo4j(knowledge_graph):
    print("Checking if the graph is already stored...")

    with driver.session() as session:
        # Check if any nodes or relationships already exist
        result = session.run("MATCH (n) RETURN COUNT(n) AS node_count")
        node_count = result.single()["node_count"]

        # If nodes exist, skip saving the graph
        if node_count > 0:
            print("Graph already exists in Neo4j. Skipping graph storage.")
            return

        print("Graph not found. Saving the graph to Neo4j...")

        # Save nodes and relationships
        for u, v, data in knowledge_graph.edges(data=True):
            query = """
            MERGE (a:Entity {name: $node1})
            MERGE (b:Entity {name: $node2})
            MERGE (a)-[r:RELATIONSHIP {type: $relationship}]->(b)
            """
            session.run(query, node1=u, node2=v, relationship=data.get("type", "RELATED"))

        print("Graph successfully saved to Neo4j!")
