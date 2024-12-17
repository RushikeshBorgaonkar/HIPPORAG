load_dotenv()  # Load environment variables from .env file
from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
# Retrieve Neo4j connection details from .env file
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="rUSHI_572"

# Initialize Neo4j Driver
# neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)
print(graph)

def save_graph_to_neo4j(graph: DiGraph, neo4j_graph: Neo4jGraph):
   
    try:
        # Iterate through the nodes
        for node, node_props in graph.nodes(data=True):
            neo4j_graph.query(
                """
                MERGE (n:Node {id: $id})
                SET n += $props
                """,
                parameters={"id": node, "props": node_props}
            )
        
        # Iterate through the edges
        for source, target, edge_props in graph.edges(data=True):
            neo4j_graph.query(
                """
                MATCH (a:Node {id: $source_id})
                MATCH (b:Node {id: $target_id})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r += $props
                """,
                parameters={"source_id": source, "target_id": target, "props": edge_props}
            )
        
        print("Graph successfully saved to Neo4j!")
    
    except Exception as e:
        print(f"Error saving graph to Neo4j: {e}")


save_graph_to_neo4j(knowledge_graph, graph)