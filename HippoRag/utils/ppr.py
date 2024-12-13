import networkx as nx
from difflib import get_close_matches

def match_query_concepts_to_graph(query_concepts, graph_nodes, fuzzy_threshold=0.6):
    matched_concepts = []
    
    for concept in query_concepts:
        if concept in graph_nodes:
            matched_concepts.append(concept)
        else:
            # Fuzzy matching with a threshold
            closest_match = get_close_matches(concept, graph_nodes, n=1, cutoff=fuzzy_threshold)
            if closest_match:
                print(f"Fuzzy match found for '{concept}': '{closest_match[0]}'")
                matched_concepts.append(closest_match[0])
            else:
                print(f"No match found for: {concept}")
    return matched_concepts

def apply_ppr(graph, query_concepts, alpha=0.85):
    """
    Apply Personalized PageRank (PPR) to the graph using query concepts as seeds.
    """
    graph_nodes = [normalize_text(node) for node in graph.nodes()]
    print(f"Graph nodes: {graph_nodes}")
    
    matched_concepts = match_query_concepts_to_graph(query_concepts, graph_nodes)
    print("MATCHED CONCEPTS :", matched_concepts)
    
    if not matched_concepts:
        return {"message": f"No query concepts {query_concepts} could be matched to the graph nodes."}
    
    personalization = {node: 1 if node in matched_concepts else 0 for node in graph.nodes()}
    print("Personalization:", personalization)

    try:
        scores = nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except ZeroDivisionError:
        return {"message": "Graph is disconnected, and PageRank cannot be applied!"}

    print("Scores:", scores)
    return scores



def normalize_text(text):
    """
    Normalize text by lowercasing and stripping special characters.
    """
    return text.lower().strip()