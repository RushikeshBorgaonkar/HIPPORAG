def retrieve_subgraph(graph, scores, top_k=5):
    """
    Retrieve the top-k relevant subgraph based on PPR scores.
    """
    top_nodes = sorted(scores, key=scores.get, reverse=True)[:top_k]
    subgraph = graph.subgraph(top_nodes)
    return subgraph