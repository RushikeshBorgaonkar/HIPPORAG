from langchain_groq import ChatGroq


def generate_augmented_response(query, context_str):
    """
    Generate a response using Groq LLM by providing context and a query.
    """
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    prompt = f"""
    You are an assistant that provides comprehensive answers by analyzing and synthesizing information from the given context. 
    The context provided is the graph data like nodes and edges which are connecting to some information . So you have to generate a repsonse on this given context and dont tell that this node or this is edge on those nodes and edges generate a response Context: {context_str}
    Answer the following question using the provided context: {query}
    """

    response = llm.invoke(prompt)
    
    return response.content

def extract_textual_subgraph_data(subgraph):
    """
    Convert the subgraph into a textual string representing the nodes and edges.
    """
    context_str = ""
    print(f"Subgraph ke nodes : {subgraph.nodes()}")
    for node in subgraph.nodes():
        neighbors = list(subgraph.neighbors(node))
        context_str += f"Node: {node} -> Neighbors: {', '.join(map(str, neighbors))}\n"
        
        for neighbor in neighbors:
            edge_data = subgraph.get_edge_data(node, neighbor)  
            context_str += f"Edge: {node} -> {neighbor} with data: {edge_data}\n"
    
    return context_str

