from langchain_groq import ChatGroq


def generate_augmented_response(query, context_str):
   
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    prompt = f"""
    You are an advanced assistant capable of synthesizing and analyzing information from a structured dataset.
    The context provided is extracted from a knowledge graph but has been converted into meaningful textual information.
    Focus only on the insights, facts, and relationships derived from the graph, and exclude references to structural details like "nodes" or "edges."
    
    Here is the context: 
    {context_str}
    
    Based on this context, answer the following query comprehensively and concisely:
    {query}
    """

    response = llm.invoke(prompt)
    
    return response.content

def extract_textual_subgraph_data(subgraph):
   
    context_str = ""
    print(f"Subgraph ke nodes : {subgraph.nodes()}")
    for node in subgraph.nodes():
        neighbors = list(subgraph.neighbors(node))
        context_str += f"Node: {node} -> Neighbors: {', '.join(map(str, neighbors))}\n"
        
        for neighbor in neighbors:
            edge_data = subgraph.get_edge_data(node, neighbor)  
            context_str += f"Edge: {node} -> {neighbor} with data: {edge_data}\n"
    
    return context_str

