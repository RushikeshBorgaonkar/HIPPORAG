from langchain_groq import ChatGroq


def generate_augmented_response(query, context_str):
   
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    prompt = f"""
    You are an assistant that provides comprehensive answers by analyzing and synthesizing information from the given context. 
    The context is derived from the relationships and relevant information in the knowledge graph. The graph data includes various connected concepts, but you should not directly mention the nodes, edges, or their relationships in your answer. Only use the high-level information that the context provides.

    Context: {context_str}

    Answer the following question based on the provided context: {query}
    """

    response = llm.invoke(prompt)
    
    return response.content

def extract_textual_subgraph_data(subgraph):
    context_str = ""
    
    for node in subgraph.nodes():
        context_str += f"{node}\n"  
    
    for node in subgraph.nodes():
        neighbors = list(subgraph.neighbors(node))
        for neighbor in neighbors:
            edge_data = subgraph.get_edge_data(node, neighbor)
            context_str += f"{node} -> {neighbor}: {edge_data['label']}\n"
    
    return context_str

