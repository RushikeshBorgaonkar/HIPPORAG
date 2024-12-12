from langgraph.graph import StateGraph, START, END

from networkx import DiGraph  
from flask import Flask, request, render_template
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from pydantic import BaseModel,ConfigDict
from utils import (
    data_loading,
    built_knowledge_graph,
    extract_query_concepts,
    extracting_triplets,
    generate_response_LLM,
    ppr,
    subgraph_retrieval,
    visualize,
    workflow_graph   
)

app = Flask(__name__)

class KnowledgeGraphState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 
    query: str ="tell me about django?"
    top_k: int =10
    text_chunks: list = []  
    all_triples: list = []
    knowledge_graph: DiGraph = DiGraph() 
    query_concepts: list = []
    ppr_scores: dict = {}
    subgraph: DiGraph = DiGraph() 
    context_data: str = ""
    response: str = ""

def update_state(state: KnowledgeGraphState, new_state) :
    return state.model_copy(update=new_state)  


def load_data(state: KnowledgeGraphState) :
    pdf_path ="C:/Users/Coditas-Admin/Desktop/POC HIPPO RAG/Hippo/Basic KG/Python data.pdf"   
    print(f"Loading data from {pdf_path}...")
    
    try:
        # Load the text chunks using your method
        text_chunks = data_loading.load_pdf_with_langchain(pdf_path)
        
        if isinstance(text_chunks, list):
            return update_state(state, {"text_chunks": text_chunks})
        else:
            print(f"Error: Expected a list of text chunks, but got {type(text_chunks)}.")
            return state
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return state
    
def extract_triplets(state: KnowledgeGraphState) :
    text_chunks = state.text_chunks

    all_triples = []
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)}...")
        try:
            triples = extracting_triplets.transform_corpus_to_knowledge_graph(chunk)
            all_triples.extend(triples)
            print("Triples: ",triples)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")

    print("Triplets : ",all_triples)
    return update_state(state, {"all_triples": all_triples})

def build_knowledge_graph(state: KnowledgeGraphState) :
    all_triples = state.all_triples
    knowledge_graph = built_knowledge_graph.build_knowledge_graph_from_llm(all_triples)
    print("Knowledge Graph:", knowledge_graph)
    visualize.visualize_graph(knowledge_graph, title="Whole Knowledge Graph")
    return update_state(state, {"knowledge_graph": knowledge_graph})

def extract_query_concepts_node(state: KnowledgeGraphState) :
    query = state.query
    query_concepts = extract_query_concepts.query_concepts(query)
    print(f"Query concepts: {query_concepts}")
    return update_state(state, {"query_concepts": query_concepts})

def personalized_pagerank(state: KnowledgeGraphState) :
    knowledge_graph = state.knowledge_graph
    query_concepts = state.query_concepts
    ppr_scores = ppr.apply_ppr(knowledge_graph, query_concepts)
    return update_state(state, {"ppr_scores": ppr_scores})

def retrieve_relevant_subgraph(state: KnowledgeGraphState):
    knowledge_graph = state.knowledge_graph
    ppr_scores = state.ppr_scores
    top_k = state.top_k
    subgraph = subgraph_retrieval.retrieve_subgraph(knowledge_graph, ppr_scores, top_k=top_k)
    visualize.visualize_graph(subgraph, title="Relevant Subgraph")
    return update_state(state, {"subgraph": subgraph})

def generate_response(state: KnowledgeGraphState):
    subgraph = state.subgraph
    query = state.query
    context_data = generate_response_LLM.extract_textual_subgraph_data(subgraph)
    print("Subgraph Data:", context_data)
    response = generate_response_LLM.generate_augmented_response(query, context_data)
    print("Response:", response)
    return update_state(state, {"response": response})


# Initialize StateGraph
workflow = StateGraph(KnowledgeGraphState)

# Add nodes to the workflow with updated names
workflow.add_node("load_data", load_data)
workflow.add_node("extract_triplets", extract_triplets)
workflow.add_node("build_knowledge_graph", build_knowledge_graph)
workflow.add_node("extract_query_concepts", extract_query_concepts_node)
workflow.add_node("personalized_pagerank", personalized_pagerank)
workflow.add_node("retrieve_subgraph", retrieve_relevant_subgraph)
workflow.add_node("generate_response", generate_response)

# Define workflow connections
workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "extract_triplets")
workflow.add_edge("extract_triplets", "build_knowledge_graph")
workflow.add_edge("build_knowledge_graph", "extract_query_concepts")
workflow.add_edge("extract_query_concepts", "personalized_pagerank")
workflow.add_edge("personalized_pagerank", "retrieve_subgraph")
workflow.add_edge("retrieve_subgraph", "generate_response")
workflow.add_edge("generate_response", END)

   
# Flask routes
@app.route("/", methods=["GET", "POST"])
def chat():
    response = None
    query = ""
    graph_path = None

    if request.method == "POST":
        query = request.form["query"]
        state = KnowledgeGraphState(query=query)  # Initialize with the user's query
        compiled_workflow = workflow.compile()
        result = compiled_workflow.invoke(state)
        response = result["response"] # Extract the response
        # Generate workflow graph
        work_flow_graph = workflow_graph.create_workflow_graph(workflow)
        workflow_graph.visualize_workflow(work_flow_graph)
        

    return render_template("chat.html", query=query, response=response)

if __name__ == "__main__":
    app.run(debug=True)