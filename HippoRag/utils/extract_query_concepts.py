from langchain_groq import ChatGroq

def query_concepts(query):
    """
    Extract key concepts from the query using Groq LLM.
    """
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    prompt = f"""
    Extract the key concepts from the following query. The concepts should be the primary entities or topics in the query. 
    You should only extract meaningful, relevant concepts and avoid stopwords or unrelated words.

    Query: "{query}"

    Extracted Concepts: 
    """
    
    response = llm.invoke(prompt)
    concepts = response.content.strip().split(",")  
    concepts = [normalize_text(concept.strip()) for concept in concepts]

    return concepts


def normalize_text(text):
    """
    Normalize text by lowercasing and stripping special characters.
    """
    return text.lower().strip()