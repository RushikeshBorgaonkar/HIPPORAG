from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re
import json
load_dotenv()

def transform_corpus_to_knowledge_graph(corpus_text):
    """
    Transform the corpus into a knowledge graph by querying an LLM (Groq).
    The LLM will generate triples (head, relation, tail) representing knowledge from the text.
    """
    # Initialize the LLM (Groq)
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    # Prepare the prompt for extracting triples
    prompt = f"""
    You are a knowledge extraction assistant. Given the following corpus of text, extract the relationships between entities and provide them in the form of triples (head, relation, tail).
    
    Corpus:
    {corpus_text}
    
    Please list all the triples in the following format: (head, relation, tail).
    """

    # Generate the response from the LLM
    response = llm.invoke(prompt)

    # Parse the response to extract triples
    triples = parse_triples(response)

    return triples


def parse_triples(response):
    """
    Parse the response from the LLM into a list of triples.
    Handles both JSON and plain-text formats.
    """
    triples = []
    try:
        # Attempt to parse JSON response
        triples = json.loads(response.content)
    except json.JSONDecodeError:
        # If the response is not JSON, fallback to plain text parsing
        lines = response.content.split("\n")
        
        # Regex pattern to capture (head, relation, tail) in plain-text format
        pattern = r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)"
        
        for line in lines:
            line = line.strip()
            match = re.search(pattern, line)
            if match:
                head, relation, tail = match.groups()
                triples.append((head.strip(), relation.strip(), tail.strip()))
            else:
                print(f"Skipping malformed line: {line}")
    except Exception as e:
        print(f"Error parsing triples from LLM response: {e}")

    return triples
