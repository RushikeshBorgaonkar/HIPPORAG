import spacy
nlp = spacy.load("en_core_web_sm")

def query_concepts(query):
    """
    Extract key concepts from the query using spaCy dependency parsing or Groq if needed.
    """
    doc = nlp(query)
    concepts = []

    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "attr", "pobj"):
            concepts.append(normalize_text(token.text))

    return concepts


def normalize_text(text):
    """
    Normalize text by lowercasing and stripping special characters.
    """
    return text.lower().strip()