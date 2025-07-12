from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from gemini_client import ask_gemini

# Load Vector DB (created by embed_transcripts.py)
persist_dir = "vector_db/transcripts"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

# ✅ Use MMR for better and faster results (optional)
def get_relevant_chunks(query: str, k: int = 8) -> list[Document]:
    return vectordb.max_marginal_relevance_search(query, k=k)

def build_prompt(query: str, chunks: list[Document]) -> str:
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    return f"""
You are a financial analyst assistant. Your job is to answer the user's question based **only** on the verified transcript data provided in the context below.

Instructions:
- If the answer is found in the context, give a short, clear, and fact-based answer.
- If the answer is not directly present but can be inferred from the context, provide a short summary. If no information is found at all, say: "Information not available in the transcripts."
- If the user's question is too vague or ambiguous, politely ask for clarification instead of guessing.

Context:
{context}

User Question:
{query}

Answer:""".strip()


# ✅ Final RAG function
def answer_question_with_rag(query: str) -> str:
    chunks = get_relevant_chunks(query)
    prompt = build_prompt(query, chunks)
    return ask_gemini(prompt)
