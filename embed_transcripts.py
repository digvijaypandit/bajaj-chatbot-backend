import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ✅ Clean up old vector DB
persist_dir = "vector_db/transcripts"
if os.path.exists(persist_dir):
    print(f"🧹 Removing old vector DB at: {persist_dir}")
    shutil.rmtree(persist_dir)

# ✅ Transcript file paths
pdf_paths = [
    "data/transcripts/Earnings Call Transcript Q1 - FY25.pdf",
    "data/transcripts/Earnings Call Transcript Q2 - FY25.pdf",
    "data/transcripts/Earnings Call Transcript Q3 - FY25.pdf",
    "data/transcripts/Earnings Call Transcript Q4 - FY25.pdf"
]

# ✅ Load and chunk transcripts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
all_chunks = []

for path in pdf_paths:
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    print(f"📄 Loading: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)
    all_chunks.extend(chunks)

print(f"✅ Loaded and chunked {len(all_chunks)} transcript chunks.")

# 🔍 Preview some chunks to verify content
for i, chunk in enumerate(all_chunks[:3]):
    print(f"\n🔹 Chunk {i+1} Preview:\n{chunk.page_content[:500]}\n---")

# 🧠 Create vector DB from chunks
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=persist_dir
)

print(f"\n✅ Transcripts embedded and stored at: {persist_dir}")
