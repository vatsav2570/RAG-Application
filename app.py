import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -------- Environment Variables --------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not (PINECONE_API_KEY and PINECONE_ENV and INDEX_NAME and GEMINI_API_KEY):
    st.error("Please set all environment variables: PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, GEMINI_API_KEY")
    st.stop()

# -------- Imports --------
from pinecone import Pinecone
from google import genai
from PyPDF2 import PdfReader

# -------- Initialize Clients --------
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(INDEX_NAME)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# -------- PDF Text Extraction --------
def extract_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size=800, overlap=100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# -------- Embeddings --------
def extract_vector(emb_obj):
    """Safely get list of floats from embed object."""
    if hasattr(emb_obj, "values"):
        vec = emb_obj.values
    elif hasattr(emb_obj, "embedding"):
        vec = emb_obj.embedding
    else:
        vec = emb_obj
    return [float(x) for x in vec]

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    vectors = []
    for item in response.embeddings:
        raw_vector = extract_vector(item)
        vectors.append(raw_vector[:1024])  # match your index dim
    return vectors

# -------- Streamlit UI --------
st.title("📚 RAG App (Streamlit + Pinecone + Gemini)")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    raw_text = extract_pdf_text(uploaded_pdf)
    if not raw_text.strip():
        st.error("No text found in the PDF.")
    else:
        st.write(f"Extracted {len(raw_text)} characters of text.")
        chunks = chunk_text(raw_text)

        with st.spinner("Embedding & upserting into Pinecone..."):
            embeddings = embed_texts(chunks)
            to_upsert = [
                (str(i), vector, {"text": chunk})
                for i, (vector, chunk) in enumerate(zip(embeddings, chunks))
            ]
            index.upsert(vectors=to_upsert)

        st.success("Indexed PDF into Pinecone!")

query = st.text_input("Ask a question about the document:")

if query:
    with st.spinner("Retrieving answer..."):
        q_vec = embed_texts([query])[0]

        results = index.query(
            vector=q_vec,
            top_k=5,
            include_metadata=True
        )

        retrieved_texts = [m.metadata.get("text", "") for m in results.matches]
        context = "\n\n".join(retrieved_texts)

        prompt = (
            f"Answer the question based on the context below:\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        )

        # **Correct generation call — no invalid args**
        gen_resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = gen_resp.text  # .text gives generated text

    st.markdown("### 🧠 Answer")
    st.write(answer)
