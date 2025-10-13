# compare_policies.py
import os
import streamlit as st
from pymongo import MongoClient
from datetime import datetime
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------- Page Config ---------------------------
st.set_page_config(page_title="Manifesto Comparative Analyzer", page_icon="⚖️", layout="wide")

st.title("⚖️ Comparative Policy Analyzer")
st.caption("Compare party manifesto policies side-by-side. Summaries are generated from actual manifesto text only.")

# --------------------------- Config / Secrets ---------------------------
# Make sure GEMINI_API_KEY is configured in Streamlit secrets or environment
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY not found in Streamlit secrets. Summaries will fail without it.")

DATA_DIR = "data"  # folder containing PDFs named like BJP.pdf, INC.pdf, etc.

# --------------------------- UI Controls ---------------------------
PARTIES_AVAILABLE = [f[:-4] for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
if not PARTIES_AVAILABLE:
    st.error(f"No PDFs found in {DATA_DIR}. Put manifesto PDFs (e.g. BJP.pdf) in the data folder.")
    st.stop()

domains = [
    "Women and Child Development",
    "Education",
    "Healthcare",
    "Agriculture",
    "Employment",
    "Economy and Finance",
    "Environment and Energy",
    "Infrastructure",
    "Social Welfare",
    "Defense and Security",
]

emb_model_name="sentence-transformers/all-mpnet-base-v2"
num_chunks_per_party=4


with st.sidebar:
    st.header("⚙ Settings")
    summarize_temperature = st.slider("Summarization temperature", 0.0, 0.8, 0.2, 0.05)
    st.markdown("---")
    st.markdown("Manifestos detected:")
    for p in PARTIES_AVAILABLE:
        st.write(f"- {p}")

# --------------------------- Caches ---------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource(show_spinner=False)
def get_sentence_transformer(model_name: str):
    # Used for similarity computations between domain queries and chunk embeddings
    return SentenceTransformer(model_name)

# --------------------------- PDF -> Text helper ---------------------------
def load_pdf_text(path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = []
        for pidx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            # keep page info for metadata
            text.append((pidx + 1, page_text))
        return text
    except Exception as e:
        st.error(f"Error reading PDF {path}: {e}")
        return []

def create_documents_from_pdf(path: str, party_name: str, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    """Read PDF, split into chunks and return LangChain Document list with metadata."""
    pages = load_pdf_text(path)
    raw_texts = []
    for page_no, txt in pages:
        if txt and txt.strip():
            # attach page marker
            raw_texts.append({"page": page_no, "text": txt.strip(), "source": os.path.basename(path)})

    # Combine page texts into documents accepted by the splitter: we create simple Document objects per page
    docs_for_split = []
    for item in raw_texts:
        docs_for_split.append(Document(page_content=item["text"], metadata={"party": party_name, "source": item["source"], "page": item["page"]}))

    if not docs_for_split:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs_for_split)

    # Ensure each chunk carries party & source metadata
    for c in chunks:
        if "party" not in c.metadata:
            c.metadata["party"] = party_name
        if "source" not in c.metadata:
            c.metadata["source"] = os.path.basename(path)
    return chunks

# --------------------------- Dynamic Vectorstore Creation ---------------------------
@st.cache_resource(show_spinner=True)
def build_vectorstore_for_party(party_name: str, model_name: str) -> FAISS:
    """Create an in-memory FAISS vectorstore for the given party PDF (cached)."""
    pdf_path = os.path.join(DATA_DIR, f"{party_name}.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found for {party_name}: {pdf_path}")

    docs = create_documents_from_pdf(pdf_path, party_name, chunk_size=1000, chunk_overlap=200)
    if not docs:
        raise ValueError(f"No textual content found in {pdf_path}")

    embeddings = get_embeddings(model_name)
    vs = FAISS.from_documents(docs, embeddings)
    return vs

# --------------------------- LLM (Summarization) ---------------------------
@st.cache_resource(show_spinner=False)
def get_chat_model(temperature: float):
    # ChatGoogleGenerativeAI wrapper as used in your app
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        credentials=None,
        convert_system_message_to_human=True
    )

# --------------------------- Summarization Prompts ---------------------------
PARTY_SUMMARY_PROMPT = """
You are an assistant that summarizes policies found in a party manifesto.
Instructions:
- Using ONLY the CONTEXT below (which consists of manifesto excerpts), produce a concise factual summary of the party's POLICIES and PROMISES related to the domain "{domain}".
- Do NOT include criticism, commentary, or external knowledge.
- Focus on concrete schemes, commitments, programs, or explicit policy actions.
- Output should include a short bullet list of the main points and a one-sentence short summary.

CONTEXT:
{context}
"""

COMPARISON_PROMPT = """
You are an assistant that COMPARES policy summaries from multiple political parties.
Instructions:
- Using ONLY the party summaries provided, produce:
  1) A short comparison paragraph (2-4 sentences) highlighting main differences and similarities.
  2) A concise bullet-list that highlights 2 unique commitments per party (if available).
- Use neutral academic tone.

PARTY SUMMARIES:
{party_summaries}
"""

# --------------------------- UI Elements ---------------------------
st.markdown("## Comparative Analysis Mode")
st.markdown("Select parties and a policy domain; the app will build per-party vectorstores on demand, retrieve domain-specific chunks, summarize each party's policies, and create a short comparison.")

selected_parties = st.multiselect("Select Parties to Compare", PARTIES_AVAILABLE, default=PARTIES_AVAILABLE[:2])
selected_domain = st.selectbox("Select Policy Domain", domains)
run = st.button("🔍 Generate Comparison")


# --------------------------- Run comparison ---------------------------
def retrieve_domain_chunks_for_party(party: str, domain: str, k: int, model_name: str):
    """Build vectorstore for the party (cached) and retrieve top-k chunks semantically related to the domain query."""
    vs = build_vectorstore_for_party(party, model_name)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    query = f"Policies related to {domain}"
    docs = retriever.get_relevant_documents(query)
    return docs, vs

def summarize_party(party: str, docs: List[Document], domain: str, chat_llm: ChatGoogleGenerativeAI):
    """Create a summary for a party using the chat model given a list of documents."""
    if not docs:
        return "No relevant policy content found in this manifesto for the selected domain."

    context = "\n\n".join([f"- {d.page_content.strip()[:1200]}" for d in docs])  # keep excerpts
    prompt = PARTY_SUMMARY_PROMPT.format(domain=domain, context=context)

    try:
        # Using chat_model.invoke to be consistent with your integration
        resp = chat_llm.invoke(prompt)
        # handle different response shapes
        summary = getattr(resp, "content", None) or (resp[0].content if isinstance(resp, (list, tuple)) and len(resp) else str(resp))
        return summary.strip()
    except Exception as e:
        st.error(f"Error calling LLM for {party}: {e}")
        return "LLM error - could not summarize."


# --------------------------- Execution ---------------------------
if run:
    if not selected_parties:
        st.warning("Select at least one party to compare.")
    else:
        chat_llm = get_chat_model(summarize_temperature)
        sentence_model = get_sentence_transformer(emb_model_name)

        st.markdown(f"### 📊 Comparing: {', '.join(selected_parties)}  —  Domain: **{selected_domain}**")
        cols = st.columns(len(selected_parties))

        party_summaries = {}
        similarity_scores = {}

        for idx, party in enumerate(selected_parties):
            with cols[idx]:
                st.subheader(party)
                try:
                    docs, vs = retrieve_domain_chunks_for_party(party, selected_domain, num_chunks_per_party, emb_model_name)
    

                    # Summarize
                    summary = summarize_party(party, docs, selected_domain, chat_llm)
                    st.markdown("**Summary (auto-generated):**")
                    st.write(summary)

                    party_summaries[party] = summary


                except Exception as e:
                    st.error(f"Error processing {party}: {e}")

        # Comparison synthesis across parties
        if party_summaries:
            st.markdown("## 🔎 Cross-Party Comparison")
            party_summaries_text = "\n\n".join([f"{p}:\n{party_summaries[p]}" for p in party_summaries])
            comp_prompt = COMPARISON_PROMPT.format(party_summaries=party_summaries_text)
            try:
                comp_resp = chat_llm.invoke(comp_prompt)
                comp_text = getattr(comp_resp, "content", None) or (comp_resp[0].content if isinstance(comp_resp, (list, tuple)) and len(comp_resp) else str(comp_resp))
                st.markdown("**Comparison Summary:**")
                st.write(comp_text)
            except Exception as e:
                st.error(f"Error generating comparison summary: {e}")

# --------------------------- End of file ---------------------------
