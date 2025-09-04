import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from gtts import gTTS
from io import BytesIO
from PyPDF2 import PdfReader
import re
from datetime import datetime
import asyncio
import edge_tts
import tempfile
from pymongo import MongoClient
from langchain.schema import Document

# --------------------------- Page Config ---------------------------
st.set_page_config(page_title="Manifesto Analyzer", page_icon="📜")

# --------------------------- Custom Styling ---------------------------
st.markdown("""
    <style>
    body { font-family: 'Segoe UI', sans-serif; }
    .css-18e3th9 { background: linear-gradient(to right, #1f1c2c, #928dab); color: white; }
    .css-1d391kg { background: #121212 !important; color: #fefefe; }
    .css-1v3fvcr { color: #ff79c6 !important; font-weight: bold; }
    .block-container { padding: 2rem 1rem; }
    .st-bx { border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    .stTextInput, .stSelectbox, .stSlider, .stCheckbox { font-weight: bold; }
    .stChatMessage { background-color: #1e1e1e; border-left: 4px solid #50fa7b; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage p { color: #f1f1f1; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #ffb86c; }
    .stButton>button { background-color: #bd93f9; color: white; border-radius: 10px; padding: 0.5rem 1rem; border: none; }
    </style>
""", unsafe_allow_html=True)

# --------------------------- MongoDB ---------------------------
MONGO_URI = st.secrets["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client["manifesto_ai"]
logs_collection = db["chat_logs"]

# --------------------------- API Key ---------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --------------------------- Load Vector Store ---------------------------
@st.cache_resource(show_spinner=True)
def load_vectorstore_from_disk(folder_path):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
        credentials=None
    )
    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

all_manifestos_store = load_vectorstore_from_disk("vectorstores/acts")

# --------------------------- PDF Helper ---------------------------
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def create_manifesto_vector(pdf_filename):
    pdf_path = os.path.join("data", pdf_filename)
    manifesto_text = extract_text_from_pdf(pdf_path)
    manifesto_doc = Document(page_content=manifesto_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    return FAISS.from_documents([manifesto_doc], embeddings)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.title("📜 Manifesto Analyzer")
    st.markdown("### ⚡ What can I do?")
    st.markdown("- 🔍 Search across manifestos")
    st.markdown("- 💬 Ask questions about policies")
    st.markdown("- 🧠 Get contextual insights")

    st.markdown("---")

    manifestos = [f for f in os.listdir("data") if f.lower().endswith(".pdf")]
    selected_manifesto = st.selectbox("📚 Choose a Manifesto", ["All Manifestos"] + manifestos)

    if "messages" in st.session_state:
        if 'last_selected_manifesto' not in st.session_state:
            st.session_state.last_selected_manifesto = selected_manifesto
        if st.session_state.last_selected_manifesto != selected_manifesto:
            st.session_state.messages = []
            st.session_state.last_selected_manifesto = selected_manifesto

    st.markdown("---")
    st.subheader("🛠️ Settings")
    temperature = st.slider("🔥 Response Creativity", 0.0, 1.0, 0.7, 0.1)

    st.markdown("---")
    st.info("Crafted with ❤️ by Dhyan Shah")
    st.caption("Manifesto Analyzer v1.0 | Democracy Meets AI ✨")

# --------------------------- Main Header ---------------------------
st.title("📜 Manifesto Analyzer – Your Political AI Assistant")

# Retriever logic
if selected_manifesto == "All Manifestos":
    st.caption("📘 Chatting using all manifestos")
    retriever = all_manifestos_store.as_retriever(search_kwargs={"k": 3})
else:
    st.caption(f"🎓 Chatting about: `{selected_manifesto}`")
    retriever = create_manifesto_vector(selected_manifesto).as_retriever(search_kwargs={"k": 3})

# --------------------------- Memory + LLM ---------------------------
memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=temperature,
    credentials=None,
    convert_system_message_to_human=True
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    verbose=True
)

def save_chat_log(question, answer, source_docs, manifesto_name):
    logs_collection.insert_one({
        "timestamp": datetime.utcnow(),
        "user_question": question,
        "assistant_response": answer,
        "used_manifesto": manifesto_name if manifesto_name != "All Manifestos" else None,
        "source_documents": [doc.metadata.get("source", "unknown") for doc in source_docs] if source_docs else [],
        "chat_id": st.session_state.get("chat_id", "default_session")
    })

# --------------------------- Init Messages ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🎉 Welcome to Manifesto Analyzer. Ask about any manifesto or choose one from the sidebar!"}
    ]

# --------------------------- Chat Loop ---------------------------
prompt = st.chat_input("💬 Ask something about policies or promises...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing manifestos..."):
            try:
                question = st.session_state.messages[-1]["content"]
                response = retrieval_chain({
                    "question": question,
                    "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages if msg["role"] != "assistant"]
                })

                answer = response['answer']
                st.write(answer)

                async def synthesize_and_play(text):
                    voice = "en-US-AriaNeural"  # You can change voices
                    mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    tts = edge_tts.Communicate(text, voice)
                    await tts.save(mp3_path)
                    with open(mp3_path, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")

                clean_answer = re.sub(r"[*_`#>\-]+", "", answer)
                asyncio.run(synthesize_and_play(clean_answer))

                save_chat_log(question, answer, response.get("source_documents", []), selected_manifesto)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if 'source_documents' in response:
                    with st.expander("📚 View Source Sections"):
                        for i, doc in enumerate(response['source_documents']):
                            st.write(f"📄 Section {i + 1}:")
                            st.write(doc)
                            st.write("---")

            except Exception as e:
                st.error(f"⚠️ Error generating response: {str(e)}")
                st.exception(e)
