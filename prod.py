import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from gtts import gTTS
# from io import BytesIO
from PyPDF2 import PdfReader
# import re
from datetime import datetime
# import asyncio
# import edge_tts
# import tempfile
from pymongo import MongoClient
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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
feedback_collection = db["feedback"]

# --------------------------- API Key ---------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --------------------------- Load Vector Store ---------------------------
@st.cache_resource(show_spinner=True)
def load_vectorstore_from_disk(folder_path):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

all_manifestos_store = load_vectorstore_from_disk("vectorstores_1.0/acts")


#---------------------------- Load Feedback collection--------------
def load_historical_feedback():
    feedback_data = list(feedback_collection.find({}))
    print(f"Loaded {len(feedback_data)} feedback entries from DB.")
    return feedback_data

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
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents([manifesto_doc], embeddings)

# --------------------------- Feedback Score Helper -------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def adjust_scores_with_feedback(query, retrieved_docs):
    feedback_data = load_historical_feedback()
    query_emb = embedding_model.encode([query])

    for doc in retrieved_docs:
        doc_emb = embedding_model.encode([doc.page_content])
        doc.metadata['relevance_score'] = 1.0  # baseline

        applicable_feedback = []
        for fb in feedback_data:
            fb_query_emb = embedding_model.encode([fb["query"]])
            sim = cosine_similarity(query_emb, fb_query_emb)[0][0]

            # consider only related feedback
            if sim > 0.5:  
                applicable_feedback.append(fb)

        if applicable_feedback:
            avg_score = sum((f["relevance_score"] + f["quality_score"]) for f in applicable_feedback) / (2 * len(applicable_feedback))
            adjustment = (avg_score - 3) / 3 * 0.3  # same as blog logic (30% boost)
            doc.metadata['relevance_score'] *= (1 + adjustment)

    # Re-rank based on adjusted scores
    return sorted(retrieved_docs, key=lambda x: x.metadata.get('relevance_score', 1.0), reverse=True)

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

# When the user sends a new message
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing manifestos..."):
            try:
                question = st.session_state.messages[-1]["content"]
                # response = retrieval_chain({
                #     "question": question,
                #     "chat_history": [
                #         (msg["role"], msg["content"])
                #         for msg in st.session_state.messages
                #         if msg["role"] != "assistant"
                #     ]
                # })

                # answer = response['answer']
                
                 # ------------------ Step 1: Retrieve documents ------------------
                docs = retriever.get_relevant_documents(question)

                # ------------------ Step 2: Apply feedback re-ranking ------------------
                docs = adjust_scores_with_feedback(question, docs)[:3]  # top 3 optimized docs

                # ------------------ Step 3: Generate response with top documents ------------------
                context = "\n\n".join([d.page_content for d in docs])
                response = chat_model.invoke(prompt.format(context=context, question=question))
                answer = response.content
                
                st.write(answer)
                # ------------------ Step 4: Display and save ------------------
                st.write(answer)
                save_chat_log(question, answer, docs, selected_manifesto)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # ------------------ Step 5: Source documents expander ------------------
                if docs:
                    with st.expander("📚 View Source Sections"):
                        for i, doc in enumerate(docs):
                            st.write(f"📄 Section {i + 1}:")
                            st.write(doc)
                            st.write("---")

                # Store last response and question in session_state for feedback
                st.session_state["last_question"] = question
                st.session_state["last_answer"] = answer
                st.session_state["last_response_docs"] = docs
                            
            except Exception as e:
                st.error(f"⚠️ Error generating response: {str(e)}")
                st.exception(e)

# ✅ Feedback expander should be shown even after rerun
if "last_answer" in st.session_state:
    with st.expander("💭 Give Feedback on this Response"):
        st.markdown("### 🙏 Help us improve Manifesto Analyzer!")
        relevance = st.slider("🔍 Relevance of references", 1, 5, 3, key="relevance_slider")
        quality = st.slider("✨ Overall quality of the answer", 1, 5, 3, key="quality_slider")

        if st.button("✅ Submit Feedback", key=f"feedback_{st.session_state.get('chat_id', 'default')}"):
            feedback_data = {
                "timestamp": datetime.utcnow(),
                "query": st.session_state["last_question"],
                "assistant_response": st.session_state["last_answer"],
                "used_manifesto": selected_manifesto if selected_manifesto != "All Manifestos" else None,
                "relevance_score": relevance,
                "quality_score": quality,
                "source_documents": [
                    doc.metadata.get("source", "unknown")
                    for doc in st.session_state.get("last_response_docs", [])
                ],
                "chat_id": st.session_state.get("chat_id", "default_session")
            }
            feedback_collection.insert_one(feedback_data)
            st.success("✅ Feedback submitted successfully! Thank you 💙")
