# dashboard.py
import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
from datetime import datetime

# --------------------------- Config ---------------------------
MONGO_URI = st.secrets["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client["manifesto_ai"]
logs_collection = db["chat_logs"]
feedback_collection = db["feedback"]

st.set_page_config(page_title="Manifesto Analyzer Dashboard", layout="wide")
st.title("📊 Manifesto Analyzer Dashboard")
st.markdown("Visual insights from user queries and feedback data.")

# --------------------------- Load Data ---------------------------
@st.cache_data(show_spinner=True)
def load_logs():
    logs = list(logs_collection.find({}))
    return pd.DataFrame(logs)

@st.cache_data(show_spinner=True)
def load_feedback():
    feedback = list(feedback_collection.find({}))
    return pd.DataFrame(feedback)

logs_df = load_logs()
feedback_df = load_feedback()

# --------------------------- Queries Over Time ---------------------------
st.subheader("📈 Queries Over Time")
if not logs_df.empty:
    logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
    queries_over_time = logs_df.groupby(logs_df['timestamp'].dt.date).size().reset_index(name='num_queries')
    fig_queries = px.line(queries_over_time, x='timestamp', y='num_queries', markers=True,
                          labels={'timestamp':'Date', 'num_queries':'Number of Queries'},
                          title="Daily Queries Trend")
    st.plotly_chart(fig_queries, use_container_width=True)
else:
    st.info("No query data available yet.")

# --------------------------- Feedback Distribution ---------------------------
st.subheader("⭐ Feedback Distribution")
if not feedback_df.empty:
    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
    col1, col2 = st.columns(2)
    with col1:
        fig_rel = px.histogram(feedback_df, x='relevance_score', nbins=5, title="Relevance Score Distribution")
        st.plotly_chart(fig_rel, use_container_width=True)
    with col2:
        fig_qual = px.histogram(feedback_df, x='quality_score', nbins=5, title="Quality Score Distribution")
        st.plotly_chart(fig_qual, use_container_width=True)
else:
    st.info("No feedback data available yet.")

# --------------------------- Top Parties / Manifestos ---------------------------
st.subheader("🏛️ Feedback per Manifesto / Party")
if not feedback_df.empty:
    party_feedback = feedback_df.groupby('used_manifesto')[['relevance_score', 'quality_score']].mean().reset_index()
    fig_party = px.bar(party_feedback, x='used_manifesto', y=['relevance_score','quality_score'],
                       barmode='group', title="Average Feedback per Party/Manifesto")
    st.plotly_chart(fig_party, use_container_width=True)
else:
    st.info("No manifesto feedback yet.")

# --------------------------- Most Asked Queries ---------------------------
st.subheader("💬 Most Asked Queries")
if not logs_df.empty:
    top_queries = logs_df['user_question'].value_counts().head(10).reset_index()
    top_queries.columns = ['query', 'count']
    fig_top_queries = px.bar(top_queries, x='query', y='count', title="Top 10 User Questions")
    st.plotly_chart(fig_top_queries, use_container_width=True)
else:
    st.info("No queries yet.")

# --------------------------- Optional Filters ---------------------------
st.sidebar.header("Filters")
date_filter = st.sidebar.date_input("Select Date (optional)")
if date_filter:
    filtered_logs = logs_df[logs_df['timestamp'].dt.date == date_filter]
    st.write(f"Number of queries on {date_filter}: {len(filtered_logs)}")
