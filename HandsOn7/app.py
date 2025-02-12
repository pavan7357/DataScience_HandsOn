
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("flights_data.csv")

# Combine relevant fields into text
df["text"] = (
    "Flight from " + df["Route"] + 
    ", Airline: " + df["Airline"] +
    ", Price: $" + df["Price"].astype(str) +
    ", Duration: " + df["Duration"] +
    ", Stops: " + df["Stops"].astype(str)
)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("flight_faiss_index.bin")

# ✅ Fix: Define the function inside `app.py`
def retrieve_flights(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]]
    return results

# Streamlit UI
st.title("✈ AI-Powered Flight Search System")

query = st.text_input("🔍 Enter your flight preference (e.g., 'Cheapest direct flight from JFK to LAX'): ")

if query:
    results = retrieve_flights(query)
    st.subheader("💰 Matching Flights")
    for _, row in results.iterrows():
        st.write(f"🛫 **Airline:** {row['Airline']}")
        st.write(f"💰 **Price:** ${row['Price']}")
        st.write(f"📍 **From:** {row['Route']}")
        st.write(f"🕒 **Duration:** {row['Duration']}")
        st.write(f"🚏 **Stops:** {row['Stops']}")
        st.write("---")

