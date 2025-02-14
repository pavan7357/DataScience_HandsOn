
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 🎯 AI Text Generation Model (Fine-tuned with LoRA)
MODEL_NAME = "gpt2"  # Use an open-source LLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

# LoRA Configuration (Optimized)
lora_config = LoraConfig(
    r=16,  # Higher rank for more expressivity
    lora_alpha=32,  # Stronger weight for adaptation
    target_modules=["attn.c_proj"],  
    lora_dropout=0.05  # Less dropout for stable training
)

model = get_peft_model(model, lora_config)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Amadeus API credentials (Replace with your own API key and secret)
API_KEY = "rqA3aNRf3YpGLGaJp0FYlgKoNLoBTQmr"
API_SECRET = "XNmPIcrdCfqwgq1U"

# Function to get Amadeus API token
def get_amadeus_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        return None

# Function to fetch flight prices from Amadeus API
def fetch_flights(origin, destination, departure_date):
    token = get_amadeus_token()
    if not token:
        return []

    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}

    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "adults": 1
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        flights = []
        for flight in data.get("data", [])[:5]:  # Retrieve top 5 flights
            flights.append({
                "Airline": flight["validatingAirlineCodes"][0],
                "Price": f"${flight['price']['total']}",
                "Departure": flight["itineraries"][0]["segments"][0]["departure"]["at"],
                "Arrival": flight["itineraries"][0]["segments"][-1]["arrival"]["at"],
                "Duration": flight["itineraries"][0]["duration"],
                "Stops": len(flight["itineraries"][0]["segments"]),
                "Route": f"{origin} to {destination}"
            })
        return flights
    else:
        return []

# 📌 Fetch Flight Data
flights_data = fetch_flights("JFK", "MCI", "2025-02-17")
df = pd.DataFrame(flights_data)

# 🔍 AI-Powered Flight Search with FAISS
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# Convert flight data into descriptions
flight_descriptions = [
    f"Flight from {row['Route']} by {row['Airline']} costs {row['Price']}, "
    f"departure at {row['Departure']}, arrival at {row['Arrival']}, "
    f"duration {row['Duration']}, with {row['Stops']} stops."
    for _, row in df.iterrows()
]

if not flight_descriptions:
    flight_descriptions = ["No flight data available."]

embeddings = model_embed.encode(flight_descriptions)

# FAISS Index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings, dtype=np.float32))

def search_flights(query):
    query_embedding = model_embed.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=3)
    results = [flight_descriptions[i] for i in I[0] if i < len(flight_descriptions)]
    return results if results else ["No matching flights found."]

# 🤖 AI Text Generation for Travel Recommendations
def generate_ai_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids, 
        max_length=50,  # Set a reasonable max length
        num_return_sequences=1,  # Generate only one response
        no_repeat_ngram_size=3,  # Prevent n-gram repetition
        temperature=0.7,  # Add randomness to avoid repetition
        top_k=50,  # Filter unlikely words
        top_p=0.9,  # Consider only highly probable tokens
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# 🌟 Beautified Streamlit UI
st.set_page_config(page_title="AI Flight Recommender", page_icon="✈️", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #3498db;'>✈️ AI-Powered Flight Recommender</h1>
    <p style='text-align: center; font-size: 18px;'>Find the best flights & get AI-powered travel insights!</p>
    """,
    unsafe_allow_html=True
)

st.write("### Enter your flight query below:")

user_query = st.text_input("")

if st.button("🔍 Find Flights"):
    search_results = search_flights(user_query)
    
    st.subheader("✈️ Matching Flights:")
    for result in search_results:
        st.success(result)

if st.button("🤖 AI Travel Advice"):
    ai_response = generate_ai_response(user_query)
    st.subheader("💡 AI Travel Insights:")
    st.info(ai_response)


