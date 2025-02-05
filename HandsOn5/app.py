import streamlit as st
import google.generativeai as genai
from PIL import Image
import os

# Set up API key
genai.configure(api_key=os.getenv("AIzaSyA7Jnfr9CljQGvCP9ys0lp00gdK_U8_toU"))
model = genai.GenerativeModel("gemini-1.5-pro")

# Function to generate text response
def generate_text_response(prompt):
    response = model.generate_content(prompt)
    return response.text

# Function to process an image
def process_image(image):
    img = Image.open(image)  # Handle Streamlit uploaded file
    response = model.generate_content(["Describe this image.", img])  # Provide a basic text prompt
    return response.text

# Function to process text + image
def process_text_image(image, text_prompt):
    img = Image.open(image)
    response = model.generate_content([text_prompt, img])  # Combine prompt & image
    return response.text

# Streamlit UI
st.title("Gemini 1.5 Pro Multi-Modal AI Hands-On")
st.subheader("Bring Your Own Dataset (Kaggle, Local, APIs)")

# Text Processing
st.write("## Text Processing")
text_input = st.text_area("Enter a prompt:")
if st.button("Generate Text Response"):
    st.write(generate_text_response(text_input))

# Image Processing
st.write("## Image Processing")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Analyze Image"):
        st.write(process_image(uploaded_file))

# Text + Image Processing
st.write("## Text + Image Processing")
text_prompt_2 = st.text_area("Enter a text prompt for the image:")
if uploaded_file and text_prompt_2 and st.button("Analyze Text + Image"):
    st.write(process_text_image(uploaded_file, text_prompt_2))

# Kaggle Dataset Integration
st.write("## Load Kaggle Dataset")
dataset_name = st.text_input("Enter Kaggle dataset name:")
if st.button("Download Dataset"):
    os.system(f"mkdir -p ./data && kaggle datasets download -d {dataset_name} -p ./data && unzip ./data/{dataset_name}.zip -d ./data")
    st.write(f"Downloaded {dataset_name} successfully!")
EOF
