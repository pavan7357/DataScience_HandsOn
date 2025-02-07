import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn

# Load Model
class MultiModalModel(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", image_model_name="resnet18", numerical_input_size=2, num_classes=5):
        super(MultiModalModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.image_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_model.fc = nn.Identity()
        self.num_fc = nn.Linear(numerical_input_size, 128)
        self.classifier = nn.Linear(768 + 512 + 128, num_classes)

    def forward(self, text_tokens, image_tensor, numerical_data):
        text_features = self.text_model(**text_tokens).last_hidden_state[:, 0, :]
        image_features = self.image_model(image_tensor)
        numerical_features = self.num_fc(numerical_data)
        combined_features = torch.cat((text_features, image_features, numerical_features), dim=1)
        return self.classifier(combined_features)

# Load Trained Model
model = MultiModalModel(num_classes=5)
model.load_state_dict(torch.load("multi_modal_movie_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load Preprocessing Utilities
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# Genre Mapping Dictionary
genre_mapping = {
    0: "Action",
    1: "Drama",
    2: "Comedy",
    3: "Romance",
    4: "Sci-Fi"
}

# Streamlit UI
st.title("🎬 Multi-Modal Movie Genre Predictor")

# Text Input
description = st.text_area("📝 Enter Movie Description")

# Numerical Inputs
box_office = st.number_input("💰 Box Office Collection (in million USD)", min_value=0.0, step=0.1)
rating = st.number_input("⭐ IMDb Rating", min_value=1.0, max_value=10.0, step=0.1)

# Image Upload
image_file = st.file_uploader("🖼️ Upload Movie Poster", type=["jpg", "png"])

if st.button("🔮 Predict Genre"):
    if description and image_file and box_office and rating:
        # Preprocess Inputs
        text_tokens = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        image = Image.open(image_file).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0)
        numerical_tensor = torch.tensor([[rating, box_office]], dtype=torch.float32)

        # Make Predictions
        with torch.no_grad():
            predictions = model(text_tokens, image_tensor, numerical_tensor)
            predicted_label = torch.argmax(predictions, dim=1).item()

        # Convert Predicted Label to Genre Name
        genre_name = genre_mapping.get(predicted_label, "Unknown Genre")

        st.success(f"🎭 Predicted Genre: {genre_name}")
    else:
        st.warning("⚠️ Please provide all inputs!")
