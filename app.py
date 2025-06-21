import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# Generator architecture (same as training)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_embed(labels)], 1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# Load model
@st.cache_resource
def load_generator():
    model = Generator()
    model_path = "models/cgan_generator.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload cgan_generator.pth to the models/ directory.")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Generate digit images
def generate_digit(model, digit, num_images=5):
    z = torch.randn(num_images, 100)
    labels = torch.tensor([digit] * num_images)
    with torch.no_grad():
        imgs = model(z, labels).squeeze().numpy()
    return imgs

# Streamlit UI
st.title("🧠 Handwritten Digit Generator (0–9)")
st.write("This app uses a Conditional GAN trained on MNIST to generate handwritten digits.")

digit = st.selectbox("Select a digit to generate:", list(range(10)))
model = load_generator()
images = generate_digit(model, int(digit), num_images=5)

# Display generated images
st.subheader(f"Generated images of digit {digit}")
fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axs[i].imshow(images[i], cmap="gray")
    axs[i].axis("off")
st.pyplot(fig)
