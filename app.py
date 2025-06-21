import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
    model.load_state_dict(torch.load('models/cgan_generator.pth', map_location='cpu'))
    model.eval()
    return model

# Generate digit
def generate_digit(model, digit, num_images=5):
    z = torch.randn(num_images, 100)
    labels = torch.tensor([digit] * num_images)
    with torch.no_grad():
        imgs = model(z, labels).squeeze().numpy()
    return imgs

# UI
st.title("Handwritten Digit Generator")
digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
model = load_generator()
images = generate_digit(model, int(digit))

# Display images
fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axs[i].imshow(images[i], cmap="gray")
    axs[i].axis("off")
st.pyplot(fig)
