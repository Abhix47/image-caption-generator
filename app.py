"""
Streamlit demo for the optimized CNN + LSTM image caption generator.

The CNN (ResNet50) extracts a 2048-D image feature.
The trained LSTM decoder generates the caption one word at a time.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import streamlit as st


ARTIFACTS = Path("artifacts")
MODEL_FILE = ARTIFACTS / "caption_model.pt"
VOCAB_FILE = ARTIFACTS / "vocab.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD = "<pad>"
SOS = "<start>"
EOS = "<end>"
UNK = "<unk>"


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# -----------------------------
# CNN feature extractor
# -----------------------------
@st.cache_resource
def load_cnn():
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    cnn = nn.Sequential(*list(resnet.children())[:-1])
    cnn.eval().to(DEVICE)

    for p in cnn.parameters():
        p.requires_grad = False

    return cnn


# -----------------------------
# LSTM decoder
# -----------------------------
class LSTMDecoder(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size):
        super().__init__()

        self.feature_projection = nn.Linear(
            feature_size,
            embed_size,
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def generate(
        self,
        feature,
        stoi,
        itos,
        max_len=30,
    ):
        feature_token = self.feature_projection(feature).unsqueeze(1)

        token = torch.tensor(
            [[stoi[SOS]]],
            dtype=torch.long,
            device=feature.device,
        )

        hidden = None
        words = []

        # Feed the image feature first.
        _, hidden = self.lstm(
            feature_token,
            hidden,
        )

        for _ in range(max_len):
            embedding = self.embedding(token)

            output, hidden = self.lstm(
                embedding,
                hidden,
            )

            logits = self.fc(output[:, -1, :])

            next_token = logits.argmax(dim=-1)

            index = int(next_token.item())
            word = itos[index]

            if word == EOS:
                break

            if word not in (PAD, SOS):
                words.append(word)

            token = next_token.unsqueeze(1)

        return " ".join(words)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists() or not VOCAB_FILE.exists():
        return None

    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    decoder = LSTMDecoder(
        checkpoint["feature_size"],
        checkpoint["embed_size"],
        checkpoint["hidden_size"],
        checkpoint["vocab_size"],
    ).to(DEVICE)

    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    return decoder, vocab["stoi"], vocab["itos"]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Image Caption Generator")
st.caption(
    "CNN (ResNet50) vision encoder + LSTM language decoder"
)

st.write(
    "Upload an image and the model will generate a natural-language caption."
)

model_data = load_model()

if model_data is None:
    st.error(
        "Trained model not found. Run `python train.py` first."
    )
    st.stop()

decoder, stoi, itos = model_data

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True,
    )

    if st.button(
        "Generate Caption",
        type="primary",
    ):
        with st.spinner("Analyzing image..."):
            cnn = load_cnn()

            image_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feature = cnn(image_tensor).flatten(1)

                caption = decoder.generate(
                    feature,
                    stoi,
                    itos,
                )

        st.success("Caption generated")
        st.markdown(
            f"### 💬 {caption}"
        )
