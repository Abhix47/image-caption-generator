"""
Image Caption Generator - Streamlit app.

Uses a pretrained BLIP vision-language model with beam search.
The Flickr8k ResNet50 + LSTM baseline is available under "Advanced"
if you have trained it (python train.py).
"""

import hashlib
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

from blip_captioner import BLIP_MODELS, DEVICE, blip_caption, load_blip
from qwen_captioner import QWEN_MODELS, load_qwen, qwen_caption

ALL_MODELS = {**QWEN_MODELS, **BLIP_MODELS}

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE = ARTIFACTS / "caption_model.pt"
VOCAB_FILE = ARTIFACTS / "vocab.json"
PAD, SOS, EOS = "<pad>", "<start>", "<end>"

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {text-align: center; padding: 1.2rem 0 1.6rem 0;}
.hero h1 {
    font-size: 2.8rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #7c5cff, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {color: #9aa3b5; font-size: 1.05rem; margin-top: .4rem;}
.caption-card {
    background: linear-gradient(135deg, #1b2133, #161b26);
    border: 1px solid #2b3350; border-left: 5px solid #7c5cff;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.caption-label {
    color: #7c5cff; font-size: .8rem; letter-spacing: .12em;
    text-transform: uppercase; font-weight: 700;
}
.caption-text {font-size: 1.6rem; font-weight: 600; line-height: 1.4; margin-top: .4rem;}
.chip {
    display: inline-block; background: #1f2740; border: 1px solid #2b3350;
    border-radius: 999px; padding: .3rem .8rem; margin: .2rem .3rem .2rem 0;
    font-size: .9rem; color: #c9d1e6;
}
.placeholder {
    border: 2px dashed #2b3350; border-radius: 14px; padding: 3rem 1rem;
    text-align: center; color: #7d869c;
}
div.stButton > button {width: 100%; border-radius: 10px; font-weight: 700; padding: .6rem;}
[data-testid="stImage"] img {border-radius: 14px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🖼️ Image Caption Generator</h1>
  <p>Upload any photo and get a natural-language description, powered by a pretrained BLIP vision-language model.</p>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Models
# -----------------------------
@st.cache_resource(show_spinner="Loading model (first run downloads weights, please wait)...")
def get_model(model_id):
    if model_id in QWEN_MODELS.values():
        return load_qwen(model_id)
    return load_blip(model_id)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_cnn():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    cnn = nn.Sequential(*list(resnet.children())[:-1])
    cnn.eval().to(DEVICE)
    for p in cnn.parameters():
        p.requires_grad = False
    return cnn


class LSTMDecoder(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.feature_projection = nn.Linear(feature_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def generate(self, feature, stoi, itos, max_len=30):
        feature_token = self.feature_projection(feature).unsqueeze(1)
        token = torch.tensor([[stoi[SOS]]], dtype=torch.long, device=feature.device)
        _, hidden = self.lstm(feature_token, None)
        words = []
        for _ in range(max_len):
            output, hidden = self.lstm(self.embedding(token), hidden)
            next_token = self.fc(output[:, -1, :]).argmax(dim=-1)
            word = itos[int(next_token.item())]
            if word == EOS:
                break
            if word not in (PAD, SOS):
                words.append(word)
            token = next_token.unsqueeze(1)
        return " ".join(words)


@st.cache_resource
def load_lstm():
    if not MODEL_FILE.exists() or not VOCAB_FILE.exists():
        return None
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
    decoder = LSTMDecoder(
        ckpt["feature_size"], ckpt["embed_size"],
        ckpt["hidden_size"], ckpt["vocab_size"],
    ).to(DEVICE)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    return decoder, vocab["stoi"], vocab["itos"]


def polish(text):
    """Clean known BLIP artifacts and format as a sentence."""
    text = text.strip()
    for junk in ("arafed ", "araffe ", "arafed, "):
        if text.lower().startswith(junk):
            text = text[len(junk):]
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


# -----------------------------
# Advanced settings
# -----------------------------
with st.expander("⚙️ Advanced settings"):
    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", list(ALL_MODELS.keys()), index=len(QWEN_MODELS))
    num_beams = c2.slider("BLIP beam width (BLIP only)", 1, 10, 5)
    prompt = st.text_input(
        "Optional prompt",
        placeholder="BLIP: caption start, e.g. 'a photography of'. Qwen: custom instruction.",
        help="Leave empty normally. For Qwen, this replaces the default description instruction.",
    )
    compare = st.checkbox("Also show the Flickr8k CNN+LSTM baseline")

# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    uploaded = st.file_uploader(
        "Drop a photo here", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    if uploaded is not None:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        st.image(image, use_container_width=True)

with right:
    if uploaded is None:
        st.markdown(
            '<div class="placeholder">📤 Upload a JPG, PNG or WebP image<br>'
            "and the caption will appear here.</div>",
            unsafe_allow_html=True,
        )
    else:
        key = hashlib.md5(uploaded.getvalue()).hexdigest() + f"|{model_name}|{num_beams}|{prompt}"

        # Generate automatically on upload; cache per image + settings.
        if st.session_state.get("result_key") != key:
            model_id = ALL_MODELS[model_name]
            processor, vlm = get_model(model_id)
            with st.spinner("Analyzing image (Qwen can take up to a minute on CPU)..."):
                if model_id in QWEN_MODELS.values():
                    caps = qwen_caption(
                        image, processor, vlm, prompt=prompt.strip() or None
                    )
                else:
                    caps = blip_caption(
                        image, processor, vlm,
                        prompt=prompt.strip() or None,
                        num_beams=num_beams,
                    )
            st.session_state["result_key"] = key
            st.session_state["captions"] = [polish(c) for c in caps if polish(c)]

        captions = st.session_state["captions"]

        st.markdown(
            f'<div class="caption-card"><div class="caption-label">Caption</div>'
            f'<div class="caption-text">{captions[0]}</div></div>',
            unsafe_allow_html=True,
        )
        st.code(captions[0], language=None)  # built-in copy button

        if len(captions) > 1:
            st.markdown("**Alternative phrasings**")
            st.markdown(
                "".join(f'<span class="chip">{c}</span>' for c in captions[1:]),
                unsafe_allow_html=True,
            )

        if st.button("🔄 Regenerate"):
            st.session_state["result_key"] = None
            st.rerun()

        if compare:
            st.divider()
            st.markdown("**Baseline: ResNet50 + LSTM (Flickr8k)**")
            baseline = load_lstm()
            if baseline is None:
                st.info("Baseline not trained. Run `python train.py` first.")
            else:
                decoder, stoi, itos = baseline
                with torch.no_grad():
                    feat = load_cnn()(transform(image).unsqueeze(0).to(DEVICE)).flatten(1)
                    st.write(polish(decoder.generate(feat, stoi, itos)))
