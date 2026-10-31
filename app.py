"""
Image Caption Generator - Streamlit app.

Features:
- Pretrained BLIP image-captioning models
- Qwen2-VL-2B image-captioning model
- Upload an image or select an example image
- Alternative captions and regeneration
- Optional Flickr8k ResNet50 + LSTM baseline

Required files:
    app.py
    blip_captioner.py
    qwen_captioner.py
    requirements.txt
    .streamlit/config.toml

Optional:
    examples/          Sample images
    artifacts/
        caption_model.pt
        vocab.json
"""

import hashlib
import html
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

from blip_captioner import BLIP_MODELS, DEVICE, blip_caption, load_blip
from qwen_captioner import QWEN_MODELS, load_qwen, qwen_caption


# --------------------------------------------------
# Paths and model configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
EXAMPLES_DIR = BASE_DIR / "examples"

MODEL_FILE = ARTIFACTS / "caption_model.pt"
VOCAB_FILE = ARTIFACTS / "vocab.json"

PAD, SOS, EOS = "<pad>", "<start>", "<end>"

ALL_MODELS = {**QWEN_MODELS, **BLIP_MODELS}

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    .hero {
        text-align: center;
        padding: 1.2rem 0 1.6rem 0;
    }

    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #7c5cff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #9aa3b5;
        font-size: 1.05rem;
        margin-top: .6rem;
    }

    .section-heading {
        font-size: 1.15rem;
        font-weight: 700;
        margin: .5rem 0 1rem 0;
    }

    .caption-card {
        background: linear-gradient(135deg, #1b2133, #161b26);
        border: 1px solid #2b3350;
        border-left: 5px solid #7c5cff;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        overflow-wrap: anywhere;
    }

    .caption-label {
        color: #7c5cff;
        font-size: .8rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .caption-text {
        font-size: 1.45rem;
        font-weight: 600;
        line-height: 1.5;
        margin-top: .5rem;
    }

    .chip {
        display: inline-block;
        background: #1f2740;
        border: 1px solid #2b3350;
        border-radius: 999px;
        padding: .45rem .85rem;
        margin: .2rem .3rem .2rem 0;
        font-size: .9rem;
        color: #c9d1e6;
    }

    .placeholder {
        border: 2px dashed #2b3350;
        border-radius: 14px;
        padding: 3rem 1rem;
        text-align: center;
        color: #9aa3b5;
    }

    .info-card {
        background: #161b26;
        border: 1px solid #2b3350;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #c9d1e6;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: .6rem;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🖼️ Image Caption Generator</h1>
        <p>
            Turn your images into natural-language descriptions
            using pretrained vision-language models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Caption models
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading model and downloading weights if needed...")
def get_model(model_id):
    """Load the selected pretrained captioning model."""
    if model_id in QWEN_MODELS.values():
        return load_qwen(model_id)

    return load_blip(model_id)


# --------------------------------------------------
# Flickr8k CNN + LSTM baseline
# --------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


@st.cache_resource
def load_cnn():
    """Load the pretrained ResNet50 feature extractor."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    cnn = nn.Sequential(*list(resnet.children())[:-1])
    cnn.eval().to(DEVICE)

    for parameter in cnn.parameters():
        parameter.requires_grad = False

    return cnn


class LSTMDecoder(nn.Module):
    """ResNet feature projection followed by an LSTM decoder."""

    def __init__(self, feature_size, embed_size, hidden_size, vocab_size):
        super().__init__()

        self.feature_projection = nn.Linear(feature_size, embed_size)
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
        self.fc = nn.Linear(hidden_size, vocab_size)

    def generate(self, feature, stoi, itos, max_len=30):
        feature_token = self.feature_projection(feature).unsqueeze(1)

        token = torch.tensor(
            [[stoi[SOS]]],
            dtype=torch.long,
            device=feature.device,
        )

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
    """Load trained Flickr8k baseline artifacts if available."""
    if not MODEL_FILE.exists() or not VOCAB_FILE.exists():
        return None

    with open(VOCAB_FILE, "r", encoding="utf-8") as file:
        vocab = json.load(file)

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


# --------------------------------------------------
# Caption cleanup
# --------------------------------------------------

def polish(text):
    """Clean generated text and format it as a sentence."""
    text = " ".join(text.strip().split())

    for junk in ("arafed ", "araffe ", "arafed, "):
        if text.lower().startswith(junk):
            text = text[len(junk):].strip()

    if not text:
        return ""

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?":
        text += "."

    return text


# --------------------------------------------------
# Example images
# --------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_example_images():
    """Return supported sample images from the examples folder."""
    if not EXAMPLES_DIR.exists():
        return []

    return sorted(
        [
            path
            for path in EXAMPLES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


example_images = get_example_images()


# --------------------------------------------------
# Advanced settings
# --------------------------------------------------

with st.expander("⚙️ Advanced settings"):
    model_col, beam_col = st.columns(2)

    model_name = model_col.selectbox(
        "Captioning model",
        options=list(ALL_MODELS.keys()),
        index=len(QWEN_MODELS),
    )

    num_beams = beam_col.slider(
        "BLIP beam width (BLIP only)",
        min_value=1,
        max_value=10,
        value=5,
    )

    prompt = st.text_input(
        "Optional prompt",
        placeholder="Enter an optional caption prompt or instruction",
        help=(
            "Leave empty for the default instruction. "
            "For Qwen, this can replace its default description instruction."
        ),
    )

    compare = st.checkbox(
        "Also show the Flickr8k CNN + LSTM baseline",
        value=False,
    )


# --------------------------------------------------
# Image selection
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Choose an image</div>',
    unsafe_allow_html=True,
)

upload_tab, example_tab = st.tabs(
    ["📤 Upload your image", "🖼️ Try an example"]
)

selected_image = None
image_source = None

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Maximum upload size: 20 MB.",
        key="image_uploader",
    )

    if uploaded_file is not None:
        try:
            selected_image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            image_source = hashlib.md5(
                uploaded_file.getvalue()
            ).hexdigest()

        except Exception:
            st.error("Unable to open this image. Please choose another file.")
            selected_image = None


with example_tab:
    if example_images:
        image_options = {
            path.name: path for path in example_images
        }

        selected_example_name = st.selectbox(
            "Select a sample image",
            options=list(image_options.keys()),
            key="example_image",
        )

        selected_example_path = image_options[selected_example_name]

        try:
            selected_image = ImageOps.exif_transpose(
                Image.open(selected_example_path)
            ).convert("RGB")

            image_source = hashlib.md5(
                selected_example_path.read_bytes()
            ).hexdigest()

        except Exception:
            st.error("Unable to open this example image.")
            selected_image = None
    else:
        st.info(
            "No example images found. Add JPG, PNG, or WebP images "
            "to the `examples` folder in your repository."
        )


# --------------------------------------------------
# Generate caption
# --------------------------------------------------

if selected_image is None:
    st.markdown(
        """
        <div class="placeholder">
            📷<br><br>
            Choose an example image or upload your own image.
            <br>
            The generated caption will appear here.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Selected image")
        st.image(
            selected_image,
            use_container_width=True,
        )

    with right:
        model_id = ALL_MODELS[model_name]

        result_key = (
            f"{image_source}|{model_name}|{num_beams}|{prompt.strip()}"
        )

        if st.session_state.get("result_key") != result_key:
            try:
                with st.spinner(
                    "Loading model and generating caption. "
                    "The first run may take several minutes..."
                ):
                    processor, model = get_model(model_id)

                    if model_id in QWEN_MODELS.values():
                        generated_captions = qwen_caption(
                            selected_image,
                            processor,
                            model,
                            prompt=prompt.strip() or None,
                        )
                    else:
                        generated_captions = blip_caption(
                            selected_image,
                            processor,
                            model,
                            prompt=prompt.strip() or None,
                            num_beams=num_beams,
                        )

                cleaned_captions = []
                for caption in generated_captions:
                    cleaned = polish(caption)
                    if cleaned and cleaned not in cleaned_captions:
                        cleaned_captions.append(cleaned)

                if not cleaned_captions:
                    cleaned_captions = [
                        "No caption could be generated for this image."
                    ]

                st.session_state["result_key"] = result_key
                st.session_state["captions"] = cleaned_captions

            except Exception as error:
                st.error(
                    "The caption could not be generated. "
                    "Try another model or image."
                )
                st.exception(error)
                st.stop()

        captions = st.session_state.get("captions", [])

        if captions:
            safe_caption = html.escape(captions[0])

            st.markdown(
                f"""
                <div class="caption-card">
                    <div class="caption-label">Generated caption</div>
                    <div class="caption-text">{safe_caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.code(captions[0], language=None)

            if len(captions) > 1:
                st.markdown("**Alternative captions**")

                for caption in captions[1:]:
                    st.markdown(
                        f'<span class="chip">{html.escape(caption)}</span>',
                        unsafe_allow_html=True,
                    )

            if st.button("🔄 Regenerate caption", key="regenerate"):
                st.session_state["result_key"] = None
                st.rerun()

            # ------------------------------------------
            # Optional Flickr8k baseline
            # ------------------------------------------

            if compare:
                st.divider()
                st.markdown("#### Flickr8k CNN + LSTM baseline")

                baseline = load_lstm()

                if baseline is None:
                    st.info(
                        "The baseline model is not available. "
                        "To use it, train the model with `python train.py` "
                        "and include the required files in the `artifacts` folder."
                    )
                else:
                    try:
                        decoder, stoi, itos = baseline

                        with torch.no_grad():
                            features = (
                                load_cnn()(
                                    transform(selected_image)
                                    .unsqueeze(0)
                                    .to(DEVICE)
                                )
                                .flatten(1)
                            )

                            baseline_caption = decoder.generate(
                                features,
                                stoi,
                                itos,
                            )

                        st.markdown(
                            f"""
                            <div class="info-card">
                                <strong>Baseline caption</strong><br><br>
                                {html.escape(polish(baseline_caption))}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except Exception as error:
                        st.warning(
                            "The Flickr8k baseline could not be loaded. "
                            "Check that its checkpoint and vocabulary match."
                        )
                        st.exception(error)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#7d869c; font-size:.85rem;">
        Image Caption Generator · Powered by pretrained vision-language models
        <br>
        Generated captions may contain mistakes and should be verified.
    </div>
    """,
    unsafe_allow_html=True,
)