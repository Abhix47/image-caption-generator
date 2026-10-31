"""
Image Caption Generator
Supports pretrained BLIP and Qwen image-captioning models,
image uploads, sample images, and an optional Flickr8k baseline.
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


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EXAMPLES_DIR = BASE_DIR / "examples"

MODEL_FILE = ARTIFACTS_DIR / "caption_model.pt"
VOCAB_FILE = ARTIFACTS_DIR / "vocab.json"

PAD = "<pad>"
SOS = "<start>"
EOS = "<end>"

ALL_MODELS = {**QWEN_MODELS, **BLIP_MODELS}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }

    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #7c5cff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #9aa3b5;
        font-size: 1.05rem;
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
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .caption-text {
        font-size: 1.45rem;
        font-weight: 600;
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    .chip {
        display: inline-block;
        background: #1f2740;
        border: 1px solid #2b3350;
        border-radius: 999px;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        color: #c9d1e6;
    }

    .placeholder {
        border: 2px dashed #2b3350;
        border-radius: 14px;
        padding: 3rem 1rem;
        text-align: center;
        color: #9aa3b5;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🖼️ Image Caption Generator</h1>
        <p>
            Upload an image or choose an example to generate
            a natural-language description using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="Loading AI model...")
def get_model(model_id):
    """Load the selected BLIP or Qwen model."""
    if model_id in QWEN_MODELS.values():
        return load_qwen(model_id)

    return load_blip(model_id)


# =========================================================
# OPTIONAL FLICKR8K CNN + LSTM BASELINE
# =========================================================

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
    """Load ResNet50 as a frozen image feature extractor."""
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    cnn = nn.Sequential(*list(resnet.children())[:-1])
    cnn.eval().to(DEVICE)

    for parameter in cnn.parameters():
        parameter.requires_grad = False

    return cnn


class LSTMDecoder(nn.Module):
    """LSTM decoder used by the optional Flickr8k baseline."""

    def __init__(
        self,
        feature_size,
        embed_size,
        hidden_size,
        vocab_size,
    ):
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

        self.fc = nn.Linear(hidden_size, vocab_size)

    def generate(self, feature, stoi, itos, max_len=30):
        feature_token = self.feature_projection(
            feature
        ).unsqueeze(1)

        token = torch.tensor(
            [[stoi[SOS]]],
            dtype=torch.long,
            device=feature.device,
        )

        _, hidden = self.lstm(feature_token, None)
        words = []

        for _ in range(max_len):
            output, hidden = self.lstm(
                self.embedding(token),
                hidden,
            )

            next_token = self.fc(
                output[:, -1, :]
            ).argmax(dim=-1)

            word = itos[int(next_token.item())]

            if word == EOS:
                break

            if word not in (PAD, SOS):
                words.append(word)

            token = next_token.unsqueeze(1)

        return " ".join(words)


@st.cache_resource
def load_lstm():
    """Load the trained Flickr8k baseline if its files exist."""
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


# =========================================================
# CAPTION CLEANUP
# =========================================================

def polish(text):
    """Clean the generated caption and format its punctuation."""
    text = " ".join(text.strip().split())

    for unwanted in ("arafed ", "araffe ", "arafed, "):
        if text.lower().startswith(unwanted):
            text = text[len(unwanted):].strip()

    if not text:
        return ""

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?":
        text += "."

    return text


# =========================================================
# EXAMPLE IMAGE DISCOVERY
# =========================================================

def get_example_images():
    """Find supported images inside the local examples folder."""
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


# =========================================================
# ADVANCED SETTINGS
# =========================================================

with st.expander("⚙️ Advanced settings"):
    column1, column2 = st.columns(2)

    model_name = column1.selectbox(
        "Captioning model",
        options=list(ALL_MODELS.keys()),
        index=len(QWEN_MODELS),
    )

    num_beams = column2.slider(
        "BLIP beam width",
        min_value=1,
        max_value=10,
        value=5,
        help="This setting applies to BLIP models only.",
    )

    prompt = st.text_input(
        "Optional prompt",
        placeholder="Enter an optional description instruction",
        help=(
            "Leave empty to use the model's default instruction. "
            "For Qwen, a custom prompt replaces its default instruction."
        ),
    )

    compare = st.checkbox(
        "Also show the Flickr8k CNN + LSTM baseline"
    )


# =========================================================
# IMAGE INPUT
# =========================================================

st.markdown("### Choose an image")

upload_tab, example_tab = st.tabs(
    ["📤 Upload your image", "🖼️ Try an example"]
)

selected_image = None
image_bytes = None
image_source = None

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()

            selected_image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            image_source = "upload_" + hashlib.md5(
                image_bytes
            ).hexdigest()

        except Exception:
            st.error(
                "Unable to open this image. Please select another file."
            )


with example_tab:
    example_images = get_example_images()

    if example_images:
        example_columns = st.columns(4)

        for index, example_path in enumerate(example_images[:4]):
            with example_columns[index]:
                try:
                    with Image.open(example_path) as sample:
                        st.image(
                            sample.convert("RGB"),
                            caption=example_path.stem.replace("_", " "),
                            use_container_width=True,
                        )

                    if st.button(
                        f"Use example {index + 1}",
                        key=f"example_button_{index}",
                    ):
                        st.session_state["chosen_example"] = str(
                            example_path
                        )
                        st.session_state["clear_upload"] = True
                        st.rerun()

                except Exception:
                    st.caption(
                        f"Could not display {example_path.name}"
                    )

        chosen_example = st.session_state.get("chosen_example")

        if chosen_example:
            chosen_path = Path(chosen_example)

            if chosen_path.exists():
                try:
                    image_bytes = chosen_path.read_bytes()

                    selected_image = ImageOps.exif_transpose(
                        Image.open(chosen_path)
                    ).convert("RGB")

                    image_source = "example_" + hashlib.md5(
                        image_bytes
                    ).hexdigest()

                    st.caption(
                        f"Selected example: {chosen_path.name}"
                    )

                except Exception:
                    st.error("Unable to open the selected example image.")

    else:
        st.info(
            "No sample images found. Create an 'examples' folder "
            "beside app.py and add JPG, PNG, or WebP images."
        )


# =========================================================
# IMAGE PREVIEW
# =========================================================

if selected_image is not None:
    st.divider()

    left_column, right_column = st.columns(
        [1, 1],
        gap="large",
    )

    with left_column:
        st.markdown("#### Selected image")

        st.image(
            selected_image,
            use_container_width=True,
        )

    with right_column:
        st.markdown("#### Generated caption")

        model_id = ALL_MODELS[model_name]

        result_key = (
            f"{image_source}|{model_name}|"
            f"{num_beams}|{prompt.strip()}"
        )

        if st.session_state.get("result_key") != result_key:
            try:
                with st.spinner(
                    "Analyzing image. The first run may take several minutes..."
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
                        "Sorry, no caption could be generated."
                    ]

                st.session_state["result_key"] = result_key
                st.session_state["captions"] = cleaned_captions

            except Exception as error:
                st.error(
                    "Caption generation failed. Check the selected "
                    "model and try again."
                )

                with st.expander("Technical error details"):
                    st.exception(error)

        captions = st.session_state.get("captions", [])

        if (
            captions
            and st.session_state.get("result_key") == result_key
        ):
            safe_caption = html.escape(captions[0])

            st.markdown(
                f"""
                <div class="caption-card">
                    <div class="caption-label">Caption</div>
                    <div class="caption-text">{safe_caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.code(captions[0], language=None)

            if len(captions) > 1:
                st.markdown("**Alternative captions**")

                for caption in captions[1:]:
                    safe_alternative = html.escape(caption)

                    st.markdown(
                        f'<span class="chip">{safe_alternative}</span>',
                        unsafe_allow_html=True,
                    )

            if st.button(
                "🔄 Regenerate caption",
                key="regenerate_button",
            ):
                st.session_state["result_key"] = None
                st.rerun()

            # =============================================
            # OPTIONAL BASELINE
            # =============================================

            if compare:
                st.divider()

                st.markdown(
                    "#### Flickr8k CNN + LSTM baseline"
                )

                baseline = load_lstm()

                if baseline is None:
                    st.info(
                        "The baseline is not trained or its model files "
                        "are missing. Run train.py and provide the required "
                        "artifacts to use this feature."
                    )

                else:
                    try:
                        decoder, stoi, itos = baseline

                        with torch.no_grad():
                            features = load_cnn()(
                                transform(selected_image)
                                .unsqueeze(0)
                                .to(DEVICE)
                            ).flatten(1)

                            baseline_caption = decoder.generate(
                                features,
                                stoi,
                                itos,
                            )

                        st.markdown("**Baseline caption**")
                        st.write(polish(baseline_caption))

                    except Exception as error:
                        st.warning(
                            "The baseline model could not generate a caption."
                        )

                        with st.expander("Technical error details"):
                            st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#7d869c; font-size:0.85rem;">
        Image Caption Generator · Powered by vision-language models
        <br>
        AI-generated captions may contain mistakes.
    </div>
    """,
    unsafe_allow_html=True,
)