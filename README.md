# 🖼️ Image Caption Generator

> Image captioning with a pretrained BLIP vision-language model, compared against a ResNet50 + LSTM baseline trained from scratch on Flickr8k.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://image-captions-generator.streamlit.app/)

## Overview

Image captioning combines Computer Vision and NLP to describe an image in a sentence.

| Model | Role | Training data |
|---|---|---|
| **BLIP (large/base)** | Main model | Pretrained on ~129M image-text pairs, fine-tuned on COCO captions |
| ResNet50 + LSTM | Baseline | Trained here on Flickr8k (~8k images) |

### Why two models?

The ResNet50 + LSTM baseline shows the classic encoder-decoder approach, but with
only ~8k training images it cannot generalize to arbitrary photos. BLIP uses a
Vision Transformer encoder and a transformer text decoder, pretrained at scale,
and decodes with beam search. This is the "before vs. after" of the project.

## Architecture

```text
Input image -> preprocessing -> ViT image encoder -> cross-attention text decoder
            -> beam search (width 5) -> caption
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads BLIP weights from Hugging Face (large ≈ 1.9 GB, base ≈ 1 GB).

Optional baseline (needs the Flickr8k dataset in `dataset/Flickr8k_Dataset/`):

```bash
python train.py
```

Then tick "Also show Flickr8k CNN+LSTM baseline" in the app sidebar.

## Limitations

- Captions are generic, one-sentence descriptions; fine detail, text in images, counts and rare objects can be wrong.
- BLIP-large needs ~2-4 GB RAM; the free Streamlit Community Cloud tier (~1 GB) may crash on it. Run locally or use a Hugging Face Space for the demo.
