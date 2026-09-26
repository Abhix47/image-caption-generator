# 🖼️ Image Caption Generator

An AI-based image captioning system that automatically generates
natural-language descriptions for images using a deep learning
architecture combining a ResNet50 CNN and an LSTM decoder.

## 📌 Overview

Image captioning combines Computer Vision and Natural Language
Processing to generate a textual description of an image.

This project uses:

- ResNet50 as the CNN-based vision encoder
- LSTM as the language decoder
- Flickr8k image-caption dataset
- PyTorch for deep learning
- Streamlit for the web interface

The CNN extracts visual features from the input image, and the
LSTM generates the caption sequentially, one word at a time.

---

## 🧠 System Architecture

```text
              Input Image
                   │
                   ▼
          Image Preprocessing
                   │
                   ▼
             ResNet50 CNN
            Vision Encoder
                   │
                   ▼
          2048-D Image Features
                   │
                   ▼
            Feature Projection
                   │
                   ▼
             LSTM Decoder
                   │
                   ▼
       Word-by-Word Generation
                   │
                   ▼
          Generated Caption