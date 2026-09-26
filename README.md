# 🖼️ Image Caption Generator

> AI-powered image captioning using a ResNet50 CNN encoder and LSTM language decoder.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://image-captions-generator.streamlit.app/)

Upload an image and the model will generate a natural-language caption.

## 📌 Overview
...

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