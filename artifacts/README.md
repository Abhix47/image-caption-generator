# Model Artifacts

This directory contains files generated during model training.

## Generated Files

### `caption_model.pt`

The trained LSTM caption decoder.

### `vocab.json`

The vocabulary used by the captioning model, including the mapping between words and numerical token IDs.

### `image_features.pt`

Cached image features extracted from the Flickr8k images using the pretrained ResNet50 CNN.

## Important

These files are generated automatically when running:

```bash
python train.py 