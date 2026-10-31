"""
Optimized CNN + LSTM Image Caption Generator

Dataset expected:
dataset/Flickr8k_Dataset/
    Images/*.jpg
    captions.txt

Pipeline:
1. ResNet50 extracts a 2048-dimensional feature for each image ONCE.
2. Features are cached in artifacts/image_features.pt.
3. An LSTM decoder is trained on the cached features + captions.
"""

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = Path("dataset/Flickr8k_Dataset")
IMAGE_DIR = DATA_ROOT / "Images"
CAPTION_FILE = DATA_ROOT / "captions.txt"

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

FEATURE_FILE = ARTIFACTS / "image_features.pt"
VOCAB_FILE = ARTIFACTS / "vocab.json"
MODEL_FILE = ARTIFACTS / "caption_model.pt"

PAD, SOS, EOS, UNK = "<pad>", "<start>", "<end>", "<unk>"

FEATURE_BATCH_SIZE = 32
CAPTION_BATCH_SIZE = 64
EPOCHS = 8
EMBED_SIZE = 256
HIDDEN_SIZE = 512
LEARNING_RATE = 2e-4
MIN_WORD_FREQ = 2


# -----------------------------
# Text / vocabulary
# -----------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = [PAD, SOS, EOS, UNK]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def build(self, captions):
        counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))
        for word, count in sorted(counter.items()):
            if count >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def encode(self, caption):
        return (
            [self.stoi[SOS]]
            + [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(caption)]
            + [self.stoi[EOS]]
        )

    def __len__(self):
        return len(self.itos)


# -----------------------------
# Dataset parsing
# -----------------------------
def load_captions():
    if not CAPTION_FILE.exists():
        raise FileNotFoundError(f"Missing: {CAPTION_FILE}")

    pairs = []
    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if line_number == 0 and line.lower().startswith("image,"):
                continue

            image_name, caption = line.split(",", 1)
            image_name = image_name.strip()
            caption = caption.strip()

            if (IMAGE_DIR / image_name).exists():
                pairs.append((image_name, caption))

    if not pairs:
        raise RuntimeError("No valid image-caption pairs were found.")

    return pairs


def split_by_image(pairs):
    images = sorted({name for name, _ in pairs})
    rng = random.Random(SEED)
    rng.shuffle(images)

    n = len(images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_images = set(images[:train_end])
    val_images = set(images[train_end:val_end])
    test_images = set(images[val_end:])

    train_pairs = [(n, c) for n, c in pairs if n in train_images]
    val_pairs = [(n, c) for n, c in pairs if n in val_images]
    test_pairs = [(n, c) for n, c in pairs if n in test_images]

    return train_pairs, val_pairs, test_pairs


# -----------------------------
# Image preprocessing
# -----------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -----------------------------
# ResNet feature extraction
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, image_names):
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(IMAGE_DIR / name).convert("RGB")
        return image_transform(image), name


def build_feature_extractor():
    print("Loading pretrained ResNet50...")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    # Keep everything before the final classification layer.
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval().to(DEVICE)

    for p in feature_extractor.parameters():
        p.requires_grad = False

    return feature_extractor


@torch.no_grad()
def extract_features(image_names):
    if FEATURE_FILE.exists():
        print(f"Using cached features: {FEATURE_FILE}")
        data = torch.load(FEATURE_FILE, map_location="cpu", weights_only=False)
        return data["features"], data["names"]

    extractor = build_feature_extractor()

    loader = DataLoader(
        ImageDataset(image_names),
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_features = []
    all_names = []

    total = len(image_names)

    for batch_index, (images, names) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        features = extractor(images).flatten(1).cpu()

        all_features.append(features)
        all_names.extend(list(names))

        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
            done = min(batch_index * FEATURE_BATCH_SIZE, total)
            print(f"Feature extraction: {done}/{total} images")

    features = torch.cat(all_features, dim=0)

    torch.save(
        {"features": features, "names": all_names},
        FEATURE_FILE,
    )

    print(f"Saved {features.shape} features to {FEATURE_FILE}")
    return features, all_names


# -----------------------------
# Caption dataset using cached features
# -----------------------------
class FeatureCaptionDataset(Dataset):
    def __init__(self, pairs, feature_map, vocab):
        self.pairs = pairs
        self.feature_map = feature_map
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_name, caption = self.pairs[idx]
        feature = self.feature_map[image_name]
        tokens = torch.tensor(
            self.vocab.encode(caption),
            dtype=torch.long,
        )
        return feature, tokens


def make_collate(pad_index):
    def collate(batch):
        features, captions = zip(*batch)
        features = torch.stack(features)

        max_len = max(len(c) for c in captions)
        padded = torch.full(
            (len(captions), max_len),
            pad_index,
            dtype=torch.long,
        )

        for i, c in enumerate(captions):
            padded[i, :len(c)] = c

        return features, padded

    return collate


# -----------------------------
# LSTM decoder
# -----------------------------
class LSTMDecoder(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size):
        super().__init__()

        self.feature_projection = nn.Linear(feature_size, embed_size)

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        image_token = self.feature_projection(features).unsqueeze(1)
        word_embeddings = self.embedding(captions[:, :-1])

        inputs = torch.cat([image_token, word_embeddings], dim=1)

        outputs, _ = self.lstm(inputs)
        return self.fc(outputs[:, 1:, :])


# -----------------------------
# Main training
# -----------------------------
def main():
    print("=" * 65)
    print("OPTIMIZED CNN + LSTM IMAGE CAPTION GENERATOR")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    pairs = load_captions()
    print(f"Valid caption pairs: {len(pairs)}")

    train_pairs, val_pairs, test_pairs = split_by_image(pairs)

    print(
        f"Images/pairs split: "
        f"train={len(train_pairs)}, "
        f"val={len(val_pairs)}, "
        f"test={len(test_pairs)}"
    )

    vocab = Vocabulary(MIN_WORD_FREQ)
    vocab.build(c for _, c in train_pairs)
    print(f"Vocabulary size: {len(vocab)}")

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"itos": vocab.itos, "stoi": vocab.stoi},
            f,
            ensure_ascii=False,
        )

    unique_train_images = sorted({n for n, _ in train_pairs})
    # Extract all images once so the Streamlit app can use the same
    # feature pipeline and the cache can be reused.
    all_images = sorted({n for n, _ in pairs})
    features, feature_names = extract_features(all_images)
    feature_map = {
        name: features[i]
        for i, name in enumerate(feature_names)
    }

    collate_fn = make_collate(vocab.stoi[PAD])

    train_loader = DataLoader(
        FeatureCaptionDataset(train_pairs, feature_map, vocab),
        batch_size=CAPTION_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        FeatureCaptionDataset(val_pairs, feature_map, vocab),
        batch_size=CAPTION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    decoder = LSTMDecoder(
        feature_size=2048,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(vocab),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.stoi[PAD]
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=LEARNING_RATE,
    )

    best_val = math.inf

    print("\nStarting LSTM training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {CAPTION_BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        train_loss = 0.0

        for batch_idx, (features_batch, captions_batch) in enumerate(train_loader, 1):
            features_batch = features_batch.to(DEVICE)
            captions_batch = captions_batch.to(DEVICE)

            outputs = decoder(features_batch, captions_batch)
            targets = captions_batch[:, 1:]

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        avg_train = train_loss / len(train_loader)

        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features_batch, captions_batch in val_loader:
                features_batch = features_batch.to(DEVICE)
                captions_batch = captions_batch.to(DEVICE)

                outputs = decoder(features_batch, captions_batch)
                targets = captions_batch[:, 1:]

                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                )
                val_loss += loss.item()

        avg_val = val_loss / max(1, len(val_loader))

        print(
            f"\nEpoch {epoch}/{EPOCHS} "
            f"| Train Loss: {avg_train:.4f} "
            f"| Val Loss: {avg_val:.4f}\n"
        )

        if avg_val < best_val:
            best_val = avg_val

            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "feature_size": 2048,
                    "embed_size": EMBED_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "vocab_size": len(vocab),
                },
                MODEL_FILE,
            )

            print(f"Saved best model -> {MODEL_FILE}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print(f"Model: {MODEL_FILE}")
    print(f"Vocabulary: {VOCAB_FILE}")
    print(f"Cached features: {FEATURE_FILE}")


if __name__ == "__main__":
    main()
