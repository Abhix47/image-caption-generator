"""
Pretrained vision-language captioner (BLIP) used as the main model.

BLIP was pretrained on ~129M image-text pairs and fine-tuned on COCO
captions, so it generalizes to arbitrary photos far better than a decoder
trained from scratch on Flickr8k (8k images).
"""

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

BLIP_MODELS = {
    "BLIP-large (most accurate, ~1.9 GB)": "Salesforce/blip-image-captioning-large",
    "BLIP-base (faster, ~1 GB)": "Salesforce/blip-image-captioning-base",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_blip(model_id):
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model.eval().to(DEVICE)
    return processor, model


@torch.no_grad()
def blip_caption(
    image,
    processor,
    model,
    prompt=None,
    num_beams=5,
    max_new_tokens=40,
    num_return_sequences=3,
):
    """Return a list of candidate captions (best first) using beam search."""
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(DEVICE)
    else:
        inputs = processor(image, return_tensors="pt").to(DEVICE)

    num_return_sequences = min(num_return_sequences, num_beams)

    output = model.generate(
        **inputs,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_length=8,
        repetition_penalty=1.2,
        num_return_sequences=num_return_sequences,
    )
    captions = [
        processor.decode(o, skip_special_tokens=True).strip() for o in output
    ]
    # Remove duplicates, keep order.
    return list(dict.fromkeys(captions))
