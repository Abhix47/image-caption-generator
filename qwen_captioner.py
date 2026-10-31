"""
Stronger captioner: Qwen2-VL-2B-Instruct (vision-language model).

Much larger and more capable than BLIP, so it makes far fewer mistakes on
objects, counts, colors and actions. The prompt tells it to describe only
what is clearly visible, which reduces made-up details.

Memory: ~4.5 GB download. Runs in about 9 GB RAM on CPU (float32),
or ~5 GB VRAM on a GPU. Slower than BLIP on CPU (roughly 20-60 s per image).
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

QWEN_MODELS = {
    "Qwen2-VL-2B (most accurate, needs ~9 GB RAM)": "Qwen/Qwen2-VL-2B-Instruct",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PROMPT = (
    "Describe this image in one or two clear, accurate sentences. "
    "Mention the main subjects, what they are doing, and the setting. "
    "Only state what is clearly visible; do not guess or invent details."
)


def load_qwen(model_id):
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 28 * 28,  # caps image size so CPU stays usable
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    )
    model.eval().to(DEVICE)
    return processor, model


@torch.no_grad()
def qwen_caption(image, processor, model, prompt=None, max_new_tokens=80):
    """Return a list with a single caption (greedy decoding = deterministic)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt or DEFAULT_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = output[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return [caption]
