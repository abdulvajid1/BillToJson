import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
image = Image.new("RGB", (224, 224))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract the text content in the image in markdown format"}
        ]
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print("Prompt:", prompt)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
print("Inputs keys:", inputs.keys())
