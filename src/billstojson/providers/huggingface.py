import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForMultimodalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HuggingFaceVLMModel:
    def __init__(self, hf_model_name:str="HuggingFaceTB/SmolVLM-256M-Instruct"):
        self.processor = AutoProcessor.from_pretrained(hf_model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(hf_model_name, device_map="auto")        
    
    def predict(self, inputs):
        outputs = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    def process_input(self,
                      image, 
                      SYSTEM_PROMPT:str = f"Extract the Title and Total amount of the bill/price from the picture in json format \n\n Eg: {{title: str, total_amount: int}}"):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
        ]
        
        
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)
        
        return inputs
        


if __name__ == "__main__":
    import os
    from pathlib import Path
    import random
    
    model = HuggingFaceVLMModel('HuggingFaceTB/SmolVLM-256M-Instruct')
    img_path = os.path.join('src', 'billstojson', 'data', 'eval')
    images = [os.path.join(img_path, img_file) for img_file in os.listdir(img_path)]
    
    random_pick_image = random.choice(images)
    print(random_pick_image)
    
    img = load_image(random_pick_image)
    inputs = model.process_input(img)
    print(model.predict(inputs))
    print("Successfull")



