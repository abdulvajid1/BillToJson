import torch
from PIL import Image
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HuggingFaceVLMModel:
    def __init__(self, model_name:str="HuggingFaceTB/SmolVLM-256M-Instruct"):
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.model = AutoModelForMultimodalLM.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", device_map="auto")        
    
    def predict(self, inputs):
        outputs = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    def process_input(self,
                      image: Image, 
                      SYSTEM_PROMPT:str = "Extract the text content in the image in markdown format"):
        
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
    from pathlib import Path
    model = HuggingFaceVLMModel('HuggingFaceTB/SmolVLM-256M-Instruct')
    img = Path("src/billstojson/data/eval").glob("*.jpeg")
    for i in img:
        print(i.as_posix())
        break
    
    img = load_image(i.as_posix())
    inputs = model.process_input(img)
    print(model.predict(inputs))
    print("Successfull")



