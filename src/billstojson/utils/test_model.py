import glob
from pathlib import Path


def test_model(model:str = '', path_to_eval:str = "src/billstojson/data/eval"):
    eval_images = [img_path.name for img_path in Path(path_to_eval).glob("*.jpeg")]
    model.predict(eval_images)

if __name__ == "__main__":
    test_model()