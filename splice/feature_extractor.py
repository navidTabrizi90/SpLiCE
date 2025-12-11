# step1_load_clip.py

import torch
import open_clip
from PIL import Image

class CLIPWrapper:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model + preprocess + tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    # --------------------------------------------------------
    # Encode image -> returns normalized embedding (1D vector)
    # --------------------------------------------------------
    def encode_image(self, image: Image.Image):
        with torch.no_grad():
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(img_tensor)  # shape [1, d]
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        return emb.squeeze(0).cpu().numpy()  # shape [d]

    # --------------------------------------------------------
    # Encode list of text strings -> normalized embeddings
    # --------------------------------------------------------
    def encode_text(self, texts):
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens)  # shape [B, d]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()  # shape [B, d]
