import torch
import numpy as np
import open_clip
from collections import Counter
import requests
from typing import List, Tuple

class SpLiCEConceptDictionary:
    def __init__(self, model, tokenizer, num_words=10000, num_bigrams=5000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.num_words = num_words
        self.num_bigrams = num_bigrams
        self.concept_texts = []
        self.C = None  # Raw concept embeddings (d x c)
        
    def load_laion_vocab(self):
        """Load top LAION-400m words/bigrams (paper uses filtered LAION captions)."""
        # For demo: Use dummy frequent LAION concepts (replace with real vocab download)
        # Real vocab available at: https://github.com/AI4LIFE-GROUP/SpLiCE
        demo_words = [
            'dog', 'cat', 'car', 'person', 'tree', 'water', 'sky', 'grass', 
            'house', 'food', 'face', 'man', 'woman', 'child', 'bird', 'fish'
        ] * 700  # Simulate top 10k
        
        demo_bigrams = [
            'coffee cup', 'beach sand', 'city street', 'mountain view', 
            'sports car', 'ocean waves', 'forest trees', 'sunset sky'
        ] * 625  # Simulate top 5k
        
        # Real implementation: Download from SpLiCE repo or extract from LAION captions
        self.concept_texts = demo_words[:self.num_words] + demo_bigrams[:self.num_bigrams]
        print(f"Loaded {len(self.concept_texts)} concepts ({self.num_words} words + {self.num_bigrams} bigrams)")
        
    def build_dictionary(self):
        """Encode concepts into embedding matrix C."""
        self.load_laion_vocab()
        
        # Batch encode all concepts
        texts = self.concept_texts
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            C_raw = self.model.encode_text(text_tokens)  # (c, d)
            C_raw = C_raw / C_raw.norm(dim=-1, keepdim=True)  # L2 normalize
        
        self.C = C_raw.cpu().numpy().T  # (d, c) shape for decomposition
        print(f"Concept dictionary shape: {self.C.shape}")  # (512, 15000)
        
        return self.C, self.concept_texts

# Usage with Step 1 model
model = SpLiCEModel()
concepts = SpLiCEConceptDictionary(model.model, model.tokenizer)
C, concept_names = concepts.build_dictionary()
