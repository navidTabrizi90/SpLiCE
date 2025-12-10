from datasets import load_dataset
from src.clip_encoder import CLIPEncoder
from src.concept_builder import ConceptBuilder
import inspect
import src.concept_builder

import sys
import os
sys.path.insert(0, os.path.abspath("./src"))

from concept_builder import ConceptBuilder



########################################
# STEP 1: Stream captions from OPEN LAION dataset
########################################

print("Streaming captions from Conceptual Captions (unlabeled split)...")

# This dataset is fully public and does NOT require permissions
ds = load_dataset("conceptual_captions", "unlabeled", split="train", streaming=True)



captions = []
for i, row in enumerate(ds):
    text = row.get("caption")
    if text:
        captions.append(text)
    if i == 500_000:
        break



print("Collected captions:", len(captions))


########################################
# STEP 2: Initialize CLIP Encoder
########################################

print("Loading CLIP encoder...")
encoder = CLIPEncoder(model_name="ViT-B-32")


########################################
# STEP 3: Build SpLiCE Concept Vocabulary
########################################

print("Building concept vocabulary...")

builder = ConceptBuilder(
    encoder=encoder,
    min_freq=50,       # lower min_freq because dataset subset is smaller
    max_concepts=15000  # target dictionary size (SpLiCE paper uses ~15k)
)

concept_list, C = builder.build(
    captions=captions,
    out_dir="data/"
)

print("\n============================")
print("DONE!")
print(f"Saved {len(concept_list)} concepts to data/concepts.json")
print(f"Saved concept embeddings to data/C.pt")
print("============================\n")
