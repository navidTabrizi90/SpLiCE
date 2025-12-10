import re
import json
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

print(">>> ConceptBuilder loaded from:", __file__)


class ConceptBuilder:
    def __init__(self, encoder, min_freq=50_000, max_concepts=15_000):
        """
        encoder: CLIPEncoder
        min_freq: minimum count to keep a unigram/bigram (LAION scale)
        max_concepts: target size (paper uses ~15k)
        """
        self.encoder = encoder
        self.min_freq = min_freq
        self.max_concepts = max_concepts

    def clean_caption(self, text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9 ]+", "", text)
        return text
    
    def extract_ngrams(self, captions):
        unigram_counter = Counter()
        bigram_counter = Counter()

        for cap in tqdm(captions, desc="Extracting n-grams"):
            cap = self.clean_caption(cap)
            tokens = cap.split()

            # unigrams
            for t in tokens:
                unigram_counter[t] += 1

            # bigrams
            for i in range(len(tokens)-1):
                bg = tokens[i] + " " + tokens[i+1]
                bigram_counter[bg] += 1

        return unigram_counter, bigram_counter

    def filter_ngrams(self, unigrams, bigrams):
        # threshold filtering
        uni = [w for w, c in unigrams.items() if c >= self.min_freq]
        bi = [w for w, c in bigrams.items() if c >= self.min_freq]

        # remove duplicates
        uni = list(set(uni))
        bi = list(set(bi))

        # remove bigrams that are too similar to a unigram
        filtered_bigrams = []
        for bg in bi:
            t1, t2 = bg.split()
            if t1 not in uni or t2 not in uni:
                filtered_bigrams.append(bg)

        # combine
        all_concepts = uni + filtered_bigrams

        # limit size
        all_concepts = all_concepts[:self.max_concepts]

        return all_concepts

    def encode_concepts(self, concept_list):
        embeddings = []
        for c in tqdm(concept_list, desc="Encoding concepts"):
            emb = self.encoder.encode_text([c])
            embeddings.append(emb.cpu().numpy()[0])
        return np.vstack(embeddings)

    def save(self, concept_list, C, out_dir="data/"):
        # save list
        with open(f"{out_dir}/concepts.json", "w") as f:
            json.dump(concept_list, f, indent=2)

        # save embeddings
        torch.save(torch.tensor(C), f"{out_dir}/C.pt")

    def build(self, captions, out_dir="data/"):
        # 1. n-gram extraction
        uni, bi = self.extract_ngrams(captions)

        # 2. filtering
        concept_list = self.filter_ngrams(uni, bi)

        # 3. encoding with CLIP
        C = self.encode_concepts(concept_list)

        # 4. save
        self.save(concept_list, C, out_dir)

        return concept_list, C
