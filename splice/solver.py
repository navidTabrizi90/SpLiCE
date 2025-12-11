from sklearn.linear_model import Lasso
import numpy as np

class SpLiCESolver:
    def __init__(self, C_centered, concept_texts, lambda_l1=0.1):
        self.C_centered = C_centered  # (d, c) from Step 3
        self.concept_texts = concept_texts
        self.lambda_l1 = lambda_l1  # Controls sparsity (0.1-0.3 recommended)
        self.d, self.c = C_centered.shape
        
    def solve_sparse_decomposition(self, z_centered):
        """Solve min_w ||C_centered w - z_centered||_2^2 + 2λ||w||_1, w ≥ 0."""
        # Lasso expects features x samples, so transpose: C_centered.T (c x d), z_centered.T (d,)
        lasso = Lasso(
            alpha=self.lambda_l1,  # L1 penalty strength
            positive=True,         # Non-negative constraint
            max_iter=10000,        # Ensure convergence
            selection='random'     # Better for high-dimensional
        )
        
        # Fit: w = argmin ||C w - z||^2 + λ||w||_1
        w_sparse = lasso.fit(self.C_centered.T, z_centered).coef_
        
        # Sparsity metrics
        sparsity_l0 = np.sum(w_sparse > 1e-6)  # Non-zero count
        sparsity_l1 = np.sum(w_sparse)
        
        print(f"Sparsity: l0={sparsity_l0}, l1={sparsity_l1:.3f}")
        print(f"Top active concepts: {sparsity_l0}")
        
        return w_sparse
    
    def get_top_k_concepts(self, w_sparse, k=10):
        """Extract top-k active concepts with weights."""
        # Get indices of non-zero weights, sorted by magnitude
        active_indices = np.where(w_sparse > 1e-6)[0]
        top_indices = active_indices[np.argsort(w_sparse[active_indices])[-k:][::-1]]
        
        results = {
            self.concept_texts[i]: float(w_sparse[i]) 
            for i in top_indices 
            if w_sparse[i] > 1e-6
        }
        return results

# Integrate all previous steps
model = SpLiCEModel()
concepts = SpLiCEConceptDictionary(model.model, model.tokenizer)
C_raw, concept_names = concepts.build_dictionary()

alignment = SpLiCEModalityAlignment(model.model, C_raw, concept_names)
C_centered = alignment.align_concept_dictionary()

# Test full pipeline
image = Image.open("your_image.jpg")
z_img = model.extract_image_embedding(image)
z_centered = alignment.center_and_normalize(z_img)

# Solve sparse decomposition
solver = SpLiCESolver(C_centered, concept_names, lambda_l1=0.2)
w_sparse = solver.solve_sparse_decomposition(z_centered)

# Get interpretable results
top_concepts = solver.get_top_k_concepts(w_sparse, k=10)
print("Top 10 SpLiCE concepts:")
for concept, weight in top_concepts.items():
    print(f"  {concept}: {weight:.3f}")
