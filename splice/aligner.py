import numpy as np

class ModalityAligner:
    """
    Handles Step 3 of SpLiCE:
    - Loading or computing μ_img (image mean)
    - Computing μ_con (concept mean)
    - Centering & normalizing the concept dictionary
    - Providing functions to center image embeddings
    """

    def __init__(self, C_txt, mu_img_path=None):
        """
        C_txt : numpy array of shape [num_concepts, d]
        mu_img_path : path to precomputed μ_img (recommended)
        """

        self.C_txt = C_txt  # raw text embeddings, shape [num_concepts, d]
        self.d = C_txt.shape[1]

        # -----------------------------
        # STEP 3.1 — LOAD OR CREATE μ_img
        # -----------------------------
        if mu_img_path is not None:
            self.mu_img = np.load(mu_img_path)                  # [d]
        else:
            print("[WARNING] No μ_img file provided — using zeros. "
                  "You MUST replace this with the real COCO image mean.")
            self.mu_img = np.zeros(self.d, dtype=np.float32)

        # -----------------------------
        # STEP 3.2 — COMPUTE μ_con
        # -----------------------------
        self.mu_con = C_txt.mean(axis=0)                         # [d]

        # -----------------------------
        # STEP 3.3 — CENTER & NORMALIZE CONCEPT DICTIONARY
        # -----------------------------
        C_centered = C_txt - self.mu_con  # subtract concept mean

        # normalize each concept vector
        C_centered = C_centered / np.clip(
            np.linalg.norm(C_centered, axis=1, keepdims=True), 1e-8, np.inf
        )

        # final dictionary shape: [d, num_concepts]
        self.C_centered = C_centered.T

        print("[ModalityAligner] Step 3 complete.")
        print(" - μ_img.shape =", self.mu_img.shape)
        print(" - μ_con.shape =", self.mu_con.shape)
        print(" - C_centered.shape =", self.C_centered.shape)


    # -----------------------------
    # STEP 3.4 — CENTER IMAGE EMBEDDING
    # -----------------------------
    def center_image_embedding(self, z_img):
        """
        z_img: numpy vector shape [d]
        Returns: centered & normalized image embedding
        """
        z_centered = z_img - self.mu_img
        z_centered = z_centered / np.linalg.norm(z_centered)
        return z_centered
