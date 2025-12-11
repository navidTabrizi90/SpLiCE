from splice_explainer import SpliceExplainer
from PIL import Image

def load_concepts(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]

concept_list = load_concepts("concepts.txt")

explainer = SpliceExplainer(
    concept_list=concept_list,
    mu_img_path="mu_img_mscoco.npy",
    lambda_=0.05,
)

out = explainer.decompose(Image.open("some_image.jpg"), top_k=20)
print(out["concepts"])
