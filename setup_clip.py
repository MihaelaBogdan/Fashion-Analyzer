import os
import json
import torch
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

ROOT = Path(".")
EMB_DIR = ROOT / "fashion_emb"
INDEX_DIR = ROOT / "fashion_index"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Loading CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(DEVICE)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

with open(EMB_DIR / "crop_manifest.json") as f:
    crop_manifest = json.load(f)

all_vectors, all_metadata = [], []
print("Embedding crops with CLIP (512D)...")
with torch.no_grad():
    for item in tqdm(crop_manifest):
        try:
            img = Image.open(item["crop_path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            emb = model.get_image_features(**inputs).pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_vectors.append(emb.squeeze(0).cpu().numpy())
            all_metadata.append(item)
        except Exception:
            pass

vectors_matrix = np.vstack(all_vectors).astype("float32")
dimension = vectors_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors_matrix)
faiss.write_index(index, str(INDEX_DIR / "fashion_clip.index"))

print("CLIP FAISS Index Built Successfully!")
