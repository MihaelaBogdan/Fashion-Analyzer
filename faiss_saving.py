import json
import torch
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import timm
import torch.nn as nn
import torch.nn.functional as F
from app import FashionEmbeddingModel

ROOT = Path(".")
EMB_DIR = ROOT / "fashion_emb"
INDEX_DIR = ROOT / "fashion_index"
CKPT_DIR = ROOT / "fashion_ckpt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
IMG_SIZE = 224

# Load model
model = FashionEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load manifest
with open(EMB_DIR / "crop_manifest.json") as f:
    crop_manifest = json.load(f)

# Embed all crops
all_vectors, all_metadata = [], []
with torch.no_grad():
    for item in tqdm(crop_manifest):
        try:
            img = Image.open(item["crop_path"]).convert("RGB")
            tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
            emb = model(tensor).squeeze(0).cpu().numpy()
            all_vectors.append(emb.astype("float32"))
            all_metadata.append(item)
        except Exception as e:
            continue

vectors_matrix = np.vstack(all_vectors).astype("float32")
np.save(EMB_DIR / "vectors.npy", vectors_matrix)
with open(EMB_DIR / "metadata.json", "w") as f:
    json.dump(all_metadata, f, indent=2)

# Build FAISS index
dimension = vectors_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors_matrix)
faiss.write_index(index, str(INDEX_DIR / "fashion.index"))
print(f"Done. Indexed {index.ntotal} vectors.")