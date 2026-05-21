import os
import json
import torch
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from ultralytics import YOLO
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(".")
DATASET_ROOT = ROOT / "fashion200k"
CROPS_DIR = ROOT / "fashion_crops"
EMB_DIR = ROOT / "fashion_emb"
INDEX_DIR = ROOT / "fashion_index"
CKPT_DIR = ROOT / "fashion_ckpt"
images_dir = DATASET_ROOT / "images"

for d in [DATASET_ROOT, CROPS_DIR, EMB_DIR, INDEX_DIR, CKPT_DIR, images_dir]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MAX_IMAGES = 2000

print("1. Loading dataset...")
fashion_dataset = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=False)

raw_manifest = []
print("2. Saving images...")
for idx in range(min(MAX_IMAGES, len(fashion_dataset))):
    sample = fashion_dataset[idx]
    img = sample["image"]
    if img is None: continue
    label = sample.get("productDisplayName", f"item_{idx}")
    category = sample.get("articleType", "unknown")
    img_path = images_dir / f"{idx:06d}.jpg"
    img.convert("RGB").save(img_path, quality=90)
    raw_manifest.append({"image_path": str(img_path), "label": str(label), "category": str(category)})

print("3. Running YOLO detector...")
detector = YOLO("yolov8n.pt")
crop_manifest = []

for item in tqdm(raw_manifest):
    img_path = item["image_path"]
    try:
        img = Image.open(img_path).convert("RGB")
        crop_path = CROPS_DIR / f"full_{Path(img_path).stem}.png"
        img.save(crop_path)
        crop_manifest.append({
            "crop_path": str(crop_path), "garment_type": item["category"],
            "product_label": item["label"], "category": item["category"], "source_image": img_path
        })
    except Exception as e:
        print(f"Skipped {img_path}: {e}")

with open(EMB_DIR / "crop_manifest.json", "w") as f:
    json.dump(crop_manifest, f, indent=2)

print(f"Generated {len(crop_manifest)} crops.")

print("4. Embedding all crops using pre-trained backbone...")
class FashionEmbeddingModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0, global_pool="avg")
        for param in self.backbone.parameters(): param.requires_grad = False
        backbone_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

model = FashionEmbeddingModel().to(DEVICE)
model.eval()

torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

all_vectors, all_metadata = [], []
with torch.no_grad():
    for item in tqdm(crop_manifest):
        try:
            img = Image.open(item["crop_path"]).convert("RGB")
            tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
            emb = model(tensor).squeeze(0).cpu().numpy()
            all_vectors.append(emb)
            all_metadata.append(item)
        except Exception:
            pass

vectors_matrix = np.vstack(all_vectors).astype("float32")
np.save(EMB_DIR / "vectors.npy", vectors_matrix)
with open(EMB_DIR / "metadata.json", "w") as f:
    json.dump(all_metadata, f, indent=2)

print("5. Building FAISS index...")
dimension = vectors_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors_matrix)
faiss.write_index(index, str(INDEX_DIR / "fashion.index"))

print("Setup complete! App is ready.")
