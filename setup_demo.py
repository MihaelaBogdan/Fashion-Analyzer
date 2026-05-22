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
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from app import YOLO_CONF

ROOT = Path(".")
DATASET_ROOT = ROOT / "fashion200k"
CROPS_DIR = ROOT / "fashion_crops"
EMB_DIR = ROOT / "fashion_emb"
INDEX_DIR = ROOT / "fashion_index"
CKPT_DIR = ROOT / "fashion_ckpt"
images_dir = DATASET_ROOT / "images"
YOLO_CONF = 0.25      # or whatever your preferred confidence is
MIN_CROP_SIZE = 48    # prevents tiny, pixelated 5x5 pixel crops

for d in [DATASET_ROOT, CROPS_DIR, EMB_DIR, INDEX_DIR, CKPT_DIR, images_dir]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MAX_IMAGES = 5000

print("1. Loading dataset...")
fashion_dataset = load_dataset(
    "Marqo/fashion200k",
    split="data",
    trust_remote_code=True
)

fashion_dataset = fashion_dataset.shuffle(seed=42) #shuffle to get a good variety of items in the limited set

print(f"Dataset loaded. Total samples: {len(fashion_dataset)}")
print(f"Columns: {fashion_dataset.column_names}")
print(fashion_dataset.column_names)
print(fashion_dataset[0])

raw_manifest = []
print(f"Saving {MAX_IMAGES} images to disk...")
for idx in tqdm(range(min(MAX_IMAGES, len(fashion_dataset)))):
    sample = fashion_dataset[idx]

    # Get image — HuggingFace returns PIL Image directly
    img = sample["image"]
    if img is None:
        continue

    # Get label — column name varies by dataset version
    #label    = sample.get("label", sample.get("category", sample.get("name", f"item_{idx}")))
    #category = sample.get("category", str(label).split()[0] if label else "unknown")
    label    = sample["item_ID"]
    category = sample["category1"]

    # Save image
    img_path = images_dir / f"{idx:06d}.jpg"
    img.convert("RGB").save(img_path, quality=90)

    raw_manifest.append({
        "image_path": str(img_path),
        "label":       sample["item_ID"],
        "category":    sample["category1"],
        "subcategory": sample["category2"],
        "fine_label":  sample["category3"]
    })
print(f"Saved {len(raw_manifest)} images.")

print("3. Running YOLO detector...")
detector = YOLO("yolov8n-fashionpedia-1.torchscript", verbose=False, task="detect")
print("Successfully loaded Fashionpedia YOLO model!")
print(f"Total categories: {len(detector.names)}")

crop_manifest = []

print(f"Running Fashionpedia detection on {len(raw_manifest)} images...")
print("🚀 Processing in safe chunks to prevent CUDA Out Of Memory...")

all_paths = [item["image_path"] for item in raw_manifest]

# Setăm dimensiunea unei porții textuale trimise către YOLO
# Chiar dacă batch-ul intern e 32, faptul că spargem lista mare previne supra-alocarea de 30GB
CHUNK_SIZE = 256  
BATCH_SIZE_YOLO = 32

# Inițializăm tqdm pe lungimea totală a datasetului
with tqdm(total=len(all_paths)) as pbar:
    for chunk_idx in range(0, len(all_paths), CHUNK_SIZE):
        chunk_paths = all_paths[chunk_idx:chunk_idx + CHUNK_SIZE]
        
        # Rulăm YOLO doar pentru porțiunea curentă
        results = detector.predict(
            source=chunk_paths,
            conf=YOLO_CONF,
            save=False,
            verbose=False,
            batch=BATCH_SIZE_YOLO,
            stream=True
        )

        for sub_idx, result in enumerate(results):
            # Calculăm indexul global corect din raw_manifest
            global_idx = chunk_idx + sub_idx
            item = raw_manifest[global_idx]
            img_path = item["image_path"]
            
            try:
                img = None 
                w, h = result.orig_shape[1], result.orig_shape[0]

                if len(result.boxes) == 0:
                    img = Image.open(img_path).convert("RGB")
                    crop_path = CROPS_DIR / f"full_{Path(img_path).stem}.png"
                    img.save(crop_path)
                    crop_manifest.append({
                        "crop_path": str(crop_path),
                        "garment_type": item["category"],
                        "product_label": item["label"],
                        "category": item["category"],
                        "source_image": img_path
                    })
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                garment_names = [result.names[c] for c in class_ids]

                for box_idx, (box, garment_name) in enumerate(zip(boxes, garment_names)):
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))

                    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                        continue

                    if img is None:
                        img = Image.open(img_path).convert("RGB")

                    crop = img.crop((x1, y1, x2, y2))
                    stem = Path(img_path).stem
                    crop_filename = f"{garment_name}_{stem}_{box_idx:02d}.png"
                    crop_path = CROPS_DIR / crop_filename
                    crop.save(crop_path)

                    crop_manifest.append({
                        "crop_path": str(crop_path),
                        "garment_type": garment_name,
                        "product_label": item["label"],
                        "category": item["category"],
                        "source_image": img_path
                    })

            except Exception as e:
                print(f"Skipped {img_path}: {e}")
                continue
        
        # Eliberăm agresiv memoria cache din GPU după fiecare chunk de 256 de imagini
        pbar.update(len(chunk_paths))
        torch.cuda.empty_cache()

# Save manifest
with open(EMB_DIR / "crop_manifest.json", "w") as f:
    json.dump(crop_manifest, f, indent=2)

print(f"\nExtracted {len(crop_manifest)} garment crops.")

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

# ──────────────────────────────────────────────────────────────────────
# 🚀 CLASĂ DATASET ȘI LOADER PENTRU EXTRAGERE ULTRA-RAPIDĂ ÎN LOTURI (BATCHES)
# ──────────────────────────────────────────────────────────────────────
class FastInferenceDataset(Dataset):
    def __init__(self, manifest, transform):
        self.manifest = manifest
        self.transform = transform
    def __len__(self):
        return len(self.manifest)
    def __getitem__(self, idx):
        item = self.manifest[idx]
        try:
            img = Image.open(item["crop_path"]).convert("RGB")
            tensor = self.transform(img)
            return tensor, idx
        except Exception:
            # În caz de eroare la o imagine, returnăm un tensor gol ca să nu crape batch-ul
            return torch.zeros(3, 224, 224), -1

# Trimitem câte x imagini deodată
inference_dataset = FastInferenceDataset(crop_manifest, inference_transform)
fast_loader = DataLoader(
    inference_dataset, 
    batch_size=128, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=True
)

all_vectors = [None] * len(crop_manifest)
all_metadata = [None] * len(crop_manifest)

print(f"Extracting features for {len(crop_manifest)} crops using Batch Size = 128...")
with torch.no_grad():
    for tensors, indices in tqdm(fast_loader):
        tensors = tensors.to(DEVICE)
        
        # Procesăm tot lotul de 128 dintr-o singură lovitură de GPU
        embeddings = model(tensors).cpu().numpy()
        
        for emb, idx in zip(embeddings, indices):
            idx_int = int(idx.item())
            if idx_int != -1:
                all_vectors[idx_int] = emb
                all_metadata[idx_int] = crop_manifest[idx_int]

# Curățăm eventualele imagini sărite/corupte
valid_vectors = [v for v in all_vectors if v is not None]
valid_metadata = [m for m in all_metadata if m is not None]

vectors_matrix = np.vstack(valid_vectors).astype("float32")
np.save(EMB_DIR / "vectors.npy", vectors_matrix)

with open(EMB_DIR / "metadata.json", "w") as f:
    json.dump(valid_metadata, f, indent=2)

print("5. Building FAISS index...")
dimension = vectors_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors_matrix)
faiss.write_index(index, str(INDEX_DIR / "fashion.index"))

print("Setup complete! App is ready.")
