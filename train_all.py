"""
Fashion Analyzer — Training Script
===================================
Antrenează EfficientNet (Triplet Loss) + CLIP (Contrastive Loss)
pe toate datele disponibile local.

Rulare:
    source venv/bin/activate
    python3 train_all.py
"""

import json, random, time, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

ROOT       = Path(__file__).parent.resolve()
EMB_DIR    = ROOT / "fashion_emb"
INDEX_DIR  = ROOT / "fashion_index"
CKPT_DIR   = ROOT / "fashion_ckpt"
CKPT_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
EMBED_DIM  = 512

EFF_EPOCHS          = 40
EFF_BATCH           = 32
EFF_LR_BACKBONE     = 5e-5
EFF_LR_HEAD         = 5e-4
EFF_MARGIN          = 0.4
EFF_HARD_NEG_EPOCH  = 10
EFF_MIN_SAMPLES     = 3

CLIP_EPOCHS = 10
CLIP_BATCH  = 16
CLIP_LR     = 3e-5

print(f"Device: {DEVICE}")
print("=" * 60)

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class FashionEmbeddingModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, embed_dim),
        )

    def forward(self, x):
        f = self.backbone(x)
        e = self.projection(f)
        return F.normalize(e, dim=-1)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        d_pos = (anchor - pos).pow(2).sum(1)
        d_neg = (anchor - neg).pow(2).sum(1)
        losses = F.relu(d_pos - d_neg + self.margin)
        active = (losses > 0).float().mean().item()
        return losses.mean(), active


class TripletDataset(Dataset):
    def __init__(self, metadata, transform, hard_negative=False, min_samples=3):
        self.tf = transform
        self.hard_negative = hard_negative

        cat2items = defaultdict(list)
        for item in metadata:
            p = Path(item["crop_path"])
            if p.exists():
                cat2items[item["category"]].append(item)

        self.cat2items = {k: v for k, v in cat2items.items() if len(v) >= min_samples}
        self.all_cats  = list(self.cat2items.keys())
        self.items     = [item for items in self.cat2items.values() for item in items]
        print(f"  TripletDataset: {len(self.items)} imagini, {len(self.all_cats)} categorii")

    def set_hard_negative(self, v):
        self.hard_negative = v

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        anchor_meta = self.items[idx]
        cat = anchor_meta["category"]
        pos_meta = random.choice(self.cat2items[cat])

        if self.hard_negative:
            neg_cat = random.choice([c for c in self.all_cats if c != cat])
        else:
            neg_cat = random.choice([c for c in self.all_cats if c != cat])

        neg_meta = random.choice(self.cat2items[neg_cat])
        return (
            self._load(anchor_meta["crop_path"]),
            self._load(pos_meta["crop_path"]),
            self._load(neg_meta["crop_path"]),
            cat,
        )


class CLIPFashionDataset(Dataset):
    def __init__(self, metadata):
        self.items = [m for m in metadata if Path(m["crop_path"]).exists()]
        print(f"  CLIPDataset: {len(self.items)} imagini")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img  = Image.open(item["crop_path"]).convert("RGB")
        cat  = item.get("category", "")
        label = item.get("product_label", cat)
        text  = f"a photo of a {label.lower()}, fashion item, {cat.lower()}"
        return img, text


def train_efficientnet(metadata):
    print("\n" + "="*60)
    print("ETAPA 1: EfficientNet Triplet Training")
    print("="*60)

    dataset = TripletDataset(metadata, train_tf, hard_negative=False,
                              min_samples=EFF_MIN_SAMPLES)
    loader  = DataLoader(dataset, batch_size=EFF_BATCH, shuffle=True,
                         num_workers=0, pin_memory=False, drop_last=True)

    model    = FashionEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
    criterion = TripletLoss(margin=EFF_MARGIN)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.backbone.named_parameters()
                    if "blocks" in n or "conv_head" in n or "bn2" in n],
         "lr": EFF_LR_BACKBONE},
        {"params": model.projection.parameters(), "lr": EFF_LR_HEAD},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[EFF_LR_BACKBONE * 10, EFF_LR_HEAD * 10],
        steps_per_epoch=len(loader),
        epochs=EFF_EPOCHS,
        pct_start=0.1,
    )

    best_loss = float("inf")
    history   = []

    for epoch in range(EFF_EPOCHS):
        if epoch == EFF_HARD_NEG_EPOCH:
            dataset.set_hard_negative(True)
            print(f"  [Epoch {epoch+1:02d}] → Switching to HARD negatives")

        model.train()
        epoch_loss, epoch_active, n = 0.0, 0.0, 0

        for anc, pos, neg, _ in loader:
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            optimizer.zero_grad()
            ea, ep, en = model(anc), model(pos), model(neg)
            loss, af = criterion(ea, ep, en)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_active += af
            n += 1

        avg_loss   = epoch_loss / n
        avg_active = epoch_active / n
        history.append(avg_loss)
        mode = "Hard" if epoch >= EFF_HARD_NEG_EPOCH else "Rand"
        print(f"  Epoch {epoch+1:02d}/{EFF_EPOCHS} [{mode}] | "
              f"Loss: {avg_loss:.4f} | Active: {avg_active:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"    ✔ Salvat best_model.pth (loss={best_loss:.4f})")

    print(f"\nEfficientNet training complet. Best loss: {best_loss:.4f}")
    return model


def rebuild_efficientnet_index(model, metadata):
    print("\n[Index] Reconstruim indexul FAISS pentru EfficientNet...")
    import faiss

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model.eval()
    vectors = []
    valid_meta = []

    with torch.no_grad():
        for item in metadata:
            p = Path(item["crop_path"])
            if not p.exists():
                continue
            try:
                img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                emb = model(img).squeeze(0).cpu().numpy()
                vectors.append(emb)
                valid_meta.append(item)
            except Exception:
                continue

    vectors_np = np.stack(vectors).astype("float32")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors_np)
    faiss.write_index(index, str(INDEX_DIR / "fashion.index"))

    np.save(str(EMB_DIR / "vectors.npy"), vectors_np)

    with open(EMB_DIR / "metadata.json", "w") as f:
        json.dump(valid_meta, f)

    print(f"  Index FAISS: {len(vectors_np)} vectori salvați.")


def train_clip(metadata):
    print("\n" + "="*60)
    print("ETAPA 2: CLIP Contrastive Fine-Tuning")
    print("="*60)

    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("  transformers nu este instalat. Skip CLIP.")
        return

    model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained(model_id)

    for param in clip_model.parameters():
        param.requires_grad = False
    for param in clip_model.visual_projection.parameters():
        param.requires_grad = True
    for param in clip_model.text_projection.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"  Parametri antrenabili: {trainable:,}")

    dataset = CLIPFashionDataset(metadata)

    def collate_fn(batch):
        imgs, texts = zip(*batch)
        return clip_proc(
            text=list(texts),
            images=list(imgs),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    loader = DataLoader(dataset, batch_size=CLIP_BATCH, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, drop_last=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, clip_model.parameters()),
        lr=CLIP_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CLIP_EPOCHS, eta_min=1e-6
    )

    best_loss = float("inf")

    for epoch in range(CLIP_EPOCHS):
        clip_model.train()
        epoch_loss, n = 0.0, 0

        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            out = clip_model(**batch, return_loss=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += out.loss.item()
            n += 1

        avg = epoch_loss / n
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{CLIP_EPOCHS} | Loss: {avg:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save(clip_model.state_dict(), CKPT_DIR / "clip_finetuned.pth")
            print(f"    ✔ Salvat clip_finetuned.pth (loss={best_loss:.4f})")

    print(f"\nCLIP training complet. Best loss: {best_loss:.4f}")
    return clip_model, clip_proc


def rebuild_clip_index(clip_model, clip_proc, metadata):
    print("\n[Index] Reconstruim indexul CLIP FAISS...")
    import faiss
    from transformers import CLIPProcessor, CLIPModel

    clip_model.eval()
    vectors, valid_meta = [], []

    with torch.no_grad():
        for item in metadata:
            p = Path(item["crop_path"])
            if not p.exists():
                continue
            try:
                img = Image.open(p).convert("RGB")
                inp = clip_proc(images=img, return_tensors="pt").to(DEVICE)
                emb = clip_model.get_image_features(**inp)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                vectors.append(emb.squeeze(0).cpu().numpy().astype("float32"))
                valid_meta.append(item)
            except Exception:
                continue

    clip_dim = vectors[0].shape[0]
    vectors_np = np.stack(vectors).astype("float32")
    index = faiss.IndexFlatIP(clip_dim)
    index.add(vectors_np)
    faiss.write_index(index, str(INDEX_DIR / "fashion_clip.index"))
    print(f"  CLIP Index FAISS: {len(vectors_np)} vectori salvați (dim={clip_dim}).")


if __name__ == "__main__":
    print(f"\nÎncărcăm metadatele...")
    with open(EMB_DIR / "metadata.json") as f:
        metadata = json.load(f)

    metadata = [m for m in metadata if Path(m["crop_path"]).exists()]
    print(f"  {len(metadata)} imagini valide găsite.")

    t0 = time.time()

    eff_model = train_efficientnet(metadata)
    rebuild_efficientnet_index(eff_model, metadata)

    result = train_clip(metadata)
    if result:
        clip_model, clip_proc = result
        rebuild_clip_index(clip_model, clip_proc, metadata)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLET in {elapsed/60:.1f} minute.")
    print(f"Checkpoints salvate in: {CKPT_DIR}")
    print(f"Indexuri FAISS salvate in: {INDEX_DIR}")
    print(f"{'='*60}")
