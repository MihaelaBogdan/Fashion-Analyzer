import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import faiss
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Fashion Embeddings GUI", layout="wide")

ROOT = Path(__file__).parent.resolve()
CROPS_DIR = ROOT / "fashion_crops"
EMB_DIR = ROOT / "fashion_emb"
INDEX_DIR = ROOT / "fashion_index"
CKPT_DIR = ROOT / "fashion_ckpt"

for d in [CROPS_DIR, EMB_DIR, INDEX_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

YOLO_MODEL_PATH = "yolov8n_fashion2.pt"
YOLO_CONF = 0.35
MIN_CROP_SIZE = 48
EMBED_DIM = 256
IMG_SIZE = 224

class FashionEmbeddingModel(nn.Module):
    """
    EfficientNetV2-S backbone + projection head for fashion metric learning.

    EfficientNetV2-S has 7 fused-MBConv/MBConv stages.
    We freeze stages 0-4 (basic vision features) and unfreeze stages 5-6
    (high-level semantic features most relevant to fashion similarity).
    """
    def __init__(self, embed_dim=128):
        super().__init__()

        # Load EfficientNetV2-S pretrained on ImageNet-21k then fine-tuned on ImageNet-1k
        # in21k pretraining gives better features than 1k-only for fine-grained tasks
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=True,
            num_classes=0,        # remove classification head, output raw features
            global_pool="avg"     # global average pool after last conv block → 1280-dim
        )

        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (blocks[5] and blocks[6])
        # These capture high-level semantic and texture features
        for block in list(self.backbone.blocks)[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze the final conv + bn layer
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        # Get backbone output dimension (1280 for EfficientNetV2-S)
        backbone_dim = self.backbone.num_features

        # Projection head — trained entirely from scratch
        # Compresses 1280 → 128 while learning fashion-specific similarity
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),          # dropout helps prevent overfitting on fashion200k
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        features   = self.backbone(x)               # (batch, 1280)
        embeddings = self.projection(features)       # (batch, 128)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # unit sphere
        return embeddings




class TripletLoss(nn.Module):
    """
    L(a, p, n) = max(0, d(a,p) - d(a,n) + margin)

    d() = Euclidean distance on L2-normalized vectors
    (equivalent to cosine distance since vectors are unit-normalized)

    The loss is zero when the negative is already further from the anchor
    than the positive by at least the margin.
    Only violated triplets (loss > 0) produce gradients and cause learning.
    """
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos   = torch.norm(anchor - positive, p=2, dim=1)  # (batch,)
        dist_neg   = torch.norm(anchor - negative, p=2, dim=1)  # (batch,)
        losses     = F.relu(dist_pos - dist_neg + self.margin)  # (batch,)
        active_frac = (losses > 0).float().mean().item()
        return losses.mean(), active_frac


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FashionTripletDataset(Dataset):
    def __init__(self, manifest, transform=None, hard_negative=False):
        self.manifest = manifest
        self.transform = transform
        self.hard_negative = hard_negative

        self.label_to_indices = defaultdict(list)
        for idx, item in enumerate(manifest):
            self.label_to_indices[item["category"]].append(idx)

        self.category_to_labels = defaultdict(set)
        for item in manifest:
            self.category_to_labels[item["category"]].add(item["category"])

        self.valid_labels = [label for label, indices in self.label_to_indices.items() if len(indices) >= 2]
        self.all_labels = list(self.label_to_indices.keys())

    def set_hard_negative(self, val: bool):
        self.hard_negative = val

    def _load_image(self, idx):
        path = self.manifest[idx]["crop_path"]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        anchor_item = self.manifest[idx]
        anchor_label = anchor_item["category"]

        if anchor_label not in self.label_to_indices or len(self.label_to_indices[anchor_label]) < 2:
            anchor_label = random.choice(self.valid_labels)
            idx = random.choice(self.label_to_indices[anchor_label])
            anchor_item = self.manifest[idx]

        pos_candidates = [i for i in self.label_to_indices[anchor_label] if i != idx]
        pos_idx = random.choice(pos_candidates)

        if self.hard_negative:
            anchor_category = anchor_item["category"]
            same_cat_labels = [l for l in self.category_to_labels[anchor_category] if l != anchor_label]
            if same_cat_labels:
                neg_label = random.choice(same_cat_labels)
            else:
                neg_label = random.choice([l for l in self.all_labels if l != anchor_label])
        else:
            neg_label = random.choice([l for l in self.all_labels if l != anchor_label])

        neg_idx = random.choice(self.label_to_indices[neg_label])

        return (self._load_image(idx), self._load_image(pos_idx), self._load_image(neg_idx), anchor_label)

st.sidebar.title("Device Status")
st.sidebar.info(f"**Current Device:** `{DEVICE}`")

tab1, tab3, tab4 = st.tabs(["Căutare Semantică", "AI Outfit Analyzer", "Latent Space Analytics"])

with tab1:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #e2e8f0;
        }
        /* Custom Cards (Glassmorphism) */
        .css-1d391kg, .css-1lcbmhc, .dark-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        h1, h2, h3, h4, h5 {
            color: #f8fafc !important;
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }
        /* Vibrant Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%);
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 0 20px rgba(236, 72, 153, 0.5);
            color: #fff;
        }
        /* Neon Metric Card */
        .metric-card {
            background: linear-gradient(135deg, rgba(236, 72, 153, 0.8) 0%, rgba(139, 92, 246, 0.8) 100%);
            color: white;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .metric-value {
            font-size: 26px;
            font-weight: 900;
            margin: 10px 0;
            letter-spacing: 1px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metric-label {
            font-size: 13px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        /* Hover effects for similar items */
        .sim-card {
            background: rgba(15, 23, 42, 0.6);
            padding: 12px;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            cursor: pointer;
        }
        .sim-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 25px rgba(139, 92, 246, 0.3);
            border-color: rgba(236, 72, 153, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'><span style='background: -webkit-linear-gradient(#ec4899, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Semantic Fashion Search & Embedding Visualizer</span></h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
    embed_engine = st.selectbox("Select Embedding Engine:", ["CLIP (Vision + Text Cross-Modal)", "EfficientNetV2 (Visual Only)"])
    
    if embed_engine == "CLIP (Vision + Text Cross-Modal)":
        index_path = INDEX_DIR / "fashion_clip.index"
        st.info("CLIP Model: State-of-the-art semantic search. Connects images and text in the same mathematical space. No training required!")
    else:
        index_path = INDEX_DIR / "fashion.index"
        st.info("EfficientNetV2: Baseline fast visual embeddings. Works best after custom fine-tuning!")
        
    search_type = st.radio("Tipul de Căutare", ["Poză cu o haină (Image-to-Image)", "Descriere Textuală (Text-to-Image)"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    uploaded_file = None
    text_query = ""
    
    if search_type == "Poză cu o haină (Image-to-Image)":
        uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])
    else:
        text_query = st.text_input("Caută prin text (ex: Rochie roșie elegantă):")
        
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        
    pth_files = list(CKPT_DIR.glob("*.pth"))
    pth_files.sort(key=lambda p: (p.name != "best_model.pth", p.name))
    selected_model = pth_files[0].name if pth_files else None
        
    if (img is not None or text_query) and (selected_model is not None or "CLIP" in embed_engine):
        
        with st.spinner("Processing & Encoding..."):
            
            if "CLIP" in embed_engine or search_type == "Descriere Textuală (Text-to-Image)":
                model_id = "openai/clip-vit-base-patch32"
                clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE)
                clip_processor = CLIPProcessor.from_pretrained(model_id)
                clip_model.eval()
            else:
                model = FashionEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
                model.load_state_dict(torch.load(CKPT_DIR / selected_model, map_location=DEVICE))
                model.eval()
            
            meta_path = EMB_DIR / "metadata.json"
            
            faiss_index = None
            metadata = []
            
            if index_path.exists() and meta_path.exists():
                cpu_index = faiss.read_index(str(index_path))
                faiss_index = cpu_index
                with open(meta_path) as f:
                    metadata = json.load(f)
            else:
                st.warning(f"FAISS index {index_path.name} or metadata not found. Can't retrieve similar images.")
            
            with torch.no_grad():
                if search_type == "Descriere Textuală (Text-to-Image)":
                    inputs = clip_processor(text=[text_query], return_tensors="pt").to(DEVICE)
                    emb = clip_model.get_text_features(**inputs).pooler_output
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    emb = emb.squeeze(0).cpu().numpy().astype("float32")
                else:
                    if "CLIP" in embed_engine:
                        inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
                        emb = clip_model.get_image_features(**inputs).pooler_output
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb = emb.squeeze(0).cpu().numpy().astype("float32")
                    else:
                        def embed_pil(pil_img):
                            tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                emb = model(tensor).squeeze(0).cpu().numpy()
                            return emb.astype("float32")

                        def retrieve_filtered(query_vec, garment_type, top_k=5):
                            max_db = len(metadata)
                            current_k = top_k + 1
                            results = []
                            seen = set()
                            while len(results) < top_k and current_k <= max_db:
                                sc, idx_arr = faiss_index.search(query_vec.reshape(1, -1), current_k)
                                new_found = False
                                for score, idx in zip(sc[0], idx_arr[0]):
                                    if idx == -1 or idx in seen:
                                        continue
                                    seen.add(idx)
                                    new_found = True
                                    meta = metadata[idx]
                                    meta_type = meta.get("garment_type", "").lower().strip()
                                    query_type = garment_type.lower().strip()
                                    if query_type not in meta_type and meta_type not in query_type:
                                        continue
                                    results.append((meta, float(score)))
                                    if len(results) == top_k:
                                        break
                                if not new_found:
                                    break
                                current_k = min(current_k * 2, max_db + 1)
                            if len(results) == 0:
                                sc, idx_arr = faiss_index.search(query_vec.reshape(1, -1), top_k + 1)
                                for score, idx in zip(sc[0], idx_arr[0]):
                                    if idx == -1:
                                        continue
                                    results.append((metadata[idx], float(score)))
                                    if len(results) == top_k:
                                        break
                            return results

                        all_garment_results = []
                        w, h = img.size

                        try:
                            detector = YOLO(YOLO_MODEL_PATH, task="detect")
                            detect_results = detector.predict(
                                source=img, conf=YOLO_CONF, save=False, verbose=False
                            )[0]
                            boxes     = detect_results.boxes.xyxy.cpu().numpy()
                            class_ids = detect_results.boxes.cls.cpu().numpy().astype(int)
                            garments  = [detect_results.names[c] for c in class_ids]
                        except Exception as e:
                            st.warning(f"Detector failed ({e}), embedding whole image.")
                            boxes, garments = [], []
                        
                        if len(boxes) > 0:
                            areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
                            sorted_order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
                            boxes = boxes[sorted_order]
                            garments = [garments[i] for i in sorted_order]
                        
                        if len(boxes) == 0:
                            emb = embed_pil(img)
                        else:
                            first_emb_set = False
                            for box, garment_name in zip(boxes, garments):
                                x1, y1, x2, y2 = box
                                x1, y1 = max(0, int(x1)), max(0, int(y1))
                                x2, y2 = min(w, int(x2)), min(h, int(y2))
                                if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                                    continue
                                crop = img.crop((x1, y1, x2, y2))
                                crop_emb = embed_pil(crop)
                                if not first_emb_set:
                                    emb = crop_emb
                                    first_emb_set = True
                                filtered = retrieve_filtered(crop_emb, garment_name, top_k=5)
                                all_garment_results.append((garment_name, crop_emb, filtered))
                            if not first_emb_set:
                                emb = embed_pil(img)

            if faiss_index is not None:
                if 'all_garment_results' in locals() and all_garment_results:
                    # Use filtered results from first detected garment
                    first_results = all_garment_results[0][2]
                    dummy_scores = np.array([[r[1] for r in first_results]])
                    dummy_indices = np.array([[metadata.index(r[0]) for r in first_results]])
                    scores, indices = dummy_scores, dummy_indices
                    predicted_category = first_results[0][0].get('garment_type', 'Unknown') if first_results else 'Unknown'
                else:
                    scores, indices = faiss_index.search(emb.reshape(1, -1), 5)
                    predicted_category = metadata[indices[0][0]]['category'] if len(indices[0]) > 0 else "Unknown"
            else:
                scores, indices, predicted_category = None, None, "Unknown"
            
            st.markdown(f"### Căutare după: **{text_query if search_type == 'Descriere Textuală (Text-to-Image)' else 'Imagine'}**")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
                if img is not None:
                    st.image(img, caption="Input Image", use_container_width=True)
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>Obiect Detectat</div><div class='metric-value'>{predicted_category.upper()}</div></div>", unsafe_allow_html=True)
                    
                    if scores is not None and len(scores[0]) > 0:
                        top_score = scores[0][0]
                        acc_pct = int((top_score + 1) / 2 * 100) if "CLIP" in embed_engine else int((1 - top_score/2) * 100)
                        if top_score <= 1.0: acc_pct = int(top_score * 100)
                        else: acc_pct = int((1 / (1 + top_score)) * 100)
                        st.markdown(f"<div class='metric-card' style='background: linear-gradient(135deg, #10b981 0%, #047857 100%);'><div class='metric-label'>Acuratețe Detecție</div><div class='metric-value'>{acc_pct}%</div></div>", unsafe_allow_html=True)
                else:
                    st.info(f"Query: {text_query}")
                st.markdown("</div>", unsafe_allow_html=True)
            
                import plotly.graph_objects as go
                st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
                
                # Radar Chart for Semantic / Embedding Signature
                if "CLIP" in embed_engine:
                    semantic_concepts = ["Streetwear", "Elegant", "Sport", "Vintage", "Minimalist", "Summer", "Winter", "Colorful", "Dark", "Denim", "Leather"]
                    with torch.no_grad():
                        concept_inputs = clip_processor(text=semantic_concepts, return_tensors="pt", padding=True).to(DEVICE)
                        concept_embs = clip_model.get_text_features(**concept_inputs).pooler_output
                        concept_embs = concept_embs / concept_embs.norm(dim=-1, keepdim=True)
                        concept_embs = concept_embs.cpu().numpy().astype("float32")
                    
                    sims = np.dot(concept_embs, emb.reshape(-1))
                    # Rescale to 0-1 for radar
                    top_vals = np.clip((sims + 0.1) * 2, 0, 1) # simple scaling to look good
                    categories = semantic_concepts
                    chart_title = "Semantic Concept Breakdown (Cross-Modal Projection)"
                else:
                    top_k_dims = 12
                    top_indices = np.argsort(np.abs(emb))[-top_k_dims:]
                    top_vals = emb[top_indices]
                    categories = [f"Dim {i}" for i in top_indices]
                    chart_title = "Top Embedding Dimensions (Radar Signature)"
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=top_vals,
                    theta=categories,
                    fill='toself',
                    name='Embedding Signature',
                    line_color='#ec4899',
                    fillcolor='rgba(236, 72, 153, 0.3)'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=False, range=[0 if "CLIP" in embed_engine else -1, 1], gridcolor="rgba(255,255,255,0.1)"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=10)),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    showlegend=False,
                    title=dict(text=chart_title, font=dict(color="#f8fafc", size=14)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=40, b=20, l=40, r=40),
                    font=dict(color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interactive Plotly Bar Chart instead of Matplotlib
                fig_bar = go.Figure(data=[
                    go.Bar(y=emb, marker_color='#8b5cf6')
                ])
                fig_bar.update_layout(
                    title=dict(text=f"Full Vector Embedding ({len(emb)} Dimensiuni)", font=dict(color="#f8fafc", size=14)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=40, b=20, l=10, r=10),
                    height=200,
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                    font=dict(color="#cbd5e1")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with st.expander("Analiză Matematică Avansată a Spațiului Vectorial"):
                st.markdown("""
                ### Cum funcționează magia din spate?
                1. **Extragerea Trăsăturilor (Feature Extraction):** Imaginea ta este procesată folosind arhitectura *ViT (Vision Transformer)* din CLIP sau *EfficientNetV2*. Modelul extrage un vector de densitate cu 512 dimensiuni.
                2. **L2 Normalization:** Vectorul este normalizat $\\frac{v}{\\|v\\|_2}$ pentru a fi plasat pe o hipersferă. Aceasta asigură că magnitudinea nu influențează calculul distanței.
                3. **Distanța Cosinus:** Pentru a găsi haine similare, folosim indexul FAISS pentru a calcula produsul scalar (care acum e echivalent cu distanța cosinus, datorită normalizării L2) între vectorul tău și toți cei 2000 de vectori din baza de date. Formulele sunt:
                $$ \\text{sim}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} $$
                4. **Căutare Cross-Modală:** Când folosești text, encoder-ul de text din CLIP transformă textul tău în *același spațiu matematic* ca și imaginile, permițând comparația directă!
                """)
            
            if faiss_index is not None:
                st.markdown("<h3 style='margin-top: 30px; color: #2c3e50;'>Top Articole Similare</h3>", unsafe_allow_html=True)
                sim_cols = st.columns(5)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1: continue
                    meta = metadata[idx]
                    sim_path = meta["crop_path"]
                    if Path(sim_path).exists():
                        with sim_cols[i]:
                            st.markdown(f"<div class='sim-card'>", unsafe_allow_html=True)
                            st.image(Image.open(sim_path), use_container_width=True)
                            st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: 800; color: #f8fafc; letter-spacing: 1px;'>{meta['category'].upper()}</div>", unsafe_allow_html=True)
                            
                            # Calculate similarity percentage from cosine distance (approx)
                            sim_pct = int((score + 1) / 2 * 100) if "CLIP" in embed_engine else int((1 - score/2) * 100) # Simple heuristic mapping depending on exact metric
                            if score <= 1.0:
                                sim_pct = int(score * 100)
                            else:
                                sim_pct = int((1 / (1 + score)) * 100) # L2 distance mapping
                                
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.1); border-radius: 6px; height: 8px; margin-top: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);'>
                                <div style='background: linear-gradient(90deg, #ec4899, #8b5cf6); width: {sim_pct}%; height: 100%; border-radius: 6px; box-shadow: 0 0 10px rgba(236,72,153,0.8);'></div>
                            </div>
                            <div style='text-align: right; font-size: 12px; color: #cbd5e1; margin-top: 4px; font-weight: bold;'>{sim_pct}% SIMILARITY</div>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                
                # Next Level: Similarity Matrix Heatmap (Query vs Top 5)
                st.markdown("<h3 style='margin-top: 40px; color: #2c3e50;'>Embedding Correlation Heatmap</h3>", unsafe_allow_html=True)
                st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
                st.markdown("Această matrice analizează cum se corelează matematic (Cosine Similarity) **Query-ul tău** cu cele **5 rezultate găsite**. Observă cum rezultatele nu sunt similare doar cu ce ai căutat, ci și între ele, formând un spațiu semantic coerent!")
                
                try:
                    # Exact Correlation Matrix Calculation
                    actual_embs = [emb] # index 0 is query
                    labels = ["QUERY"]
                    
                    with torch.no_grad():
                        for i, idx in enumerate(indices[0][:5]):
                            if idx == -1: continue
                            meta = metadata[idx]
                            sim_path = meta["crop_path"]
                            if not Path(sim_path).exists(): continue
                            
                            labels.append(f"Rank {i+1} ({meta['category'].title()})")
                            res_img = Image.open(sim_path).convert("RGB")
                            
                            if embed_engine == "CLIP (Vision + Text Cross-Modal)" or embed_engine == "FashionCLIP (Fine‑tuned CLIP for fashion)":
                                res_inputs = clip_processor(images=res_img, return_tensors="pt").to(DEVICE)
                                res_emb = clip_model.get_image_features(**res_inputs).pooler_output
                                res_emb = res_emb / res_emb.norm(dim=-1, keepdim=True)
                                actual_embs.append(res_emb.squeeze(0).cpu().numpy().astype("float32"))
                            elif embed_engine == "DINOv2 (Self‑Supervised Vision)":
                                res_tensor = inference_transform(res_img).unsqueeze(0).to(DEVICE)
                                res_emb = dino_model(res_tensor)
                                res_emb = res_emb / res_emb.norm(dim=-1, keepdim=True)
                                actual_embs.append(res_emb.squeeze(0).cpu().numpy().astype("float32"))
                            else:
                                res_tensor = inference_transform(res_img).unsqueeze(0).to(DEVICE)
                                res_emb = model(res_tensor).squeeze(0).cpu().numpy().astype("float32")
                                actual_embs.append(res_emb)
                                
                    actual_embs = np.array(actual_embs)
                    # Exact cosine similarity matrix using dot product (since vectors are L2 normalized)
                    corr_matrix = np.dot(actual_embs, actual_embs.T)
                    
                    # Normalize to [0,1] range for better heatmap display if negative values exist
                    corr_matrix = np.clip(corr_matrix, 0, 1)
                            
                    import plotly.express as px
                    fig_heat = px.imshow(
                        corr_matrix, 
                        x=labels, y=labels, 
                        color_continuous_scale="Purpor", # Purple to Pink scale
                        text_auto=".2f",
                        aspect="auto"
                    )
                    fig_heat.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1"),
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                except Exception as e:
                    st.error(f"Eroare heatmap: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

@st.dialog("Antrenare Retea (Fine-Tuning)", width="large")
def show_fine_tuning_dialog():
    st.markdown("<p style='font-size: 12px; color: #cbd5e1;'>Alege arhitectura, sursa de date și parametrii de antrenare:</p>", unsafe_allow_html=True)
    train_model_choice = st.selectbox("Model de Antrenat", ["EfficientNetV2 (Local)", "CLIP (Multi-Modal)"])

    dataset_source = st.selectbox("Sursă Dataset", [
        "Catalog Curent (YOLO Crops)",
        "Folder Extern (ImageFolder format)",
        "HuggingFace Hub"
    ])

    custom_dataset_path = None
    if dataset_source == "Folder Extern (ImageFolder format)":
        custom_dataset_path = st.text_input("Calea absolută către directorul cu imagini", value="/cale/catre/dataset")
    elif dataset_source == "HuggingFace Hub":
        custom_dataset_path = st.text_input("Nume Dataset HuggingFace", value="ashraq/fashion-product-images-small")
        st.info("Conexiunea la HuggingFace va rula în mod Streaming (fără descărcare completă).")

    if train_model_choice == "CLIP (Multi-Modal)":
        st.info("Vom antrena straturile de proiecție ale modelului CLIP cu Contrastive Loss.")
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)
        num_epochs = st.number_input("Număr Epoci", 1, 20, 5)
        lr = st.number_input("Learning Rate", 1e-6, 1e-3, 5e-5, format="%.6f")
        start_train = st.button("Începe Antrenarea CLIP", use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            triplet_margin = st.slider("Triplet Margin", 0.1, 1.0, 0.4, 0.05)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)
            num_epochs = st.number_input("Număr Epoci", 1, 100, 15)
        with col2:
            lr = st.number_input("Learning Rate", 1e-6, 1e-3, 1e-4, format="%.6f")
            lr_head = st.number_input("LR (Head)", 1e-5, 1e-2, 8e-4, format="%.6f")
            hard_negative_epoch = st.number_input("Hard Neg (Epoca)", 1, num_epochs, 5)
        start_train = st.button("Începe Antrenarea EfficientNet", use_container_width=True)

    log_placeholder = st.empty()
    plot_placeholder = st.empty()

    if start_train:
        # Încărcăm manifestul local de la bun început, indiferent de sursa de antrenare,
        # pentru că avem absolută nevoie de el la final pentru FAISS!
        manifest_path = EMB_DIR / "crop_manifest.json"
        if not manifest_path.exists():
            st.error("crop_manifest.json nu a fost găsit! Extrage crop-urile cu YOLO în interfața principală înainte de antrenare.")
            return
        with open(manifest_path) as f:
            local_manifest = json.load(f)

        if dataset_source == "Folder Extern (ImageFolder format)":
            if not Path(custom_dataset_path).exists():
                st.error("Directorul extern specificat nu există!")
                return

        log_text = f"**Training started on {DEVICE}**\n\n"
        log_placeholder.markdown(log_text)

        # ──────────────────────────────────────────────────────────────────────
        # REGIM ANTRENARE 1: CLIP
        # ──────────────────────────────────────────────────────────────────────
        if train_model_choice == "CLIP (Multi-Modal)":
            from transformers import CLIPProcessor, CLIPModel
            model_id = "openai/clip-vit-base-patch32"
            clip_model_ft = CLIPModel.from_pretrained(model_id).to(DEVICE)
            clip_proc_ft = CLIPProcessor.from_pretrained(model_id)
            for param in clip_model_ft.parameters(): param.requires_grad = False
            for param in clip_model_ft.visual_projection.parameters(): param.requires_grad = True
            for param in clip_model_ft.text_projection.parameters(): param.requires_grad = True

            if dataset_source == "Catalog Curent (YOLO Crops)":
                class FashionCLIPDataset(torch.utils.data.Dataset):
                    def __init__(self, mf):
                        if isinstance(mf, dict):
                            self.items = [v for v in mf.values() if v.get("is_crop", False) and Path(v["crop_path"]).exists()]
                        else:
                            self.items = [v for v in mf if v.get("is_crop", False) and Path(v["crop_path"]).exists()]
                    def __len__(self): return len(self.items)
                    def __getitem__(self, idx):
                        it = self.items[idx]
                        return Image.open(it["crop_path"]).convert("RGB"), f"a photo of a {it['category']}"
                ft_dataset = FashionCLIPDataset(local_manifest)
            elif dataset_source == "Folder Extern (ImageFolder format)":
                import torchvision.datasets as tv_datasets
                class FolderCLIPDataset(torch.utils.data.Dataset):
                    def __init__(self, root):
                        self.ds = tv_datasets.ImageFolder(root)
                    def __len__(self): return len(self.ds)
                    def __getitem__(self, idx):
                        img, label = self.ds[idx]
                        return img.convert("RGB"), f"a photo of a {self.ds.classes[label]}"
                ft_dataset = FolderCLIPDataset(custom_dataset_path)
            else:
                try:
                    from datasets import load_dataset
                    from torchvision import transforms as tv_t
                    hf_ds = load_dataset(custom_dataset_path, split="train", streaming=True)
                    st.success("Dataset HuggingFace conectat (Streaming)!")
                    _pre = tv_t.Compose([tv_t.Resize((224, 224)), tv_t.ToTensor()])
                    class HFStreamDataset(torch.utils.data.IterableDataset):
                        def __init__(self, ds, tf): self.ds, self.tf = ds, tf
                        def __iter__(self):
                            for s in self.ds:
                                try:
                                    img = s.get("image") or s.get("img")
                                    if isinstance(img, str):
                                        import requests, io
                                        img = Image.open(io.BytesIO(requests.get(img, timeout=5).content))
                                    img = img.convert("RGB")
                                    txt = f"a photo of a {s.get('product', s.get('label', 'fashion item'))}"
                                    yield self.tf(img), txt
                                except Exception: continue
                    ft_dataset = HFStreamDataset(hf_ds, _pre)
                except Exception as e:
                    st.error(f"Eroare HuggingFace: {e}")
                    return

            def clip_collate(batch):
                imgs, txts = zip(*batch)
                return clip_proc_ft(text=list(txts), images=list(imgs), return_tensors="pt", padding=True)

            use_shuffle = dataset_source != "HuggingFace Hub"
            ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=use_shuffle, collate_fn=clip_collate, num_workers=0)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, clip_model_ft.parameters()), lr=lr)
            history = {"loss": []}
            pb = st.progress(0)
            
            for epoch in range(int(num_epochs)):
                clip_model_ft.train()
                epoch_loss, n_batches = 0.0, 0
                for i, batch in enumerate(ft_loader):
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    optimizer.zero_grad()
                    out = clip_model_ft(**batch, return_loss=True)
                    out.loss.backward()
                    optimizer.step()
                    epoch_loss += out.loss.item()
                    n_batches = i + 1
                avg = epoch_loss / max(n_batches, 1)
                history["loss"].append(avg)
                log_text += f"Epoch {epoch+1}/{num_epochs} | Loss: {avg:.4f}\n\n"
                log_placeholder.markdown(log_text)

            st.success("Antrenare CLIP Completa!")
            torch.save(clip_model_ft.state_dict(), CKPT_DIR / "clip_finetuned.pth")
            #pb.progress(min(n_batches / max(len(ft_loader) if hasattr(ft_loader.dataset, "__len__") else 100, 1), 1.0))
            pb.progress((epoch + 1) / int(num_epochs))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history["loss"], marker="o", color="#ec4899")
            ax.set_title("CLIP Fine-Tuning Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True)
            plot_placeholder.pyplot(fig)

        # ──────────────────────────────────────────────────────────────────────
        # REGIM ANTRENARE 2: EFFICIENTNET (TRIPLET LOSS)
        # ──────────────────────────────────────────────────────────────────────
        else:
            if dataset_source == "Catalog Curent (YOLO Crops)":
                eff_dataset = FashionTripletDataset(local_manifest, transform=train_transform, hard_negative=False)
            elif dataset_source == "Folder Extern (ImageFolder format)":
                import torchvision.datasets as tv_datasets, random
                class TripletFolderDataset(torch.utils.data.Dataset):
                    def __init__(self, root, tf):
                        self.ds = tv_datasets.ImageFolder(root)
                        self.tf = tf
                        self.lbl2idx = {}
                        for i, (_, lbl) in enumerate(self.ds.samples):
                            self.lbl2idx.setdefault(lbl, []).append(i)
                        self.lbls = list(self.lbl2idx.keys())
                    def __len__(self): return len(self.ds)
                    def set_hard_negative(self, v): pass
                    def __getitem__(self, idx):
                        img, lbl = self.ds[idx]
                        pos_i = random.choice(self.lbl2idx[lbl])
                        neg_l = random.choice([l for l in self.lbls if l != lbl])
                        neg_i = random.choice(self.lbl2idx[neg_l])
                        return (self.tf(img.convert("RGB")),
                                self.tf(Image.open(self.ds.samples[pos_i][0]).convert("RGB")),
                                self.tf(Image.open(self.ds.samples[neg_i][0]).convert("RGB")),
                                str(lbl))
                eff_dataset = TripletFolderDataset(custom_dataset_path, train_transform)
            else:
                st.error("HuggingFace nu este suportat pentru EfficientNet Triplet. Folosiți Folder Extern.")
                return

            eff_loader = DataLoader(eff_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=(DEVICE.type != "cpu"))
            model_eff = FashionEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
            criterion = TripletLoss(margin=triplet_margin)
            
            optimizer = torch.optim.AdamW([
                {"params": [p for block in list(model_eff.backbone.blocks)[-2:] for p in block.parameters()]
                          + list(model_eff.backbone.conv_head.parameters())
                          + list(model_eff.backbone.bn2.parameters()), "lr": lr},
                {"params": model_eff.projection.parameters(), "lr": lr_head}
            ], weight_decay=1e-4)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_epochs), eta_min=1e-6)
            history = {"loss": [], "active_fraction": []}
            best_loss = float("inf")
            pb = st.progress(0)

            for epoch in range(int(num_epochs)):
                if epoch >= int(hard_negative_epoch):
                    eff_dataset.set_hard_negative(True)
                    if epoch == int(hard_negative_epoch):
                        log_text += "*Switched to HARD negatives*\n\n"
                
                model_eff.train()
                epoch_loss, epoch_active, nb = 0.0, 0.0, 0
                for i, (anc, pos, neg, _) in enumerate(eff_loader):
                    anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
                    optimizer.zero_grad()
                    ea, ep, en = model_eff(anc), model_eff(pos), model_eff(neg)
                    loss, af = criterion(ea, ep, en)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_eff.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item(); epoch_active += af; nb = i + 1
                
                
                scheduler.step()
                al, aa = epoch_loss / nb, epoch_active / nb
                history["loss"].append(al); history["active_fraction"].append(aa)
                mode = "Hard" if epoch >= int(hard_negative_epoch) else "Rand"
                pb.progress((epoch + 1) / int(num_epochs))  # ← asta
                log_text += f"Epoch {epoch+1:02d}/{num_epochs} [{mode}] | Loss: {al:.4f} | Active: {aa:.1%}\n\n"
                log_placeholder.markdown(log_text)
                
                if al < best_loss:
                    best_loss = al
                    torch.save(model_eff.state_dict(), CKPT_DIR / "best_model.pth")

            st.success("Antrenare EfficientNet Completa!")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history["loss"], marker="o"); ax1.axvline(x=int(hard_negative_epoch), color="red", linestyle="--")
            ax1.set_title("Triplet Loss"); ax1.grid(True)
            ax2.plot(history["active_fraction"], marker="o", color="orange"); ax2.axvline(x=int(hard_negative_epoch), color="red", linestyle="--")
            ax2.set_title("Active Triplet Fraction"); ax2.grid(True)
            plot_placeholder.pyplot(fig)

        # ======================================================================
        # 🌟 RE-INDEXAREA SECURIZATĂ A CATALOGULUI LOCAL ÎN FAISS
        # ======================================================================
        log_text += "🔄 **Etapa Finală:** Se re-scanează catalogul de haine local cu noul model antrenat...\n\n"
        log_placeholder.markdown(log_text)
        
        if train_model_choice == "CLIP (Multi-Modal)":
            clip_model_ft.eval()
        else:
            model_eff.eval()
            
        all_vectors, all_metadata = [], []
        items_to_index = local_manifest if isinstance(local_manifest, list) else local_manifest.values()
        
        with torch.no_grad():
            for item in items_to_index:
                try:
                    if "crop_path" in item and Path(item["crop_path"]).exists():
                        img = Image.open(item["crop_path"]).convert("RGB")
                        
                        if train_model_choice == "CLIP (Multi-Modal)":
                            inputs = clip_proc_ft(images=img, return_tensors="pt").to(DEVICE)
                            emb = clip_model_ft.get_image_features(**inputs).squeeze(0).cpu().numpy()
                        else:
                            tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
                            emb = model_eff(tensor).squeeze(0).cpu().numpy()
                            
                        all_vectors.append(emb)
                        all_metadata.append(item)
                except Exception:
                    continue

        if len(all_vectors) > 0:
            vectors_matrix = np.vstack(all_vectors).astype("float32")
            
            # Salvăm noile fișiere binare și metadata
            np.save(EMB_DIR / "vectors.npy", vectors_matrix)
            with open(EMB_DIR / "metadata.json", "w") as f:
                json.dump(all_metadata, f, indent=2)

            # Suprascriem indexul FAISS cu noile proprietăți geometrice învățate
            dimension = vectors_matrix.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors_matrix)
            faiss.write_index(index, str(INDEX_DIR / "fashion.index"))

            log_text += "🎉 **Succes Total!** Catalogul local a fost re-indexat în FAISS. Noile coordonate matematice reflectă perfect antrenarea proaspătă!"
            log_placeholder.markdown(log_text)
            st.balloons()
        else:
            st.error("Eroare gravă: Nu s-au putut citi imaginile din crop_manifest.json pentru re-indexarea finală FAISS.")


st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Administrare Sistem")
if st.sidebar.button("Deschide Panoul de Antrenare (Fine-Tuning)", use_container_width=True):
    show_fine_tuning_dialog()

with tab3:
    st.header("AI Outfit Analyzer (WOW Feature)")
    st.markdown("Încarcă o poză cu un outfit complet. AI-ul va folosi **Object Detection** (YOLO) combinat cu **CLIP Embeddings** pentru a-ți spune ce porți, din ce categorie de stil face parte (ex: streetwear), și îți va oferi **Recomandări Similare**!")
    
    analyzer_img_file = st.file_uploader("Upload Outfit Image...", type=["jpg", "jpeg", "png"], key="outfit")
    
    if analyzer_img_file is not None:
        outfit_img = Image.open(analyzer_img_file).convert("RGB")
        st.image(outfit_img, caption="Analizăm Outfit-ul...", width=400)
        
        with st.spinner("Rulăm Object Detection (Fashionpedia) și Analiză CLIP..."):
            try:
                from huggingface_hub import hf_hub_download
                yolo_path = hf_hub_download(repo_id="louisJLN/yolo8-fashionpedia", filename="results/yolov8n-fashionpedia-1.onnx", timeout=10)
                detector = YOLO(yolo_path, task='detect')
                results = detector.predict(source=outfit_img, conf=0.25, verbose=False)[0]
                
                all_boxes = results.boxes.xyxy.cpu().numpy()
                all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                boxes = all_boxes
                class_ids = all_class_ids
                if hasattr(results, 'names') and len(results.names) > 0 and isinstance(results.names, dict):
                    detected_names = [results.names[c] for c in class_ids]
                else:
                    detected_names = [f"Item {c}" for c in class_ids]
                    
            except Exception as e:
                st.warning("⚠️ Fallback pe YOLOv8n standard din cauza limitărilor de rețea.")
                detector = YOLO("yolov8n.pt")
                results = detector.predict(source=outfit_img, conf=0.25, verbose=False)[0]
                
                all_boxes = results.boxes.xyxy.cpu().numpy()
                all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                allowed_classes = {0, 24, 27, 31}
                valid_idx = [i for i, c in enumerate(all_class_ids) if c in allowed_classes]
                boxes = all_boxes[valid_idx] if len(valid_idx) > 0 else []
                class_ids = all_class_ids[valid_idx] if len(valid_idx) > 0 else []
                detected_names = [results.names[c] for c in class_ids]
            
            model_id = "openai/clip-vit-base-patch32"
            clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE)
            clip_processor = CLIPProcessor.from_pretrained(model_id)
            clip_model.eval()
            
            clothing_labels = ["sneakers", "hoodie", "jeans", "t-shirt", "dress", "jacket", "cap", "sunglasses", "boots", "sweatpants", "shirt", "skirt", "bag"]
            style_labels = ["streetwear aesthetic", "business casual", "vintage retro", "bohemian chic", "sportswear active", "elegant evening", "minimalist"]
            
            with torch.no_grad():
                inputs = clip_processor(text=clothing_labels + style_labels, images=outfit_img, return_tensors="pt", padding=True).to(DEVICE)
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()
                
            cloth_probs = probs[:len(clothing_labels)]
            style_probs = probs[len(clothing_labels):]
            
            top_clothes_idx = np.argsort(cloth_probs)[-4:][::-1]
            detected_clothes = [clothing_labels[i] for i in top_clothes_idx if cloth_probs[i] > 0.02]
            
            for name in detected_names:
                if name in ['backpack', 'handbag', 'tie', 'umbrella', 'suitcase']:
                    if name not in detected_clothes: detected_clothes.append(name)
                    
            best_style_idx = np.argmax(style_probs)
            best_style = style_labels[best_style_idx]
            
            with torch.no_grad():
                img_emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                img_emb = img_emb.squeeze(0).cpu().numpy().astype("float32")
                
            index_path = INDEX_DIR / "fashion_clip.index"
            meta_path = EMB_DIR / "metadata.json"
            
            st.success(f"### {best_style.upper()} DETECTED")
            
            detected_str = ", ".join(detected_clothes).title() if detected_clothes else "General Apparel"
            st.markdown(f"**Elemente Recunoscute**: {detected_str}")
            
            # --- AUDIO GENERATION ---
            try:
                from gtts import gTTS
                import io
                text_to_speak = f"Analiza vizuală completă. Categoria dominantă pentru acest outfit este {best_style.replace(' aesthetic', '')}. Articolele vestimentare detectate sunt: {detected_str}."
                tts = gTTS(text=text_to_speak, lang='ro')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format='audio/mp3', autoplay=True)
            except Exception as e:
                st.error(f"Eroare la generarea sunetului: {e}")
            
            if index_path.exists() and meta_path.exists():
                import faiss
                import json
                faiss_index = faiss.read_index(str(index_path))
                with open(meta_path) as f: metadata = json.load(f)
                
                scores, indices = faiss_index.search(img_emb.reshape(1, -1), 5)
                st.markdown("---")
                st.markdown("<h3 style='color: #2c3e50;'> Top Recomandări Similare din Magazin (Local)</h3>", unsafe_allow_html=True)
                
                sim_cols = st.columns(5)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1: continue
                    meta = metadata[idx]
                    sim_path = meta["crop_path"]
                    if Path(sim_path).exists():
                        with sim_cols[i]:
                            st.markdown(f"<div class='sim-card'>", unsafe_allow_html=True)
                            st.image(Image.open(sim_path), use_container_width=True)
                            st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: 800; color: #f8fafc; letter-spacing: 1px;'>{meta['category'].upper()}</div>", unsafe_allow_html=True)
                            
                            sim_pct = int((score + 1) / 2 * 100) if score <= 1.0 else int((1 / (1 + score)) * 100)
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.1); border-radius: 6px; height: 8px; margin-top: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);'>
                                <div style='background: linear-gradient(90deg, #ec4899, #8b5cf6); width: {sim_pct}%; height: 100%; border-radius: 6px; box-shadow: 0 0 10px rgba(236,72,153,0.8);'></div>
                            </div>
                            <div style='text-align: right; font-size: 12px; color: #cbd5e1; margin-top: 4px; font-weight: bold;'>{sim_pct}% MATCH</div>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                
                import urllib.parse
                from io import BytesIO
                import requests as req

                CATALOG = {
                    "sneakers":    [("1542291026-7eec264c27ff", "sneakers"), ("1606107557195-0e29a4b5b4aa", "sneakers"), ("1600185365483-26d7a4cc7519", "sneakers running")],
                    "hoodie":      [("1556821840-3a63f8a79c65", "hoodie"), ("1503341504253-dff4815485f1", "hoodie streetwear"), ("1521572163474-6864f9cf17ab", "hoodie outfit")],
                    "jeans":       [("1541099649105-f69ad21f3246", "jeans"), ("1598554747436-c9293d6a588f", "denim jeans"), ("1506629082955-511b1aa562c8", "jeans outfit")],
                    "t-shirt":     [("1521572163474-6864f9cf17ab", "t-shirt"), ("1583744946564-b52d01e7f922", "tshirt outfit"), ("1529374255-68dfe714cf3c", "white tshirt")],
                    "dress":       [("1496747611176-887e999e49f3", "dress"), ("1515372039744-b8f02a3ae446", "summer dress"), ("1539109136881-3be0616acf4b", "floral dress")],
                    "jacket":      [("1551028719-00167b16eac5", "jacket"), ("1548126032-079a0fb0099d", "leather jacket"), ("1591047139829-d91aecb6caea", "denim jacket")],
                    "cap":         [("1588850561407-ed78c282e89b", "baseball cap"), ("1534215754734-18e55168f0fd", "cap hat"), ("1521369909449-4463a878b896", "snapback cap")],
                    "sunglasses":  [("1511499767150-a7a1371514a6", "sunglasses"), ("1572635196237-14b3f281503f", "sunglasses fashion"), ("1473496169904-ecb85f3e3e22", "sunglasses summer")],
                    "boots":       [("1542291026-7eec264c27ff", "boots"), ("1608256246200-57b2e3c08d67", "ankle boots"), ("1605812860427-4024433a70fd", "boots fashion")],
                    "shirt":       [("1603252109303-2751441dd157", "shirt outfit"), ("1598033129183-c4f50c736f10", "formal shirt"), ("1596755389378-c31d21fd1273", "casual shirt")],
                    "skirt":       [("1496747611176-887e999e49f3", "skirt"), ("1515372039744-b8f02a3ae446", "midi skirt"), ("1583496661160-fb5218bebd71", "mini skirt")],
                    "bag":         [("1548036161-16ba38735571", "handbag"), ("1584917865442-de89df76afd3", "purse bag"), ("1553062407-98eeb64c6a62", "tote bag")],
                    "sweatpants":  [("1515886657613-9f3515b0c78f", "sweatpants"), ("1556821840-3a63f8a79c65", "jogger pants"), ("1506629082955-511b1aa562c8", "sweatpants outfit")],
                    "streetwear aesthetic": [("1529139574466-a303027c1d8b", "streetwear"), ("1552346989-e9f23b996c1e", "streetwear look"), ("1571781926291-c59274a27bb3", "urban streetwear")],
                    "business casual": [("1507679799987-c73779587ccf", "business casual"), ("1617127365659-c47fa864d8bc", "smart casual"), ("1490481651871-ab68de25d43d", "office outfit")],
                    "vintage retro": [("1525845859779-54d477ff291f", "vintage outfit"), ("1487222477099-a33044d3543d", "retro fashion"), ("1516762689617-e1cffcef479d", "vintage style")],
                    "bohemian chic": [("1496747611176-887e999e49f3", "boho outfit"), ("1524504388940-b1c51b5b1b3e", "bohemian style"), ("1469334031218-e382a71b716b", "boho chic")],
                    "sportswear active": [("1535743686741-6a5f0e83c6db", "sportswear"), ("1556817411-31ae72c54a3d", "activewear"), ("1506629082955-511b1aa562c8", "athletic wear")],
                    "elegant evening": [("1515886657613-9f3515b0c78f", "evening gown"), ("1496747611176-887e999e49f3", "elegant dress"), ("1583496661160-fb5218bebd71", "formal outfit")],
                    "minimalist": [("1523199455573-b04b2b47bdc7", "minimalist outfit"), ("1489987707849-2954a0f06674", "minimal fashion"), ("1465877783223-4ddc8d8c4ddf", "clean aesthetic")],
                }

                st.markdown("---")
                st.markdown("<h3 style='color: #2c3e50;'> Produse Similare din Magazine Online (Internet)</h3>", unsafe_allow_html=True)
                st.markdown("<div style='color: #94a3b8; margin-bottom: 20px;'>Am extras piesele cheie (fără OCR) direct din imaginea ta și am căutat echivalente pe platformele de shopping.</div>", unsafe_allow_html=True)
                
                # Filter out 'style' categories from catalog to only show specific products
                product_categories = [k for k in CATALOG.keys() if k not in style_labels]
                
                valid_items = [item for item in detected_names + detected_clothes if item in product_categories]
                valid_items = list(dict.fromkeys(valid_items)) # unique items
                if not valid_items:
                    valid_items = ["t-shirt", "jeans"] # fallback
                    
                inet_cols = st.columns(len(valid_items[:4]))
                for col, item in zip(inet_cols, valid_items[:4]):
                    entries = CATALOG.get(item, CATALOG.get("t-shirt"))
                    # Pick the first image variant
                    photo_id, shop_query = entries[0]
                    
                    img_url = f"https://images.unsplash.com/photo-{photo_id}?w=400&q=80&fit=crop"
                    shop_q = urllib.parse.quote(shop_query)
                    
                    try:
                        r = req.get(img_url, timeout=6)
                        if r.status_code == 200:
                            inet_img = Image.open(BytesIO(r.content)).convert("RGB")
                            with col:
                                st.markdown(f"<div class='dark-card' style='padding: 15px;'>", unsafe_allow_html=True)
                                st.image(inet_img, use_container_width=True)
                                
                                st.markdown(f"""
                                <div style='text-align: center; margin-top: 15px;'>
                                    <div style='font-weight: 900; color: #f8fafc; text-transform: uppercase; letter-spacing: 1px; font-size: 14px;'>{item}</div>
                                    <div style='color: #ec4899; font-weight: bold; font-size: 11px; margin: 5px 0 15px 0;'> ONLINE MATCH</div>
                                    <div style='display: flex; gap: 10px; justify-content: center;'>
                                        <a href='https://www.zalando.ro/catalog/?q={shop_q}' target='_blank' style='background: linear-gradient(90deg, #ec4899, #8b5cf6); color: white; padding: 8px 15px; border-radius: 8px; text-decoration: none; font-size: 12px; font-weight: bold; flex: 1;'>Zalando</a>
                                        <a href='https://www.asos.com/search/?q={shop_q}' target='_blank' style='background: rgba(255,255,255,0.1); color: white; padding: 8px 15px; border-radius: 8px; text-decoration: none; border: 1px solid rgba(255,255,255,0.2); font-size: 12px; font-weight: bold; flex: 1;'>ASOS</a>
                                    </div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception:
                        pass

with tab4:
    import tab4_logic
    
    # We load CLIP dynamically if not loaded, or just pass the ones from Tab3.
    # But since Tab3 loads them conditionally, let's load them here if needed.
    if 'clip_model' not in locals():
        model_id = "openai/clip-vit-base-patch32"
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(model_id)
        clip_model.eval()
        
    tab4_logic.render_tab4(DEVICE, clip_model, clip_processor, INDEX_DIR, EMB_DIR)
