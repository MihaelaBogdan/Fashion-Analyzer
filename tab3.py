import sys

with open("app.py", "r") as f:
    content = f.read()

content = content.replace(
    'tab1, tab2 = st.tabs(["Prediction and Inference", "Fine-Tuning Model"])',
    'tab1, tab2, tab3 = st.tabs(["Prediction and Inference", "Fine-Tuning Model", "✨ AI Outfit Analyzer"])'
)

tab3_code = """
with tab3:
    st.header("AI Outfit Analyzer (WOW Feature)")
    st.markdown("Încarcă o poză cu un outfit complet. AI-ul va folosi **Object Detection** (YOLO) combinat cu **CLIP Embeddings** pentru a-ți spune ce porți, din ce categorie de stil face parte (ex: streetwear), și îți va oferi **Recomandări Similare**!")
    
    analyzer_img_file = st.file_uploader("Upload Outfit Image...", type=["jpg", "jpeg", "png"], key="outfit")
    
    if analyzer_img_file is not None:
        outfit_img = Image.open(analyzer_img_file).convert("RGB")
        st.image(outfit_img, caption="Analizăm Outfit-ul...", width=400)
        
        with st.spinner("Rulăm Object Detection și Analiză CLIP..."):
            # 1. Object Detection (YOLO)
            detector = YOLO("yolov8n.pt")
            results = detector.predict(source=outfit_img, conf=0.25, verbose=False)[0]
            
            boxes = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            detected_names = [results.names[c] for c in class_ids]
            
            # 2. CLIP Zero-Shot Classification for Specific Clothes & Styles
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
            
            # 3. Get Outfit Similarity (Recommendation System)
            with torch.no_grad():
                img_emb = clip_model.get_image_features(pixel_values=inputs.pixel_values).pooler_output
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                img_emb = img_emb.squeeze(0).cpu().numpy().astype("float32")
                
            index_path = INDEX_DIR / "fashion_clip.index"
            meta_path = EMB_DIR / "metadata.json"
            
            st.success(f"### ✨ {best_style.upper()} DETECTED ✨")
            
            detected_str = ", ".join(detected_clothes).title() if detected_clothes else "General Apparel"
            st.markdown(f"**Obiecte Detectate în Outfit:** {detected_str}")
            
            if index_path.exists() and meta_path.exists():
                import faiss
                import json
                faiss_index = faiss.read_index(str(index_path))
                with open(meta_path) as f: metadata = json.load(f)
                
                scores, indices = faiss_index.search(img_emb.reshape(1, -1), 5)
                st.markdown("---")
                st.markdown("### 🛍️ Recomandări Similare din Magazin:")
                
                sim_cols = st.columns(5)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1: continue
                    meta = metadata[idx]
                    sim_path = meta["crop_path"]
                    if Path(sim_path).exists():
                        with sim_cols[i]:
                            st.image(Image.open(sim_path), caption=f"{meta['category']} (Match: {score:.2f})")
"""

with open("app.py", "w") as f:
    f.write(content + "\n" + tab3_code)

print("Added tab3 successfully!")
