import re

with open("app.py", "r") as f:
    content = f.read()

if "from transformers import CLIPProcessor" not in content:
    content = content.replace("import sys", "import sys\nfrom transformers import CLIPProcessor, CLIPModel")

new_tab1 = '''with tab1:
    st.header("Predict and Visualize")
    
    embed_engine = st.selectbox("Select Embedding Engine:", ["CLIP (Vision + Text Cross-Modal)", "EfficientNetV2 (Visual Only)"])
    
    if embed_engine == "CLIP (Vision + Text Cross-Modal)":
        index_path = INDEX_DIR / "fashion_clip.index"
        st.info("Using OpenAI's CLIP. State-of-the-art semantic search. No training required!")
    else:
        index_path = INDEX_DIR / "fashion.index"
        st.info("Using baseline EfficientNet. Works best after fine-tuning!")
        
    search_type = st.radio("Tipul de Căutare", ["Poză cu o haină (Image-to-Image)", "Descriere Textuală (Text-to-Image)"])
    
    uploaded_file = None
    text_query = ""
    
    if search_type == "Poză cu o haină (Image-to-Image)":
        uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])
    else:
        text_query = st.text_input("Caută prin text (ex: Rochie roșie elegantă):")
        
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        
    # Checkpoints for EfficientNet
    pth_files = list(CKPT_DIR.glob("*.pth"))
    pth_files.sort(key=lambda p: (p.name != "best_model.pth", p.name))
    selected_model = pth_files[0].name if pth_files else None
        
    if (img is not None or text_query) and (selected_model is not None or "CLIP" in embed_engine):
        
        with st.spinner("Processing & Encoding..."):
            
            # Load models
            if "CLIP" in embed_engine:
                model_id = "openai/clip-vit-base-patch32"
                clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE)
                clip_processor = CLIPProcessor.from_pretrained(model_id)
                clip_model.eval()
            else:
                model = FashionEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
                model.load_state_dict(torch.load(CKPT_DIR / selected_model, map_location=DEVICE))
                model.eval()
            
            # FAISS and Meta
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
            
            # Embed
            with torch.no_grad():
                if search_type == "Descriere Textuală (Text-to-Image)":
                    inputs = clip_processor(text=[text_query], return_tensors="pt").to(DEVICE)
                    emb = clip_model.get_text_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    emb = emb.squeeze(0).cpu().numpy().astype("float32")
                else:
                    if "CLIP" in embed_engine:
                        inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
                        emb = clip_model.get_image_features(**inputs)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb = emb.squeeze(0).cpu().numpy().astype("float32")
                    else:
                        tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
                        emb = model(tensor).squeeze(0).cpu().numpy().astype("float32")
            
            if faiss_index is not None:
                scores, indices = faiss_index.search(emb.reshape(1, -1), 5)
                predicted_category = metadata[indices[0][0]]['category'] if len(indices[0]) > 0 else "Unknown"
            else:
                scores, indices, predicted_category = None, None, "Unknown"
            
            st.markdown(f"### Căutare după: **{text_query if search_type == 'Descriere Textuală (Text-to-Image)' else 'Imagine'}**")
            if search_type == "Poză cu o haină (Image-to-Image)":
                st.markdown(f"**Obiect Detectat Vizual:** {predicted_category}")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if img is not None:
                    st.image(img, caption="Input Image", width=300)
                else:
                    st.info(f"Query: {text_query}")
            
            with col2:
                # Plot the embedding
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.bar(range(len(emb)), emb, color='steelblue', alpha=0.8)
                ax.set_title(f"Vector Embedding ({len(emb)} Dimensiuni)")
                ax.set_xlabel("Dimensiune")
                ax.set_ylabel("Valoare")
                st.pyplot(fig)
                
            with st.expander("Cum se face Embedding-ul cross-modal? (Apasă pentru explicație)"):
                st.markdown("""
                1. Textul sau imaginea sunt trecute printr-o **Rețea Neuronală**.
                2. Rețeaua traduce conceptele (cum ar fi cuvântul "rochie" sau forma fizică a unei rochii din poză) într-un șir matematic (vectorul de deasupra).
                3. Deoarece folosim **CLIP (OpenAI)**, cuvântul "rochie" și pozele cu rochii primesc **același vector matematic**! Se află în același punct în spațiul 512D.
                4. Căutăm acest vector cu FAISS (Distanța Cosinusului) în baza de date cu 2000 de haine și le returnăm pe cele mai apropiate!
                """)
            
            if faiss_index is not None:
                st.markdown("**Top 5 Articole Similare:**")
                sim_cols = st.columns(5)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1: continue
                    meta = metadata[idx]
                    sim_path = meta["crop_path"]
                    if Path(sim_path).exists():
                        with sim_cols[i]:
                            st.image(Image.open(sim_path), caption=f"{meta['garment_type']} (sim: {score:.3f})")
'''

import re
pattern = re.compile(r'with tab1:.*?(?=with tab2:)', re.DOTALL)
content = pattern.sub(new_tab1 + "\n", content)

with open("app.py", "w") as f:
    f.write(content)

print("Patched app.py successfully!")
