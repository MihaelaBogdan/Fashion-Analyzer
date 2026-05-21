import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import faiss
import json
from pathlib import Path

@st.cache_data
def get_pca_projection(emb_dir):
    vec_path = Path(emb_dir) / "vectors.npy"
    meta_path = Path(emb_dir) / "metadata.json"
    
    if not vec_path.exists() or not meta_path.exists():
        return None, None
        
    vectors = np.load(vec_path)
    with open(meta_path) as f:
        metadata = json.load(f)
        
    # sample to 1000 for faster render if large
    n_samples = min(1000, len(vectors))
    indices = np.random.choice(len(vectors), n_samples, replace=False)
    
    sampled_vecs = vectors[indices]
    sampled_meta = [metadata[i] for i in indices]
    
    pca = PCA(n_components=3)
    proj_3d = pca.fit_transform(sampled_vecs)
    
    return proj_3d, sampled_meta

def render_tab4(DEVICE, clip_model, clip_processor, INDEX_DIR, EMB_DIR):
    st.markdown("<h2 style='text-align: center; color: #ec4899;'> Latent Space Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Aprofundează conceptele matematice din spatele rețelelor neuronale cu aceste 3 demonstrații interactive avansate.</p>", unsafe_allow_html=True)
    
    feature_sel = st.radio("Alege Demonstrația:", 
        ["1. Matematică Semantică (Vector Arithmetic)", 
         "2. Proiecție 3D (PCA Latent Space)", 
         "3. Hibridizare (Latent Interpolation)",
         "4. Coerența Garderobei (Wardrobe Cohesion)"], horizontal=True)
         
    st.markdown("---")
    
    index_path = Path(INDEX_DIR) / "fashion_clip.index"
    meta_path = Path(EMB_DIR) / "metadata.json"
    
    if not index_path.exists() or not meta_path.exists():
        st.warning("Indexul CLIP nu a fost găsit. Rulați setup_clip.py.")
        return
        
    faiss_index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        metadata = json.load(f)
        
    if "Matematică" in feature_sel:
        st.markdown("###  Vector Arithmetic (Image + Text - Text)")
        st.markdown("<p style='color: #cbd5e1;'>Demonstrăm că 'Sensul' (Semantica) este o direcție pur matematică în spațiul latent. Alege o poză și aplică operații algebrice cu anumite cuvinte!</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            math_img_file = st.file_uploader("Upload Poză Bază (A)", type=["jpg", "jpeg", "png"], key="math_img")
            if math_img_file:
                img_a = Image.open(math_img_file).convert("RGB")
                st.image(img_a, caption="Imagine Bază", width=200)
                
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            text_plus = st.text_input(" Adună Conceptul (ex: 'blue', 'summer', 'floral')", key="plus")
            text_minus = st.text_input(" Scade Conceptul (ex: 'red', 'winter', 'formal')", key="minus")
            
        if st.button(" Calculează Ecuația Vectorială", use_container_width=True) and math_img_file:
            with st.spinner("Procesare matriceală..."):
                with torch.no_grad():
                    # 1. Image Embedding
                    inputs_img = clip_processor(images=img_a, return_tensors="pt").to(DEVICE)
                    img_emb = clip_model.get_image_features(**inputs_img).pooler_output
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    
                    # 2. Text Plus
                    if text_plus:
                        inputs_p = clip_processor(text=[text_plus], return_tensors="pt").to(DEVICE)
                        p_emb = clip_model.get_text_features(**inputs_p).pooler_output
                        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
                    else:
                        p_emb = torch.zeros_like(img_emb)
                        
                    # 3. Text Minus
                    if text_minus:
                        inputs_m = clip_processor(text=[text_minus], return_tensors="pt").to(DEVICE)
                        m_emb = clip_model.get_text_features(**inputs_m).pooler_output
                        m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)
                    else:
                        m_emb = torch.zeros_like(img_emb)
                        
                    # THE MATH
                    final_emb = img_emb + 0.5 * p_emb - 0.5 * m_emb
                    final_emb = final_emb / final_emb.norm(dim=-1, keepdim=True)
                    final_emb_np = final_emb.squeeze(0).cpu().numpy().astype("float32")
                    
                scores, indices = faiss_index.search(final_emb_np.reshape(1, -1), 4)
                
                st.markdown("####  Rezultatul Ecuației:")
                res_cols = st.columns(4)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1: continue
                    meta = metadata[idx]
                    sim_path = meta["crop_path"]
                    if Path(sim_path).exists():
                        with res_cols[i]:
                            st.markdown(f"<div class='dark-card' style='padding:10px; text-align:center;'>", unsafe_allow_html=True)
                            st.image(Image.open(sim_path), use_container_width=True)
                            st.markdown(f"<div style='margin-top:5px; font-weight:bold; color:#ec4899;'>Distanța Cos: {score:.3f}</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

    elif "3D" in feature_sel:
        st.markdown("###  Proiecția PCA 3D a Bazei de Date")
        st.markdown("<p style='color: #cbd5e1;'>Am redus dimensiunile embedding-urilor de la 512D la 3D folosind PCA pentru a putea 'vizualiza' spațiul semantic creat de AI. Observează cum elementele similare formează 'galaxii' sau clustere!</p>", unsafe_allow_html=True)
        
        with st.spinner("Calculăm proiecția PCA..."):
            proj_3d, meta = get_pca_projection(EMB_DIR)
            if proj_3d is not None:
                categories = [m.get("category", "unknown") for m in meta]
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=proj_3d[:, 0],
                    y=proj_3d[:, 1],
                    z=proj_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=[hash(c) % 256 for c in categories],
                        colorscale='Sunsetdark',
                        opacity=0.8,
                        line=dict(width=1, color='rgba(255,255,255,0.1)')
                    ),
                    text=[f"Cat: {c}" for c in categories],
                    hoverinfo='text'
                )])
                
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    scene=dict(
                        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                    ),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Nu s-au găsit datele pentru a genera spațiul 3D.")

    elif "Hibridizare" in feature_sel:
        st.markdown("###  Latent Space Interpolation (Găsirea Hibridului)")
        st.markdown("<p style='color: #cbd5e1;'>Ce se află *exact la jumătatea* distanței dintre două haine complet diferite? Acest instrument extrage mediul matematic dintre 2 imagini și găsește cea mai apropiată formă fizică din magazin!</p>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            img1_file = st.file_uploader("Poză A", type=["jpg", "png"], key="img1")
        with c2:
            img2_file = st.file_uploader("Poză B", type=["jpg", "png"], key="img2")
            
        if img1_file and img2_file:
            im1 = Image.open(img1_file).convert("RGB")
            im2 = Image.open(img2_file).convert("RGB")
            
            sc1, sc2, sc3 = st.columns([2, 1, 2])
            with sc1: st.image(im1, caption="Element A", use_container_width=True)
            with sc2: st.markdown("<h1 style='text-align:center; color:#ec4899; margin-top:50%;'>+</h1>", unsafe_allow_html=True)
            with sc3: st.image(im2, caption="Element B", use_container_width=True)
            
            if st.button(" Generează Hibridul (A + B / 2)", use_container_width=True):
                with st.spinner("Interpolăm spațiul..."):
                    with torch.no_grad():
                        i1 = clip_processor(images=im1, return_tensors="pt").to(DEVICE)
                        e1 = clip_model.get_image_features(**i1).pooler_output
                        e1 = e1 / e1.norm(dim=-1, keepdim=True)
                        
                        i2 = clip_processor(images=im2, return_tensors="pt").to(DEVICE)
                        e2 = clip_model.get_image_features(**i2).pooler_output
                        e2 = e2 / e2.norm(dim=-1, keepdim=True)
                        
                        # Interpolation
                        e_hybrid = (e1 + e2) / 2.0
                        e_hybrid = e_hybrid / e_hybrid.norm(dim=-1, keepdim=True)
                        e_hybrid_np = e_hybrid.squeeze(0).cpu().numpy().astype("float32")
                        
                    scores, indices = faiss_index.search(e_hybrid_np.reshape(1, -1), 4)
                    
                    st.markdown("####  Hibrizi Găsiți:")
                    res_cols = st.columns(4)
                    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        if idx == -1: continue
                        meta = metadata[idx]
                        sim_path = meta["crop_path"]
                        if Path(sim_path).exists():
                            with res_cols[i]:
                                st.markdown(f"<div class='dark-card' style='padding:10px; text-align:center;'>", unsafe_allow_html=True)
                                st.image(Image.open(sim_path), use_container_width=True)
                                st.markdown(f"<div style='margin-top:5px; font-weight:bold; color:#8b5cf6;'>Asemănare: {score:.3f}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
    elif "Coerența" in feature_sel:
        st.markdown("### Wardrobe Coherence & Style Centroid Builder")
        st.markdown("<p style='color: #cbd5e1;'>Încarcă poze cu hainele preferate din propria garderobă. AI-ul le va transforma în vectori de embeddings, va calcula gradul de coerență stilistică folosind o matrice de corelație cosinus, va deduce stilul tău predominant și îți va recomanda piese din magazin care se potrivesc ideal cu ceea ce deții deja!</p>", unsafe_allow_html=True)
        
        wardrobe_files = st.file_uploader("Încarcă poze cu hainele din garderoba ta (Selectează multiple)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="wardrobe_upload")
        
        if wardrobe_files and len(wardrobe_files) >= 2:
            st.markdown(f"**Garderoba ta conține {len(wardrobe_files)} piese selectate:**")
            
            # Afișare previzualizare piese încărcate
            grid_cols = st.columns(min(6, len(wardrobe_files)))
            images = []
            for i, f in enumerate(wardrobe_files):
                img = Image.open(f).convert("RGB")
                images.append(img)
                with grid_cols[i % 6]:
                    st.image(img, use_container_width=True, caption=f"Piesa {i+1}")
            
            if st.button("Analizează Coerența Stilistică", use_container_width=True):
                with st.spinner("Se calculează semnăturile matematice (Embeddings)..."):
                    embs = []
                    with torch.no_grad():
                        for img in images:
                            inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
                            emb = clip_model.get_image_features(**inputs).pooler_output
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                            embs.append(emb.squeeze(0).cpu().numpy().astype("float32"))
                    
                    embs = np.array(embs)
                    # Produs scalar pentru distanță Cosinus (vectori L2 normalizați)
                    sim_matrix = np.dot(embs, embs.T)
                    sim_matrix = np.clip(sim_matrix, 0, 1)
                    
                    # Media elementelor de pe diagonala superioară
                    n = len(images)
                    triu_indices = np.triu_indices(n, k=1)
                    avg_sim = np.mean(sim_matrix[triu_indices]) if n > 1 else 1.0
                    coherence_pct = int(avg_sim * 100)
                    
                    st.markdown("---")
                    
                    res_c1, res_c2 = st.columns([2, 3])
                    
                    with res_c1:
                        st.markdown("<h4 style='color: #f8fafc;'>Raport de Coerență Stilistică</h4>", unsafe_allow_html=True)
                        
                        if coherence_pct >= 75:
                            tier = "Garderobă Extrem de Coezivă "
                            color = "#10b981"
                            desc = "Piesele tale vestimentare se îmbină perfect, având o unitate estetică puternică. Ești foarte consecvent în stilul tău!"
                        elif coherence_pct >= 55:
                            tier = "Garderobă Versatilă / Mixtă "
                            color = "#3b82f6"
                            desc = "Ai un mix excelent de stiluri care pot fi combinate cu ușurință. Garderoba ta este atât practică, cât și variată!"
                        else:
                            tier = "Garderobă Eclectică / Diversă "
                            color = "#ec4899"
                            desc = "Piesele tale aparțin unor stiluri extrem de diverse. Acest lucru îți oferă unicitate, dar poate îngreuna asortarea rapidă."
                            
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); text-align: center;'>
                            <div style='font-size: 16px; color: #cbd5e1;'>Scor de Coerență</div>
                            <div style='font-size: 48px; font-weight: 900; color: {color}; margin: 10px 0;'>{coherence_pct}%</div>
                            <div style='font-size: 16px; font-weight: bold; color: #f8fafc;'>{tier}</div>
                            <p style='font-size: 13px; color: #94a3b8; margin-top: 10px; line-height: 1.4;'>{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Calculăm vectorul mediu (Centroidul Garderobei)
                        centroid_emb = np.mean(embs, axis=0)
                        centroid_emb = centroid_emb / np.linalg.norm(centroid_emb)
                        
                        # Proiectăm pe direcții stilistice semantice
                        semantic_concepts = ["Streetwear", "Business Casual", "Vintage", "Bohemian", "Sport", "Elegant Evening", "Minimalist"]
                        with torch.no_grad():
                            concept_inputs = clip_processor(text=semantic_concepts, return_tensors="pt", padding=True).to(DEVICE)
                            concept_embs = clip_model.get_text_features(**concept_inputs).pooler_output
                            concept_embs = concept_embs / concept_embs.norm(dim=-1, keepdim=True)
                            concept_embs = concept_embs.cpu().numpy().astype("float32")
                        
                        style_similarities = np.dot(concept_embs, centroid_emb)
                        dominant_concept = semantic_concepts[np.argmax(style_similarities)]
                        
                        st.markdown(f"<div style='margin-top: 15px; text-align: center; font-size: 14px; color: #f8fafc;'>Direcția stilistică dominantă: <span style='color: #8b5cf6; font-weight: bold;'>{dominant_concept}</span></div>", unsafe_allow_html=True)
                        
                    with res_c2:
                        st.markdown("<h4 style='color: #f8fafc;'>Matricea de Corelație Estetică (Cosinus Similarity)</h4>", unsafe_allow_html=True)
                        import plotly.express as px
                        labels = [f"Piesa {i+1}" for i in range(n)]
                        fig_heat = px.imshow(
                            sim_matrix,
                            labels=dict(x="Piese din Garderobă", y="Piese din Garderobă", color="Similaritate"),
                            x=labels, y=labels,
                            color_continuous_scale="Sunsetdark"
                        )
                        fig_heat.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=250,
                            font=dict(color="#cbd5e1")
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                        
                    # Recomandări FAISS folosind Centroidul Garderobei
                    st.markdown("---")
                    st.markdown("<h3 style='color: #f8fafc; text-align: center;'>Piese Recomandate pentru Completarea Garderobei</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 14px;'>Folosind Centroidul Garderobei tale, am interogat catalogul local pentru a găsi piesele care se armonizează cel mai bine cu ceea ce deții deja.</p>", unsafe_allow_html=True)
                    
                    # Recomandări FAISS folosind Centroidul Garderobei (deduplicate)
                    scores, indices = faiss_index.search(centroid_emb.reshape(1, -1), 30)
                    
                    unique_recs = []
                    seen_paths = set()
                    for score, idx in zip(scores[0], indices[0]):
                        if idx == -1: continue
                        meta = metadata[idx]
                        sim_path = meta["crop_path"]
                        if Path(sim_path).exists():
                            if sim_path in seen_paths:
                                continue
                            seen_paths.add(sim_path)
                            unique_recs.append((score, meta))
                            if len(unique_recs) >= 4:
                                break
                    
                    rec_cols = st.columns(len(unique_recs))
                    for i, (score, meta) in enumerate(unique_recs):
                        sim_path = meta["crop_path"]
                        with rec_cols[i]:
                            st.markdown(f"<div class='dark-card' style='padding:10px; text-align:center;'>", unsafe_allow_html=True)
                            st.image(Image.open(sim_path), use_container_width=True)
                            st.markdown(f"<div style='margin-top:5px; font-weight:bold; color:#ec4899;'>{meta['category'].upper()}</div>", unsafe_allow_html=True)
                            
                            match_pct = int((score + 1) / 2 * 100) if score <= 1.0 else int((1 / (1 + score)) * 100)
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.1); border-radius: 6px; height: 6px; margin-top: 8px; overflow: hidden;'>
                                <div style='background: linear-gradient(90deg, #ec4899, #8b5cf6); width: {match_pct}%; height: 100%; border-radius: 6px;'></div>
                            </div>
                            <div style='text-align: right; font-size: 11px; color: #cbd5e1; margin-top: 4px;'>{match_pct}% Compatibilitate</div>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                                
        elif wardrobe_files:
            st.warning("Vă rugăm să selectați cel puțin 2 piese vestimentare pentru a putea calcula corelațiile și gradul de coerență stilistică.")
