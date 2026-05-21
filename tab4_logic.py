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
         "3. Hibridizare (Latent Interpolation)"], horizontal=True)
         
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
