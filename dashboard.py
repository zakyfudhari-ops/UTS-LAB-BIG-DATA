"""
Dashboard Klasifikasi Gambar: Kucing vs Anjing
Dikembangkan dari struktur referensi (Mussliadii/GambarMus) dengan desain,
alur halaman, dan fitur yang dibangun ulang: tema warna berbeda, dua model
(.pt & .h5) yang bisa dibandingkan langsung, anotasi hasil pada gambar,
riwayat prediksi per sesi, dan form feedback yang tersimpan ke CSV.

Struktur folder:
├── dashboard.py
├── requirements.txt
├── packages.txt
├── model/
│   ├── KucingAnjing.pt
│   └── MZakyAlFudhari_Laporan2.h5
└── sample_images/        (opsional)
"""

import io
import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from plotly import express as px

# ==========================================================
# KONFIGURASI — sesuaikan dengan model kamu
# ==========================================================
CLASS_NAMES = ["Kucing", "Anjing"]     # urutan harus sama dengan saat training
IMG_SIZE = (224, 224)                   # ukuran input model
PYTORCH_MODEL_PATH = "model/KucingAnjing.pt"
KERAS_MODEL_PATH = "model/MZakyAlFudhari_Laporan2.h5"
FEEDBACK_CSV = "feedbacks.csv"

NAMA_PEMBUAT = "M. Zaky Al Fudhari"     # ganti sesuai nama kamu
NIM_PEMBUAT = "NIM kamu di sini"        # ganti sesuai NIM kamu
KAMPUS_DEFAULT = "Universitas kamu"

st.set_page_config(
    page_title="Dashboard Klasifikasi Kucing vs Anjing",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# TEMA (hijau-teal, beda dari referensi yang biru)
# ==========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f3d3e, #1f6f6b, #2fae8c);
    color: #eafff6;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }

.block-container {
    padding-top: 1.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

@keyframes riseIn {
    0% { opacity: 0; transform: translateY(16px); }
    100% { opacity: 1; transform: translateY(0); }
}

.main-title, .section-title, .result-card, .info-card {
    animation: riseIn 0.6s ease forwards;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.main-title:hover, .result-card:hover { transform: scale(1.01); }

.main-title {
    background: linear-gradient(135deg, #10534f, #1c8c76);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #5fe3c0;
    padding: 8px 14px;
    border-radius: 8px;
    margin: 14px 0;
}

.result-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    margin-top: 10px;
}

.info-card {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 14px;
    text-align: justify;
}

.footer-center { text-align: center; opacity: 0.85; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================
defaults = {
    "page": "home",
    "user_name": "",
    "user_campus": "",
    "history": [],
    "last_label": None,
    "last_conf": None,
    "last_model": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ==========================================================
# LOAD MODEL (cached)
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_pytorch_model(path: str):
    import torch
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    from tensorflow import keras
    return keras.models.load_model(path)


def predict_pytorch(model, pil_image: Image.Image, class_names):
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])
    x = transform(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def predict_keras(model, pil_image: Image.Image, class_names):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def annotate_image(pil_image: Image.Image, label: str, confidence: float) -> Image.Image:
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    text = f"{label} — {confidence * 100:.1f}%"

    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, annotated.width // 18))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 10

    draw.rectangle([(10, 10), (10 + tw + pad * 2, 10 + th + pad * 2)], fill=(15, 83, 79))
    draw.text((10 + pad, 10 + pad), text, fill=(95, 227, 192), font=font)
    return annotated


# ==========================================================
# HALAMAN: AWAL
# ==========================================================
def halaman_awal():
    st.markdown(f'<div class="main-title">🐾 DASHBOARD KLASIFIKASI KUCING vs ANJING</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Klasifikasi gambar dengan dua model AI (PyTorch & Keras)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-card">Aplikasi ini mendeteksi apakah gambar yang diunggah berisi kucing atau anjing, '
        'lengkap dengan tingkat keyakinan (confidence) model. Kamu bisa membandingkan hasil dari dua model berbeda '
        'yang sudah dilatih sebelumnya. Klik tombol di bawah untuk mulai.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Lanjut ke Registrasi →", use_container_width=False):
        st.session_state["page"] = "register"


# ==========================================================
# HALAMAN: REGISTRASI
# ==========================================================
def halaman_registrasi():
    st.markdown('<div class="main-title">📝 Registrasi Singkat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Isi data berikut sebelum masuk ke dashboard</div>', unsafe_allow_html=True)

    with st.form("form_reg"):
        name = st.text_input("Nama Lengkap", value=st.session_state["user_name"])
        campus = st.text_input("Asal Kampus", value=st.session_state["user_campus"] or KAMPUS_DEFAULT)
        submitted = st.form_submit_button("Masuk ke Dashboard →")

        if submitted:
            if not name.strip() or not campus.strip():
                st.warning("Mohon isi Nama dan Kampus terlebih dahulu.")
            else:
                st.session_state["user_name"] = name.strip()
                st.session_state["user_campus"] = campus.strip()
                st.session_state["page"] = "main"

    if st.button("⬅️ Kembali"):
        st.session_state["page"] = "home"


# ==========================================================
# HALAMAN: DASHBOARD UTAMA
# ==========================================================
def halaman_main():
    st.sidebar.markdown("### 🐾 Navigasi")
    mode = st.sidebar.radio(
        "Pilih model klasifikasi:",
        ["Model PyTorch (.pt)", "Model Keras (.h5)", "Bandingkan Keduanya"],
    )
    st.sidebar.info("📤 Unggah gambar kucing/anjing untuk mulai klasifikasi.")

    with st.sidebar.expander("⚙️ Kalibrasi urutan label"):
        st.caption(
            "Kalau hasil prediksi kebalik (kucing dibilang anjing atau sebaliknya), "
            "aktifkan ini untuk membalik urutan label tanpa edit kode."
        )
        swap_labels = st.checkbox("Balik urutan label (Kucing ⇄ Anjing)", value=False)
    class_names = list(reversed(CLASS_NAMES)) if swap_labels else CLASS_NAMES

    st.markdown('<div class="main-title">🔎 Dashboard Klasifikasi Gambar</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-card">Halo <b>{st.session_state["user_name"]}</b> dari '
        f'<b>{st.session_state["user_campus"]}</b> — pilih mode di sidebar lalu unggah gambar.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("📂 Unggah Gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(io.BytesIO(uploaded_file.read()))
        col1, col2 = st.columns(2)

        with col1:
            st.image(pil_image, caption="Gambar diunggah", use_container_width=True)

        results = []

        with st.spinner("🧠 Model sedang menganalisis gambar..."):
            if mode in ("Model PyTorch (.pt)", "Bandingkan Keduanya"):
                try:
                    model = load_pytorch_model(PYTORCH_MODEL_PATH)
                    label, conf, probs = predict_pytorch(model, pil_image, class_names)
                    results.append(("PyTorch (.pt)", label, conf, probs))
                except Exception as e:
                    st.error(f"Gagal menjalankan model PyTorch: {e}")

            if mode in ("Model Keras (.h5)", "Bandingkan Keduanya"):
                try:
                    model = load_keras_model(KERAS_MODEL_PATH)
                    label, conf, probs = predict_keras(model, pil_image, class_names)
                    results.append(("Keras (.h5)", label, conf, probs))
                except Exception as e:
                    st.error(f"Gagal menjalankan model Keras: {e}")

        if results:
            with col2:
                st.markdown('<div class="section-title">Hasil Klasifikasi</div>', unsafe_allow_html=True)
                for model_name, label, conf, probs in results:
                    st.markdown(
                        f'<div class="result-card"><b>{model_name}</b><br>'
                        f'Prediksi: <b>{label}</b><br>Confidence: <b>{conf*100:.2f}%</b></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(conf)

                    prob_dict = {class_names[i]: f"{probs[i]*100:.2f}%" for i in range(len(class_names))}
                    st.json(prob_dict)

            # gambar beranotasi dari hasil pertama
            top_label, top_conf = results[0][1], results[0][2]
            annotated = annotate_image(pil_image, top_label, top_conf)
            st.markdown('<div class="section-title">Gambar dengan Anotasi</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=False)

            # donut chart confidence
            benar = top_conf * 100
            salah = 100 - benar
            pie_df = pd.DataFrame({"Hasil": ["Confidence", "Sisa"], "Persentase": [benar, salah]})
            fig = px.pie(
                pie_df, values="Persentase", names="Hasil",
                color="Hasil",
                color_discrete_map={"Confidence": "#5fe3c0", "Sisa": "#0f3d3e"},
                hole=0.45,
            )
            fig.update_traces(textinfo="label+percent")
            fig.update_layout(
                width=300, height=300, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.markdown('<div class="section-title">Tingkat Keyakinan Model</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})

            # simpan ke riwayat
            st.session_state["history"].append({
                "waktu": datetime.datetime.now().strftime("%H:%M:%S"),
                "file": uploaded_file.name,
                "mode": mode,
                "prediksi": top_label,
                "confidence_%": round(top_conf * 100, 2),
            })
            st.session_state["last_label"] = top_label
            st.session_state["last_conf"] = top_conf * 100
            st.session_state["last_model"] = mode

            if st.button("💬 Berikan Feedback"):
                st.session_state["page"] = "feedback"
    else:
        st.info("👆 Unggah gambar terlebih dahulu untuk mulai klasifikasi.")

    if st.session_state["history"]:
        st.markdown('<div class="section-title">📜 Riwayat Prediksi Sesi Ini</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Registrasi"):
        st.session_state["page"] = "register"


# ==========================================================
# HALAMAN: FEEDBACK
# ==========================================================
def halaman_feedback():
    st.markdown('<div class="main-title">💬 Feedback Pengguna</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Terima kasih sudah mencoba — beri masukanmu di bawah</div>', unsafe_allow_html=True)

    with st.form("form_feedback"):
        rating = st.slider("Rating (1–5)", 1, 5, 5)
        comment = st.text_area("Komentar / Saran", height=120)
        submitted = st.form_submit_button("Kirim Feedback")

        if submitted:
            row = {
                "timestamp": datetime.datetime.now().isoformat(),
                "nama": st.session_state["user_name"],
                "kampus": st.session_state["user_campus"],
                "model": st.session_state["last_model"],
                "prediksi": st.session_state["last_label"],
                "confidence": st.session_state["last_conf"],
                "rating": rating,
                "komentar": comment,
            }
            df_row = pd.DataFrame([row])
            try:
                import os
                if not os.path.exists(FEEDBACK_CSV):
                    df_row.to_csv(FEEDBACK_CSV, index=False, mode="w", encoding="utf-8")
                else:
                    df_row.to_csv(FEEDBACK_CSV, index=False, mode="a", header=False, encoding="utf-8")
                st.success("✅ Feedback berhasil dikirim — terima kasih!")
            except Exception as e:
                st.error(f"Gagal menyimpan feedback: {e}")

    if st.button("⬅️ Kembali ke Dashboard"):
        st.session_state["page"] = "main"


# ==========================================================
# ROUTING
# ==========================================================
pages = {
    "home": halaman_awal,
    "register": halaman_registrasi,
    "main": halaman_main,
    "feedback": halaman_feedback,
}
pages.get(st.session_state["page"], halaman_awal)()

st.markdown("---")
st.markdown(
    f"<div class='footer-center'>Dibuat oleh <b>{NAMA_PEMBUAT}</b> | {NIM_PEMBUAT}<br>"
    "Proyek Praktikum Big Data — Klasifikasi Gambar Kucing vs Anjing</div>",
    unsafe_allow_html=True,
)
