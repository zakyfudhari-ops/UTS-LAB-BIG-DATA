"""
Pemindai Spesies — Klasifikasi Gambar Kucing vs Anjing
Dikembangkan dari struktur referensi: alur halaman
serupa (mulai -> daftar -> pindai), tapi desain, komponen, dan kode ditulis
ulang total. Konsep visual: "field scanner" — foto yang diunggah
ditampilkan seperti slide foto di meja lab lengkap dengan bracket fokus
ala viewfinder kamera, dan hasil tiap model ditampilkan sebagai catatan
pindaian dengan dial lingkaran, bukan kartu seragam bergaya SaaS.

Fitur:
- Dua model (.pt YOLO & .h5 CNN) yang bisa dijalankan sendiri atau dibandingkan
- Anotasi label + confidence langsung di atas foto
- Riwayat pindaian per sesi

Struktur folder:
├── dashboard.py
├── requirements.txt
├── packages.txt
└── model/
    ├── KucingAnjing.pt
    └── AnjingKucing.h5
"""

import io
import os
import base64
import datetime
from zoneinfo import ZoneInfo

WIB = ZoneInfo("Asia/Jakarta")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ==========================================================
# KONFIGURASI
# ==========================================================
CLASS_NAMES = ["Anjing", "Kucing"]      # urutan alfabetis (default Keras flow_from_directory)
IMG_SIZE = (224, 224)
PYTORCH_MODEL_PATH = "model/KucingAnjing.pt"
KERAS_MODEL_PATH = "model/AnjingKucing.h5"

NAMA_PEMBUAT = "M. Zaky Al Fudhari"      # ganti sesuai nama kamu
NIM_PEMBUAT = "2308108010064"         # ganti sesuai NIM kamu
KAMPUS_DEFAULT = "Universitas Syiah Kuala"

STEP_KEYS = ["home", "register", "main"]
STEP_LABELS = ["Mulai", "Daftar", "Pindai"]

st.set_page_config(
    page_title="Pemindai Spesies",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# TEMA — "Field Scanner": charcoal-moss + amber IR-glow
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --bg:#0D1711;
    --surface:#1E2F25;
    --surface-2:#28402F;
    --line: rgba(237,232,221,0.12);
    --accent:#F2994A;
    --accent-soft: rgba(242,153,74,0.20);
    --sage:#8FB996;
    --coral:#E08A72;
    --text:#F3EFE4;
    --text-dim:#A9BBAD;
}

html, body, [class*="css"] { font-family:'IBM Plex Sans', sans-serif; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 520px at 88% 8%, rgba(242,153,74,0.10), transparent 60%),
        radial-gradient(760px 520px at 6% 96%, rgba(143,185,150,0.09), transparent 60%),
        var(--bg);
    color: var(--text);
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
.block-container {
    background: var(--surface);
    border-radius: 26px;
    margin: 28px auto !important;
    padding: 36px 42px 30px 42px !important;
    max-width: 1180px !important;
    box-shadow: 0 30px 70px rgba(0,0,0,0.38);
    border: 1px solid rgba(255,255,255,0.05);
    position: relative;
    clip-path: inset(0 round 26px);
}
.block-container::before{
    content:""; position:absolute; top:-80px; right:-80px; width:260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, rgba(242,153,74,0.14), transparent 70%); pointer-events:none;
}
.block-container::after{
    content:""; position:absolute; bottom:-100px; left:-60px; width:260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, rgba(143,185,150,0.10), transparent 70%); pointer-events:none;
}

/* ---------- header & step track ---------- */
.header-bar{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:8px; position:relative; z-index:1; }
.wordmark{ font-family:'Big Shoulders Display'; font-weight:800; font-size:21px; letter-spacing:.01em; color:var(--text); }
.step-track{ display:flex; align-items:center; }
.step{ display:flex; align-items:center; }
.step .step-dot{ width:8px; height:8px; border-radius:50%; background:var(--line); display:inline-block; flex:none; }
.step.done .step-dot{ background:var(--sage); }
.step.active .step-dot{ background:var(--accent); box-shadow:0 0 0 4px var(--accent-soft); }
.step .step-label{ font-size:12px; color:var(--text-dim); margin:0 10px 0 7px; white-space:nowrap; }
.step.active .step-label{ color:var(--text); font-weight:600; }
.step:not(:last-child) .step-label{ position:relative; padding-right:18px; }
.step:not(:last-child) .step-label::after{ content:""; position:absolute; right:0; top:50%; width:14px; height:1px; background:var(--line); }

@keyframes heroIn { 0%{ opacity:0; transform:translateY(10px);} 100%{ opacity:1; transform:translateY(0);} }
.fade-in{ animation: heroIn .55s ease both; }

/* ---------- hero (home) ---------- */
.hero-title{ font-family:'Big Shoulders Display'; font-weight:800; font-size:clamp(32px,4.4vw,50px); line-height:1.04; color:var(--text); margin:22px 0 10px 0; max-width:720px; }
.hero-sub{ font-size:15px; color:var(--text-dim); max-width:560px; line-height:1.65; margin-bottom:6px; }
.hero-art{ background:var(--surface-2); border-radius:20px; border:1px solid var(--line); padding:10px; margin-top:8px; overflow:hidden; }
.hero-art img{ width:100%; height:auto; display:block; border-radius:12px; }

/* ---------- intro / section heading ---------- */
.intro{ border-left:3px solid var(--accent); padding:2px 0 2px 16px; margin:8px 0 22px 0; color:var(--text-dim); font-size:14.5px; line-height:1.65; max-width:680px; }
.section-heading{ font-family:'Big Shoulders Display'; font-weight:700; font-size:19px; color:var(--text); margin:26px 0 10px 0; }

/* ---------- specimen frame (foto) ---------- */
.specimen{ position:relative; background:var(--surface-2); border:1px solid var(--line); border-radius:16px; padding:14px; overflow:hidden; }
.specimen img{ width:100%; display:block; border-radius:8px; }
.specimen .corner{ position:absolute; width:16px; height:16px; border-color:var(--accent); }
.specimen .corner.tl{ top:6px; left:6px; border-top:2px solid; border-left:2px solid; }
.specimen .corner.tr{ top:6px; right:6px; border-top:2px solid; border-right:2px solid; }
.specimen .corner.bl{ bottom:6px; left:6px; border-bottom:2px solid; border-left:2px solid; }
.specimen .corner.br{ bottom:6px; right:6px; border-bottom:2px solid; border-right:2px solid; }
.specimen-caption{ font-family:'IBM Plex Mono'; font-size:11px; color:var(--text-dim); margin:8px 0 4px 2px; }

/* ---------- field log (hasil per model) ---------- */
.log{ border-top:1px solid var(--line); margin-top:4px; }
.log-row{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 2px; border-bottom:1px solid var(--line); }
.row-main{ display:flex; align-items:center; gap:14px; min-width:0; }
.row-icon{ font-size:21px; width:30px; text-align:center; flex:none; }
.row-model{ font-size:12px; color:var(--text-dim); margin-bottom:2px; }
.row-label{ font-family:'Big Shoulders Display'; font-weight:700; font-size:25px; color:var(--text); line-height:1; }
.row-note{ font-size:12px; color:var(--coral); margin-top:5px; }

.ring{ width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.ring-inner{ width:46px; height:46px; border-radius:50%; background:var(--surface); display:flex; align-items:center; justify-content:center; }
.ring-value{ font-family:'IBM Plex Mono'; font-size:12.5px; font-weight:600; color:var(--text); }

.prob-breakdown{ margin:4px 0 4px 44px; }
.prob-row{ display:grid; grid-template-columns:64px 1fr 44px; align-items:center; gap:10px; margin:5px 0; }
.prob-name{ font-size:12px; color:var(--text-dim); }
.prob-track{ height:5px; background:var(--surface-2); border-radius:3px; overflow:hidden; }
.prob-fill{ height:100%; background:var(--accent); }
.prob-pct{ font-family:'IBM Plex Mono'; font-size:11px; color:var(--text-dim); text-align:right; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:var(--bg); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child{ padding-top:1.5rem; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color:var(--text) !important; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color:var(--text-dim) !important; }

/* ---------- widgets ---------- */
[data-testid="stFileUploaderDropzone"]{ background:var(--surface-2); border:1px dashed var(--line); border-radius:14px; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea{
    background:var(--surface-2) !important; color:var(--text) !important; border:1px solid var(--line) !important; border-radius:10px !important;
}
[data-testid="stAlert"]{ background:var(--surface-2); border:1px solid var(--line); border-radius:12px; color:var(--text); }

[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button{
    font-weight:600; font-size:14px; border-radius:10px; padding:0.6rem 1.4rem; border:none;
    background:var(--accent); color:#241607; transition:filter .15s ease, transform .15s ease;
}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover{ filter:brightness(1.08); transform:translateY(-1px); }
[data-testid="stButton"] button[kind="secondary"]{ background:transparent; border:1px solid var(--line); color:var(--text-dim); }
[data-testid="stButton"] button[kind="secondary"]:hover{ border-color:var(--accent); color:var(--text); filter:none; }

[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }

.footer{ text-align:left; color:var(--text-dim); font-size:12.5px; line-height:1.7; border-top:1px solid var(--line); padding-top:16px; margin-top:34px; }
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
    "play_login_sound": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ==========================================================
# LOAD MODEL (cached)
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_pytorch_model(path: str):
    # Model .pt ini dilatih/disimpan lewat framework Ultralytics (YOLO),
    # jadi dimuat pakai YOLO(), bukan torch.load() biasa.
    from ultralytics import YOLO
    return YOLO(path)


@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    from tensorflow import keras
    return keras.models.load_model(path)


def predict_pytorch(model, pil_image: Image.Image, class_names):
    # Model YOLO ini bisa berupa model klasifikasi (punya r.probs) ATAU
    # model deteksi objek (punya r.boxes, tidak punya r.probs). Tangani keduanya.
    # Threshold confidence diturunkan (default YOLO 0.25) karena model ini
    # dilatih dari nol dengan data terbatas, jadi confidence-nya cenderung rendah.
    results = model.predict(np.array(pil_image.convert("RGB")), conf=0.10, verbose=False)
    r = results[0]
    names = [model.names[i] for i in range(len(model.names))]

    if r.probs is not None:
        probs_obj = r.probs
        idx = int(probs_obj.top1)
        conf = float(probs_obj.top1conf)
        probs = probs_obj.data.cpu().numpy()
        label = names[idx]
        return label, conf, probs, names

    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        raise ValueError(
            "Tidak ada objek terdeteksi walau threshold sudah diturunkan ke 10%. "
            "Model ini kemungkinan butuh dilatih ulang dengan data/epoch lebih banyak."
        )

    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    best_idx = int(np.argmax(confs))
    best_cls = int(clss[best_idx])
    conf = float(confs[best_idx])
    label = names[best_cls]

    probs = np.zeros(len(names))
    probs[best_cls] = conf
    return label, conf, probs, names


def predict_keras(model, pil_image: Image.Image, class_names):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs, class_names


def annotate_image(pil_image: Image.Image, label: str, confidence: float) -> Image.Image:
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    text = f"{label}  {confidence * 100:.1f}%"

    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, annotated.width // 18))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 10

    draw.rectangle([(10, 10), (10 + tw + pad * 2, 10 + th + pad * 2)], fill=(27, 43, 35))
    draw.text((10 + pad, 10 + pad), text, fill=(227, 162, 59), font=font)
    return annotated


# ==========================================================
# HELPER TAMPILAN
# ==========================================================
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


MEOW_SOUND_URL = "https://www.myinstants.com/media/sounds/m-e-o-w.mp3"


def play_meow_sound():
    st.markdown(
        f'<audio autoplay style="display:none">'
        f'<source src="{MEOW_SOUND_URL}" type="audio/mpeg"></audio>',
        unsafe_allow_html=True,
    )


def render_header():
    current = STEP_KEYS.index(st.session_state["page"]) if st.session_state["page"] in STEP_KEYS else 0
    dots = ""
    for i, label in enumerate(STEP_LABELS):
        state = "done" if i < current else ("active" if i == current else "")
        dots += f'<div class="step {state}"><span class="step-dot"></span><span class="step-label">{label}</span></div>'
    st.markdown(
        f'<div class="header-bar"><div class="wordmark">🎞️ Pemindai Spesies</div>'
        f'<div class="step-track">{dots}</div></div>',
        unsafe_allow_html=True,
    )


def specimen_frame(img: Image.Image, caption: str):
    b64 = pil_to_base64(img)
    st.markdown(
        f'<div class="specimen"><div class="corner tl"></div><div class="corner tr"></div>'
        f'<div class="corner bl"></div><div class="corner br"></div>'
        f'<img src="data:image/png;base64,{b64}" /></div>'
        f'<div class="specimen-caption">{caption}</div>',
        unsafe_allow_html=True,
    )


def confidence_ring_html(pct: float, color: str) -> str:
    angle = max(0.0, min(360.0, pct * 3.6))
    return (
        f'<div class="ring" style="background: conic-gradient({color} {angle}deg, var(--surface-2) {angle}deg 360deg);">'
        f'<div class="ring-inner"><span class="ring-value">{pct:.0f}%</span></div></div>'
    )


def prob_breakdown_html(names, probs) -> str:
    order = sorted(range(len(names)), key=lambda i: -probs[i])
    rows = ""
    for i in order:
        pct = probs[i] * 100
        rows += (
            f'<div class="prob-row"><span class="prob-name">{names[i]}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="prob-pct">{pct:.1f}%</span></div>'
        )
    return f'<div class="prob-breakdown">{rows}</div>'


def render_result_row(icon: str, model_name: str, label: str, conf: float, names, probs):
    pct = conf * 100
    color = "var(--accent)" if pct >= 50 else "var(--coral)"
    ring = confidence_ring_html(pct, color)
    note = '<div class="row-note">Keyakinan rendah — hasil ini kurang bisa diandalkan.</div>' if pct < 50 else ""
    st.markdown(
        f'<div class="log-row"><div class="row-main"><div class="row-icon">{icon}</div>'
        f'<div class="row-text"><div class="row-model">{model_name}</div>'
        f'<div class="row-label">{label}</div>{note}</div></div>{ring}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(prob_breakdown_html(names, probs), unsafe_allow_html=True)


# ==========================================================
# HALAMAN: AWAL
# ==========================================================
HERO_IMAGE_PATH = "assets/hero-pets.png"


def halaman_awal():
    render_header()
    col_text, col_art = st.columns([0.56, 0.44])
    with col_text:
        st.markdown(
            '<div class="fade-in">'
            '<div class="hero-title">Satu foto, dua model,<br>satu jawaban: kucing atau anjing.</div>'
            '<div class="hero-sub">Unggah fotonya. Model YOLO dan model CNN sama-sama memeriksa gambar itu, '
            'lalu masing-masing melaporkan hasilnya secara terpisah.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Isi data diri", type="primary"):
            st.session_state["page"] = "register"
    with col_art:
        if os.path.exists(HERO_IMAGE_PATH):
            b64 = base64.b64encode(open(HERO_IMAGE_PATH, "rb").read()).decode()
            st.markdown(
                f'<div class="hero-art fade-in"><img src="data:image/png;base64,{b64}" /></div>',
                unsafe_allow_html=True,
            )


# ==========================================================
# HALAMAN: REGISTRASI
# ==========================================================
def halaman_registrasi():
    render_header()
    st.markdown(
        '<div class="intro">Sebelum memindai, isi identitasmu dulu. '
        'Data ini ikut tercatat pada setiap hasil pindaian di sesi ini.</div>',
        unsafe_allow_html=True,
    )

    with st.form("form_reg"):
        name = st.text_input("Nama lengkap", value=st.session_state["user_name"])
        campus = st.text_input("Asal kampus", value=st.session_state["user_campus"] or KAMPUS_DEFAULT)
        submitted = st.form_submit_button("Masuk ke pemindai", type="primary")

        if submitted:
            if not name.strip() or not campus.strip():
                st.warning("Nama dan kampus belum diisi.")
            else:
                st.session_state["user_name"] = name.strip()
                st.session_state["user_campus"] = campus.strip()
                st.session_state["page"] = "main"
                st.session_state["play_login_sound"] = True

    if st.button("Kembali", type="secondary"):
        st.session_state["page"] = "home"


# ==========================================================
# HALAMAN: PINDAI (dashboard utama)
# ==========================================================
def halaman_main():
    render_header()

    if st.session_state.get("play_login_sound"):
        play_meow_sound()
        st.session_state["play_login_sound"] = False

    st.sidebar.markdown("**Model**")
    st.sidebar.caption(
        "Setiap foto dipindai oleh dua model sekaligus: YOLO menandai posisi objek sebelum "
        "memberi label, CNN langsung mengklasifikasikan seluruh gambar."
    )

    # Urutan label dibalik otomatis dari default alfabetis — hasil pengujian manual
    # menunjukkan urutan terbalik ini yang cocok dengan output model CNN saat ini.
    class_names = list(reversed(CLASS_NAMES))

    st.markdown(
        f'<div class="intro">Halo {st.session_state["user_name"]} dari {st.session_state["user_campus"]}. '
        f'Unggah satu foto, lalu baca catatan pindaian di sebelah kanan.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Foto untuk dipindai", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(io.BytesIO(uploaded_file.read()))
        col1, col2 = st.columns([0.46, 0.54])

        with col1:
            specimen_frame(pil_image, f"{uploaded_file.name} · {uploaded_file.size / 1024:.1f} KB")

        results = []
        with st.spinner("Memindai gambar..."):
            try:
                model = load_pytorch_model(PYTORCH_MODEL_PATH)
                label, conf, probs, names = predict_pytorch(model, pil_image, class_names)
                results.append(("📷", "YOLO (deteksi)", label, conf, probs, names))
            except Exception as e:
                st.error(f"YOLO gagal memindai: {e}")

            try:
                model = load_keras_model(KERAS_MODEL_PATH)
                label, conf, probs, names = predict_keras(model, pil_image, class_names)
                results.append(("🧪", "CNN (klasifikasi)", label, conf, probs, names))
            except Exception as e:
                st.error(f"CNN gagal memindai: {e}")

        if results:
            with col2:
                st.markdown('<div class="section-heading">Catatan pindaian</div>', unsafe_allow_html=True)
                st.markdown('<div class="log">', unsafe_allow_html=True)
                for icon, model_name, label, conf, probs, names in results:
                    render_result_row(icon, model_name, label, conf, names, probs)
                st.markdown('</div>', unsafe_allow_html=True)

            top_icon, top_model, top_label, top_conf = results[0][0], results[0][1], results[0][2], results[0][3]
            annotated = annotate_image(pil_image, top_label, top_conf)
            st.markdown('<div class="section-heading">Foto dengan anotasi</div>', unsafe_allow_html=True)
            acol, _ = st.columns([0.46, 0.54])
            with acol:
                specimen_frame(annotated, f"Ditandai berdasarkan {top_model}")

            st.session_state["history"].append({
                "waktu": datetime.datetime.now(WIB).strftime("%H:%M:%S"),
                "file": uploaded_file.name,
                "prediksi": top_label,
                "confidence_%": round(top_conf * 100, 2),
            })
    else:
        st.info("Belum ada foto. Unggah satu untuk memulai pindaian.")

    if st.session_state["history"]:
        st.markdown('<div class="section-heading">Riwayat sesi ini</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)

    st.write("")
    if st.button("Kembali ke pendaftaran", type="secondary"):
        st.session_state["page"] = "register"


# ==========================================================
# ROUTING
# ==========================================================
pages = {
    "home": halaman_awal,
    "register": halaman_registrasi,
    "main": halaman_main,
}
pages.get(st.session_state["page"], halaman_awal)()

st.markdown(
    f'<div class="footer">{NAMA_PEMBUAT}<br>{NIM_PEMBUAT}<br>Proyek UTS Big Data, pemindai kucing dan anjing.</div>',
    unsafe_allow_html=True,
)
