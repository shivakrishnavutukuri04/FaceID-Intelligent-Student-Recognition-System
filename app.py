import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras_facenet import FaceNet
#from PIL import Image
#import os
import time

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🎓",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12151f 100%);
        border-right: 1px solid #2d2f3e;
    }

    /* Cards */
    .card {
        background: #1a1d2e;
        border: 1px solid #2d2f3e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #1e2235 100%);
        border: 1px solid #3d5afe;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .unknown-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #2a1a1a 100%);
        border: 1px solid #ff5252;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    /* Info rows */
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #2d2f3e;
    }
    .info-label { color: #8b8fa8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
    .info-value { color: #e0e4f0; font-size: 15px; font-weight: 500; }

    /* Badge */
    .badge {
        display: inline-block;
        background: #3d5afe22;
        border: 1px solid #3d5afe66;
        color: #82b1ff;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Title */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #82b1ff, #e040fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-sub { color: #8b8fa8; font-size: 1rem; margin-bottom: 32px; }

    /* Confidence bar */
    .conf-bar-bg {
        background: #2d2f3e;
        border-radius: 99px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 99px;
        background: linear-gradient(90deg, #3d5afe, #82b1ff);
        transition: width 0.4s ease;
    }

    /* Upload area */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #3d3f52 !important;
        border-radius: 12px !important;
        padding: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3d5afe, #651fff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Status pill */
    .status-ok  { color: #69f0ae; font-size: 13px; font-weight: 600; }
    .status-err { color: #ff5252; font-size: 13px; font-weight: 600; }

    /* Metric */
    .metric-box {
        background: #1a1d2e;
        border: 1px solid #2d2f3e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #82b1ff; }
    .metric-lbl { font-size: 12px; color: #8b8fa8; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar – configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    model_pkl = st.text_input(
        "Model (.pkl) path",
        value=r"C:\Users\vutuk\OneDrive\Desktop\FaceRecognitionProject\models\face_recognition_facenet_model.pkl",
        help="Absolute path to your joblib model file"
    )
    landmarker_path = st.text_input(
        "Face Landmarker (.task) path",
        value=r"C:\Users\vutuk\OneDrive\Desktop\FaceRecognitionProject\face_landmarker.task",
        help="Absolute path to MediaPipe face_landmarker.task"
    )
    students_csv = st.text_input(
        "Students CSV path",
        value=r"C:\Users\vutuk\OneDrive\Desktop\FaceRecognitionProject\metadata\students.csv",
        help="Absolute path to students.csv"
    )

    st.markdown("---")
    confidence_threshold = st.slider(
        "Confidence threshold (%)",
        min_value=0, max_value=100, value=60,
        help="Predictions below this are shown as Unknown"
    )

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
    <div style='color:#8b8fa8;font-size:13px;line-height:1.7'>
    • <b style='color:#e0e4f0'>Detection:</b> MediaPipe FaceLandmarker<br>
    • <b style='color:#e0e4f0'>Embedding:</b> FaceNet (512-d)<br>
    • <b style='color:#e0e4f0'>Classifier:</b> KNN + StandardScaler<br>
    • <b style='color:#e0e4f0'>Metadata:</b> CSV lookup
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Model loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(pkl_path, task_path):
    model_data = joblib.load(pkl_path)
    knn     = model_data["knn"]
    encoder = model_data["encoder"]
    scaler  = model_data["scaler"]

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    embedder = FaceNet()
    return knn, encoder, scaler, detector, embedder


@st.cache_data(show_spinner=False)
def load_students(csv_path):
    return pd.read_csv(csv_path)


# ─────────────────────────────────────────────
#  Core logic (mirrors your notebook)
# ─────────────────────────────────────────────
def detect_and_crop_face(image_bgr, detector):
    if image_bgr is None:
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    h, w = image_bgr.shape[:2]
    xs = [lm.x * w for lm in face]
    ys = [lm.y * h for lm in face]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    fw, fh = x_max - x_min, y_max - y_min
    pad_x  = int(fw * 0.15)
    pad_top    = int(fh * 0.30)
    pad_bottom = int(fh * 0.10)
    x_min = max(0, x_min - pad_x)   
    y_min = max(0, y_min - pad_top)
    x_max = min(w, x_max + pad_x) 
    y_max = min(h, y_max + pad_bottom)
    return image_bgr[y_min:y_max, x_min:x_max]


def extract_features(image_bgr, embedder):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (160, 160))
    return embedder.embeddings([rgb])[0]


def predict_identity(image_bgr, knn, encoder, scaler, detector, embedder, threshold):
    cropped = detect_and_crop_face(image_bgr, detector)
    if cropped is None:
        return None, None, None, None
    feats = extract_features(cropped, embedder).reshape(1, -1)
    feats = scaler.transform(feats)
    probs = knn.predict_proba(feats)[0]
    idx   = np.argmax(probs)
    conf  = probs[idx] * 100
    name  = encoder.inverse_transform([idx])[0]
    return name, conf, cropped, conf >= threshold


def get_student_info(name, students_df):
    row = students_df[students_df["Name"] == name]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {
        "Name":       r["Name"],
        "Role":       r["Role"],
        "Department": r["Department"],
        "Batch":      r.get("Batch", "—"),
        "Mobile":     r.get("mobile", "—"),
        "Gmail":      r.get("gmail", "—"),
        "Validity":   r.get("validity", "—"),
    }


# ─────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🎓 Face Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a photo to identify a student using FaceNet + KNN</div>', unsafe_allow_html=True)

# ── Load models ──────────────────────────────
models_ok = False
try:
    with st.spinner("Loading models…"):
        knn, encoder, scaler, detector, embedder = load_models(landmarker_path, landmarker_path)
        # fix arg order
        knn, encoder, scaler, detector, embedder = load_models.__wrapped__(model_pkl, landmarker_path)
except Exception:
    pass

# Proper load with correct args
try:
    knn, encoder, scaler, detector, embedder = load_models(model_pkl, landmarker_path)
    models_ok = True
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")

students_df = None
try:
    students_df = load_students(students_csv)
except Exception as e:
    st.warning(f"Could not load students.csv: {e}")

# ── Status bar ───────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    icon = "🟢" if models_ok else "🔴"
    st.markdown(f'<div class="metric-box"><div class="metric-val">{icon}</div><div class="metric-lbl">Models</div></div>', unsafe_allow_html=True)
with c2:
    icon = "🟢" if students_df is not None else "🔴"
    n    = len(students_df) if students_df is not None else 0
    st.markdown(f'<div class="metric-box"><div class="metric-val">{icon} {n}</div><div class="metric-lbl">Students in DB</div></div>', unsafe_allow_html=True)
with c3:
    nc = len(encoder.classes_) if models_ok else 0
    st.markdown(f'<div class="metric-box"><div class="metric-val">{nc}</div><div class="metric-lbl">Trained Classes</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload & Identify ────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Choose a photo (JPG / PNG / JPEG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded image", use_container_width=True)

        run_btn = st.button("🔍  Identify Person", disabled=not models_ok)
    else:
        st.markdown("""
        <div style='text-align:center;padding:40px 0;color:#8b8fa8'>
            <div style='font-size:3rem'>📸</div>
            <div style='margin-top:8px'>Drop an image to get started</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("#### 🪪 Recognition Result")

    if uploaded and run_btn:
        with st.spinner("Detecting face & running recognition…"):
            t0 = time.time()
            name, conf, cropped, passed = predict_identity(
                img_bgr, knn, encoder, scaler, detector, embedder, confidence_threshold
            )
            elapsed = time.time() - t0

        if name is None:
            st.markdown("""
            <div class="unknown-card">
                <div style='font-size:3rem'>😶</div>
                <div style='font-size:1.3rem;font-weight:700;color:#ff5252;margin-top:12px'>No Face Detected</div>
                <div style='color:#8b8fa8;margin-top:8px;font-size:14px'>
                    Make sure the image contains a clearly visible face.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not passed:
            st.markdown(f"""
            <div class="unknown-card">
                <div style='font-size:3rem'>❓</div>
                <div style='font-size:1.3rem;font-weight:700;color:#ff5252;margin-top:12px'>Unknown Person</div>
                <div style='color:#8b8fa8;margin-top:8px;font-size:14px'>
                    Best match: <b style='color:#e0e4f0'>{name}</b> — but confidence too low ({conf:.1f}%)
                </div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf}%;background:linear-gradient(90deg,#ff5252,#ff8a80)"></div></div>
            </div>
            """, unsafe_allow_html=True)

        else:
            info = get_student_info(name, students_df) if students_df is not None else None

            display_name = info["Name"] if info else name

            st.markdown(f"""
            <div class="result-card">
                <div style='font-size:3rem'>✅</div>
                <div style='font-size:1.5rem;font-weight:700;color:#82b1ff;margin-top:12px'>{display_name}</div>
                <div style='margin-top:10px'><span class="badge">Identified</span></div>
                <div style='margin-top:14px;font-size:13px;color:#8b8fa8'>Confidence</div>
                <div style='font-size:1.8rem;font-weight:700;color:#69f0ae'>{conf:.1f}%</div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf}%"></div></div>
                <div style='margin-top:10px;color:#8b8fa8;font-size:12px'>Inference time: {elapsed*1000:.0f} ms</div>
            </div>
            """, unsafe_allow_html=True)

            # Cropped face
            if cropped is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                face_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_b:
                    st.image(face_rgb, caption="Detected face", width=130)

            # Student details
            if info:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📋 Student Details**")
                rows = [
                    ("Role",       info["Role"]),
                    ("Department", info["Department"]),
                    ("Batch",      info["Batch"]),
                    ("Mobile",     info["Mobile"]),
                    ("Gmail",      info["Gmail"]),
                    ("Valid Until",info["Validity"]),
                ]
                for label, value in rows:
                    st.markdown(f"""
                    <div class="info-row">
                        <span class="info-label">{label}</span>
                        <span class="info-value">{value}</span>
                    </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#8b8fa8'>
            <div style='font-size:3rem'>🔍</div>
            <div style='margin-top:12px'>Results will appear here after identification</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Students table (expandable)
# ─────────────────────────────────────────────
if students_df is not None:
    st.markdown("---")
    with st.expander("👥 All Registered Students"):
        st.dataframe(
            students_df,
            use_container_width=True,
            hide_index=True,
        )
