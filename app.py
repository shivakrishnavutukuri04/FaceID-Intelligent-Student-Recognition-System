import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras_facenet import FaceNet
from pathlib import Path
import time


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NEXUS • Intelligent Face Recognition",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "face_recognition_facenet_model.pkl"
LANDMARKER_PATH = BASE_DIR / "face_landmarker.task"
STUDENTS_PATH = BASE_DIR / "students.csv"


# ============================================================
# PROFESSIONAL DESIGN SYSTEM
# Uses st.html instead of st.markdown for UI HTML.
# This avoids Streamlit Markdown interpreting HTML as code.
# ============================================================

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #070910;
    --panel: #101522;
    --panel2: #0c111d;
    --line: rgba(255,255,255,.08);
    --muted: #7c879f;
    --text: #f4f7ff;
    --blue: #6c7cff;
    --violet: #9b6cff;
    --cyan: #5ee7ff;
    --green: #63f5ae;
    --red: #ff667d;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 7% 2%, rgba(91,105,255,.14), transparent 25%),
        radial-gradient(circle at 95% 6%, rgba(170,83,255,.12), transparent 25%),
        radial-gradient(circle at 50% 95%, rgba(49,211,255,.06), transparent 30%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0c111d,#070910);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

h1,h2,h3,h4 {
    font-family: "Space Grotesk", sans-serif !important;
}

/* ---------- Header ---------- */

.nx-header {
    position: relative;
    padding: 20px 0 34px;
}

.nx-topline {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #aeb9ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}

.nx-live {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 16px rgba(99,245,174,.9);
}

.nx-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(2.8rem,5vw,5.2rem);
    line-height: .94;
    font-weight: 700;
    letter-spacing: -2px;
    margin-top: 18px;
    max-width: 850px;
    color: #fff;
}

.nx-title span {
    background: linear-gradient(100deg,#fff 15%,#aebaff 48%,#c28cff 88%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nx-copy {
    max-width: 680px;
    margin-top: 18px;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.7;
}

.nx-chip {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 11px;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: #a9b3ca;
    background: rgba(255,255,255,.025);
    font-size: 10px;
    letter-spacing: 1px;
}

/* ---------- Stats ---------- */

.nx-stat {
    position: relative;
    min-height: 122px;
    padding: 20px;
    border-radius: 20px;
    overflow: hidden;
    background: linear-gradient(145deg,rgba(19,25,41,.94),rgba(10,14,24,.94));
    border: 1px solid var(--line);
    box-shadow: 0 18px 50px rgba(0,0,0,.20);
}

.nx-stat:after {
    content: "";
    position: absolute;
    width: 110px;
    height: 110px;
    right: -55px;
    top: -55px;
    border-radius: 50%;
    background: rgba(104,120,255,.12);
    filter: blur(4px);
}

.nx-stat-icon {
    font-size: 18px;
    color: #a8b4ff;
}

.nx-stat-value {
    margin-top: 10px;
    font-family: "Space Grotesk", sans-serif;
    font-size: 25px;
    font-weight: 700;
    color: #f3f6ff;
}

.nx-stat-value.good { color: var(--green); }
.nx-stat-value.bad { color: var(--red); }

.nx-stat-label {
    margin-top: 3px;
    color: #68738e;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* ---------- Panels ---------- */

.nx-panel {
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 24px;
    background:
        linear-gradient(145deg,rgba(17,23,38,.96),rgba(9,13,23,.96));
    box-shadow:
        0 25px 70px rgba(0,0,0,.20),
        inset 0 1px 0 rgba(255,255,255,.025);
}

.nx-panel-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 15px;
    margin-bottom: 18px;
}

.nx-panel-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #f2f5ff;
}

.nx-panel-sub {
    color: #69748e;
    font-size: 12px;
    margin-top: 4px;
    line-height: 1.5;
}

.nx-panel-tag {
    flex-shrink: 0;
    padding: 6px 9px;
    border-radius: 8px;
    border: 1px solid rgba(99,245,174,.16);
    background: rgba(99,245,174,.06);
    color: var(--green);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ---------- Upload ---------- */

div[data-testid="stFileUploader"] {
    border: 1px dashed rgba(112,129,255,.42) !important;
    border-radius: 18px !important;
    background: rgba(7,10,18,.55) !important;
    padding: 10px !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(151,161,255,.85) !important;
    background: rgba(30,35,60,.35) !important;
}

.nx-drop {
    min-height: 220px;
    border: 1px dashed rgba(112,129,255,.24);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background:
        radial-gradient(circle at 50% 25%,rgba(104,120,255,.08),transparent 40%),
        rgba(5,8,15,.35);
}

.nx-drop-icon {
    font-size: 46px;
    color: #8d9aff;
    opacity: .85;
}

.nx-drop-title {
    margin-top: 12px;
    color: #bfc7dd;
    font-family: "Space Grotesk", sans-serif;
    font-size: 15px;
    font-weight: 600;
}

.nx-drop-sub {
    margin-top: 6px;
    color: #606b84;
    font-size: 11px;
}

/* ---------- Button ---------- */

.stButton > button {
    width: 100%;
    min-height: 52px;
    border: 0;
    border-radius: 14px;
    background: linear-gradient(105deg,#586bff,#795dff 55%,#a75ce7);
    color: #fff;
    font-weight: 700;
    letter-spacing: .4px;
    box-shadow: 0 12px 32px rgba(91,87,255,.25);
    transition: .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 17px 40px rgba(91,87,255,.38);
}

/* ---------- Scan ---------- */

.nx-scan {
    position: relative;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    border-radius: 20px;
    border: 1px solid rgba(105,125,255,.23);
    background: #090e1a;
}

.nx-scan-frame {
    width: 145px;
    height: 170px;
    border: 1px solid rgba(112,132,255,.45);
    border-radius: 28px;
    position: relative;
}

.nx-scan-frame:before,
.nx-scan-frame:after {
    content: "";
    position: absolute;
    inset: 15px;
    border: 1px solid rgba(94,231,255,.22);
    border-radius: 20px;
}

.nx-scan-line {
    position: absolute;
    left: 12%;
    right: 12%;
    height: 2px;
    top: 0;
    background: linear-gradient(90deg,transparent,#68dfff,#b77cff,transparent);
    box-shadow: 0 0 18px rgba(104,223,255,.9);
    animation: nxscan 1.8s infinite ease-in-out;
}

@keyframes nxscan {
    0% { top: 8%; opacity: 0; }
    15% { opacity: 1; }
    85% { opacity: 1; }
    100% { top: 92%; opacity: 0; }
}

.nx-scan-label {
    margin-top: 17px;
    color: #aeb9ff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.3px;
}

.nx-scan-sub {
    margin-top: 6px;
    color: #626d86;
    font-size: 10px;
}

/* ---------- Identity result ---------- */

.nx-verified {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 26px;
    background:
        radial-gradient(circle at 88% 5%,rgba(109,99,255,.20),transparent 32%),
        linear-gradient(145deg,#171d33,#0c111e);
    border: 1px solid rgba(112,128,255,.25);
}

.nx-verified:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(#5effad,#6f79ff,#c27cff);
}

.nx-eyebrow {
    color: #7d88a3;
    font-size: 9px;
    letter-spacing: 1.8px;
    font-weight: 700;
}

.nx-name {
    margin-top: 6px;
    font-family: "Space Grotesk", sans-serif;
    color: #fff;
    font-size: 29px;
    font-weight: 700;
}

.nx-verified-pill {
    display: inline-block;
    margin-top: 10px;
    padding: 6px 10px;
    border-radius: 999px;
    color: var(--green);
    background: rgba(99,245,174,.07);
    border: 1px solid rgba(99,245,174,.20);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
}

.nx-confidence {
    margin-top: 23px;
}

.nx-confidence-top {
    display: flex;
    justify-content: space-between;
    color: #737e98;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.nx-confidence-top b {
    color: var(--green);
}

.nx-track {
    height: 7px;
    margin-top: 8px;
    border-radius: 999px;
    background: rgba(255,255,255,.06);
    overflow: hidden;
}

.nx-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#596cff,#8b69ff,#64f5ad);
    box-shadow: 0 0 14px rgba(100,245,173,.22);
}

.nx-time {
    margin-top: 10px;
    color: #59647d;
    font-size: 10px;
}

/* ---------- Details ---------- */

.nx-details {
    margin-top: 15px;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: rgba(6,9,16,.48);
}

.nx-details-title {
    font-family: "Space Grotesk", sans-serif;
    color: #e8ecfb;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 8px;
}

.nx-row {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,.045);
}

.nx-row:last-child { border-bottom: 0; }

.nx-key {
    color: #68738c;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.nx-value {
    color: #e4e9fa;
    font-size: 11px;
    font-weight: 600;
    text-align: right;
}

/* ---------- Empty / unknown ---------- */

.nx-empty {
    min-height: 285px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #657089;
}

.nx-empty-icon {
    font-size: 42px;
    color: #7886b9;
    opacity: .75;
}

.nx-empty-title {
    margin-top: 14px;
    color: #aab3c9;
    font-family: "Space Grotesk", sans-serif;
    font-size: 14px;
    font-weight: 600;
}

.nx-empty-sub {
    margin-top: 6px;
    color: #606b84;
    font-size: 10px;
}

.nx-unknown {
    padding: 42px 25px;
    text-align: center;
    border-radius: 22px;
    background: radial-gradient(circle at 50% 0%,rgba(255,70,100,.11),transparent 45%),#1c0e17;
    border: 1px solid rgba(255,90,115,.24);
}

.nx-unknown-title {
    color: #ff7285;
    font-family: "Space Grotesk", sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
}

.nx-unknown-sub {
    color: #858da4;
    font-size: 11px;
    margin-top: 7px;
}

/* ---------- Sidebar ---------- */

.nx-side-brand {
    font-family: "Space Grotesk", sans-serif;
    color: #fff;
    font-size: 20px;
    font-weight: 700;
}

.nx-side-sub {
    color: #66718b;
    font-size: 9px;
    letter-spacing: 1.1px;
    margin-top: 3px;
}

.nx-pipeline {
    color: #7e89a3;
    font-size: 11px;
    line-height: 2.1;
}

.nx-pipeline b {
    color: #dbe1ff;
}

.nx-side-note {
    color: #59647c;
    font-size: 9px;
    line-height: 1.6;
}

/* Footer */
.nx-footer {
    text-align: center;
    color: #4d576e;
    font-size: 9px;
    letter-spacing: .8px;
    margin-top: 42px;
}
</style>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.html("""
    <div class="nx-side-brand">◉ NEXUS FACE ID</div>
    <div class="nx-side-sub">INTELLIGENT STUDENT RECOGNITION</div>
    """)

    st.markdown("")
    st.markdown("### ⚙️ Recognition Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        help="Predictions below this value are classified as Unknown.",
    )

    st.markdown("---")
    st.markdown("### 🧠 AI Pipeline")

    st.html("""
    <div class="nx-pipeline">
        <b>01</b> MediaPipe FaceLandmarker<br>
        <b>02</b> FaceNet 512-D Embedding<br>
        <b>03</b> StandardScaler<br>
        <b>04</b> KNN Classification<br>
        <b>05</b> Student Metadata Lookup
    </div>
    """)

    st.markdown("---")

    st.html("""
    <div class="nx-side-note">
        NEXUS FACE ID<br>
        Intelligent face recognition for student identity verification.<br><br>
        Local project files are resolved automatically from the application folder.
    </div>
    """)


# ============================================================
# MODEL FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models(pkl_path, task_path):
    model_data = joblib.load(pkl_path)

    knn = model_data["knn"]
    encoder = model_data["encoder"]
    scaler = model_data["scaler"]

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


def detect_and_crop_face(image_bgr, detector):
    if image_bgr is None:
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]
    h, w = image_bgr.shape[:2]

    xs = [lm.x * w for lm in face]
    ys = [lm.y * h for lm in face]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    fw = x_max - x_min
    fh = y_max - y_min

    pad_x = int(fw * 0.15)
    pad_top = int(fh * 0.30)
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


def predict_identity(
    image_bgr,
    knn,
    encoder,
    scaler,
    detector,
    embedder,
    threshold,
):
    cropped = detect_and_crop_face(image_bgr, detector)

    if cropped is None:
        return None, None, None, None

    features = extract_features(cropped, embedder).reshape(1, -1)
    features = scaler.transform(features)

    probabilities = knn.predict_proba(features)[0]

    idx = np.argmax(probabilities)
    confidence = probabilities[idx] * 100
    name = encoder.inverse_transform([idx])[0]

    return name, confidence, cropped, confidence >= threshold


def get_student_info(name, students_df):
    row = students_df[students_df["Name"] == name]

    if len(row) == 0:
        return None

    r = row.iloc[0]

    return {
        "Name": r["Name"],
        "Role": r["Role"],
        "Department": r["Department"],
        "Batch": r.get("Batch", "—"),
        "Mobile": r.get("mobile", "—"),
        "Gmail": r.get("gmail", "—"),
        "Validity": r.get("validity", "—"),
    }


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="nx-header">
    <div class="nx-topline">
        <span class="nx-live"></span>
        AI RECOGNITION ENGINE • ONLINE
    </div>

    <div class="nx-title">
        <span>Identity,</span><br>
        recognized intelligently.
    </div>

    <div class="nx-copy">
        A computer-vision identity platform that detects a face,
        generates a FaceNet embedding, classifies it with KNN,
        and retrieves the corresponding student profile.
    </div>

    <div class="nx-chip">
        MEDIAPIPE &nbsp;•&nbsp; FACENET &nbsp;•&nbsp; KNN &nbsp;•&nbsp; REAL-TIME INFERENCE
    </div>
</div>
""")


# ============================================================
# LOAD
# ============================================================

models_ok = False
load_error = None

try:
    with st.spinner("Initializing recognition engine..."):
        knn, encoder, scaler, detector, embedder = load_models(
            str(MODEL_PATH),
            str(LANDMARKER_PATH),
        )
    models_ok = True
except Exception as e:
    load_error = str(e)
    st.error(f"Recognition engine could not start: {e}")


students_df = None

try:
    students_df = load_students(str(STUDENTS_PATH))
except Exception as e:
    st.warning(f"Student database could not be loaded: {e}")


# ============================================================
# STATUS CARDS
# ============================================================

student_count = len(students_df) if students_df is not None else 0
class_count = len(encoder.classes_) if models_ok else 0

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.html(f"""
    <div class="nx-stat">
        <div class="nx-stat-icon">◉</div>
        <div class="nx-stat-value {'good' if models_ok else 'bad'}">
            {"ONLINE" if models_ok else "OFFLINE"}
        </div>
        <div class="nx-stat-label">Recognition Engine</div>
    </div>
    """)

with s2:
    st.html(f"""
    <div class="nx-stat">
        <div class="nx-stat-icon">◎</div>
        <div class="nx-stat-value">{student_count}</div>
        <div class="nx-stat-label">Registered Students</div>
    </div>
    """)

with s3:
    st.html(f"""
    <div class="nx-stat">
        <div class="nx-stat-icon">◇</div>
        <div class="nx-stat-value">{class_count}</div>
        <div class="nx-stat-label">Trained Identities</div>
    </div>
    """)

with s4:
    st.html(f"""
    <div class="nx-stat">
        <div class="nx-stat-icon">⚡</div>
        <div class="nx-stat-value">{confidence_threshold}%</div>
        <div class="nx-stat-label">Match Threshold</div>
    </div>
    """)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN WORKSPACE
# ============================================================

left, right = st.columns([1, 1], gap="large")


# ============================================================
# LEFT — SCAN
# ============================================================

with left:
    st.html("""
    <div class="nx-panel">
        <div class="nx-panel-head">
            <div>
                <div class="nx-panel-title">◈ Identity Scan</div>
                <div class="nx-panel-sub">
                    Provide a clear face image to begin verification.
                </div>
            </div>
            <div class="nx-panel-tag">INPUT</div>
        </div>
    </div>
    """)

    uploaded = st.file_uploader(
        "Upload face image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(
                img_rgb,
                caption=uploaded.name,
                use_container_width=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        run_btn = st.button(
            "◉  ANALYZE IDENTITY",
            disabled=not models_ok,
        )

    else:
        st.html("""
        <div class="nx-drop">
            <div>
                <div class="nx-drop-icon">◇</div>
                <div class="nx-drop-title">Ready for a face</div>
                <div class="nx-drop-sub">
                    JPG • JPEG • PNG
                </div>
            </div>
        </div>
        """)

        run_btn = False


# ============================================================
# RIGHT — RESULT
# ============================================================

with right:
    st.html("""
    <div class="nx-panel">
        <div class="nx-panel-head">
            <div>
                <div class="nx-panel-title">◉ Recognition Intelligence</div>
                <div class="nx-panel-sub">
                    Verification output from the AI recognition pipeline.
                </div>
            </div>
            <div class="nx-panel-tag">OUTPUT</div>
        </div>
    """)

    if uploaded and run_btn:

        st.html("""
        <div class="nx-scan">
            <div class="nx-scan-frame">
                <div class="nx-scan-line"></div>
            </div>
            <div class="nx-scan-label">ANALYZING IDENTITY</div>
            <div class="nx-scan-sub">
                Detecting → Embedding → Matching
            </div>
        </div>
        """)

        with st.spinner(""):
            start_time = time.time()

            name, confidence, cropped, passed = predict_identity(
                img_bgr,
                knn,
                encoder,
                scaler,
                detector,
                embedder,
                confidence_threshold,
            )

            elapsed = time.time() - start_time

        if name is None:
            st.html("""
            <div class="nx-unknown">
                <div style="font-size:40px;color:#ff7285;">◌</div>
                <div class="nx-unknown-title">No Face Detected</div>
                <div class="nx-unknown-sub">
                    Try a clearer image with the face fully visible.
                </div>
            </div>
            """)

        elif not passed:
            st.html(f"""
            <div class="nx-unknown">
                <div style="font-size:40px;color:#ff7285;">?</div>
                <div class="nx-unknown-title">Identity Not Verified</div>
                <div class="nx-unknown-sub">
                    Best model match: <b style="color:#e7ebff;">{name}</b>
                </div>

                <div class="nx-confidence">
                    <div class="nx-confidence-top">
                        <span>Match confidence</span>
                        <b style="color:#ff7182;">{confidence:.1f}%</b>
                    </div>

                    <div class="nx-track">
                        <div
                            class="nx-fill"
                            style="width:{confidence}%;background:linear-gradient(90deg,#ff536f,#ff8b7a);"
                        ></div>
                    </div>
                </div>
            </div>
            """)

        else:
            info = (
                get_student_info(name, students_df)
                if students_df is not None
                else None
            )

            display_name = info["Name"] if info else name

            st.html(f"""
            <div class="nx-verified">
                <div class="nx-eyebrow">VERIFIED IDENTITY</div>
                <div class="nx-name">{display_name}</div>

                <div class="nx-verified-pill">
                    ✓ IDENTITY CONFIRMED
                </div>

                <div class="nx-confidence">
                    <div class="nx-confidence-top">
                        <span>Recognition confidence</span>
                        <b>{confidence:.1f}%</b>
                    </div>

                    <div class="nx-track">
                        <div
                            class="nx-fill"
                            style="width:{confidence}%;"
                        ></div>
                    </div>
                </div>

                <div class="nx-time">
                    Inference completed in {elapsed * 1000:.0f} ms
                </div>
            </div>
            """)

            if cropped is not None:
                st.markdown("<br>", unsafe_allow_html=True)

                face_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

                _, center, _ = st.columns([1, 1.2, 1])

                with center:
                    st.image(
                        face_rgb,
                        caption="Detected face",
                        width=180,
                    )

            if info:
                rows = [
                    ("Role", info["Role"]),
                    ("Department", info["Department"]),
                    ("Batch", info["Batch"]),
                    ("Mobile", info["Mobile"]),
                    ("Gmail", info["Gmail"]),
                    ("Valid Until", info["Validity"]),
                ]

                row_html = ""

                for label, value in rows:
                    row_html += f"""
                    <div class="nx-row">
                        <span class="nx-key">{label}</span>
                        <span class="nx-value">{value}</span>
                    </div>
                    """

                st.html(f"""
                <div class="nx-details">
                    <div class="nx-details-title">Student Profile</div>
                    {row_html}
                </div>
                """)

    else:
        st.html("""
        <div class="nx-empty">
            <div class="nx-empty-icon">◉</div>
            <div class="nx-empty-title">Awaiting Identity Scan</div>
            <div class="nx-empty-sub">
                Upload an image to start the recognition pipeline.
            </div>
        </div>
        """)

    st.html("</div>")


# ============================================================
# REGISTERED STUDENTS
# ============================================================

if students_df is not None:
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.html("""
    <div class="nx-panel">
        <div class="nx-panel-title">◎ Registered Identity Database</div>
        <div class="nx-panel-sub">
            Current identities available to the recognition engine.
        </div>
    </div>
    """)

    with st.expander("VIEW REGISTERED STUDENTS"):
        st.dataframe(
            students_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="nx-footer">
    NEXUS FACE ID &nbsp;•&nbsp;
    MEDIAPIPE &nbsp;•&nbsp;
    FACENET &nbsp;•&nbsp;
    KNN &nbsp;•&nbsp;
    INTELLIGENT IDENTITY RECOGNITION
</div>
""")
