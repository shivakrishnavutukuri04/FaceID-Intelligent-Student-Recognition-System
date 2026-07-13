# 🎓 Face Recognition Student Identification System

An AI-powered student identification system that recognizes students from uploaded images using deep learning-based facial recognition. The application combines MediaPipe Face Landmarker for face detection, FaceNet for feature extraction, and a KNN classifier for identity prediction. A Streamlit interface allows users to upload images and instantly retrieve student details.

---

## 🚀 Features

- Face detection using MediaPipe Face Landmarker
- Deep feature extraction using FaceNet
- Student recognition using KNN Classifier
- Confidence score prediction
- Unknown face detection
- Student information retrieval
- Interactive Streamlit web application
- Fast and lightweight prediction pipeline

---

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenCV
- MediaPipe
- FaceNet
- Scikit-learn
- NumPy
- Pandas
- Joblib

---


## ⚙️ System Workflow

```
Input Image
      │
      ▼
MediaPipe Face Detection
      │
      ▼
Face Cropping
      │
      ▼
FaceNet Embedding Extraction
      │
      ▼
Feature Scaling
      │
      ▼
KNN Classification
      │
      ▼
Identity Prediction
      │
      ▼
Display Student Details
```

---

## 📊 Machine Learning Pipeline

- Face Detection → MediaPipe Face Landmarker
- Face Embedding → FaceNet (512-dimensional embeddings)
- Feature Scaling → StandardScaler
- Classification → K-Nearest Neighbors (KNN)
- User Interface → Streamlit

---

## 📸 Screenshots

Add screenshots here after running the application.

Example:

- Home Page
- Image Upload
- Recognition Result
- Student Details

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Face-Recognition-Student-Identification-System.git
```

Move into the project

```bash
cd Face-Recognition-Student-Identification-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- Real-time webcam recognition
- Attendance management
- Face anti-spoofing
- Multiple face recognition
- Database integration (MySQL/PostgreSQL)
- REST API using FastAPI
- Docker deployment
- Cloud deployment

---

## 👨‍💻 Author

**Shiva Krishna**

Machine Learning | Computer Vision | Python Developer
