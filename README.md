# DeepShield-AI
AI-Powered Image and Phishing Detection System
# 🛡️ DeepShield AI
### AI-Powered Image Authenticity & Phishing Detection System

DeepShield AI is a web-based security application designed to help users identify potentially AI-generated or manipulated images, analyze suspicious website URLs, and inspect email content for common phishing indicators. It combines deep learning with rule-based security analysis in a single interface.

> **Project Goal:** Make AI-assisted image authenticity checks and phishing awareness tools accessible through one user-friendly web application.

---

## 🚀 Key Features

### 🖼️ 1. AI-Generated Image Detection
- Analyze uploaded images using a trained deep learning model.
- Classify images according to the model's learned categories.
- Display prediction results through a simple web interface.
- Support visual explanation using Grad-CAM heatmaps, where available.

### 🌐 2. Website URL Phishing Analysis
- Analyze submitted URLs for common phishing indicators.
- Check URL structure, HTTPS usage, IP-based hostnames, suspicious subdomains, URL shorteners, and other heuristic signals.
- Highlight potentially suspicious URLs and provide a risk assessment.

**Note:** URL-based heuristic analysis does not guarantee that a website is safe or malicious.

### 📧 3. Email Phishing Analysis
- Examine email text for common phishing-related patterns.
- Identify suspicious language and indicators that may require further investigation.
- Inspect available email authentication information, including SPF, DKIM, and DMARC, when provided.

### 📊 4. Results & Reports
- Present analysis results through a web interface.
- Generate visual explanations for supported image predictions.
- Support downloadable reports where implemented.
- Store generated files locally while the application is running, subject to the application's configuration.

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Deep Learning | TensorFlow / Keras |
| Image Classification | MobileNetV2-based model |
| Image Explanation | Grad-CAM |
| Frontend |
