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
| Frontend | HTML, CSS, JavaScript |
| Version Control | Git & GitHub |

---

## 🏗️ Project Structure

```text
DeepShield-AI/
├── app.py
├── deepshield_model.keras
├── gradcam.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── email.html
│   └── website.html
├── static/
│   └── logo.png
├── train_model.py
├── evaluate_model.py
├── evaluate_validation.py
├── extract_dataset.py
├── find_threshold.py
└── check_layers.py
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11 recommended for the development environment.
- Git.
- Sufficient memory and storage for TensorFlow and the trained model.

### 1. Clone the repository

```bash
git clone https://github.com/pavindoss/DeepShield-AI.git
cd DeepShield-AI
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local URL printed by Flask in your terminal, usually:

```text
http://127.0.0.1:5000
```

---

## 🧠 Model Evaluation

The image classification model has been evaluated on a held-out test set during development.

**Recorded test accuracy: 93.92%**

This is a development evaluation result, not a guarantee of performance on every image or on images from unseen sources. Actual performance may vary with dataset quality, image preprocessing, and differences between training and real-world images.

---

## 🔐 Security & Responsible Use

- Treat all predictions as indicators, not definitive proof.
- Do not upload confidential, private, or sensitive images or emails unless you understand how the application stores and processes them.
- Never enter passwords, API keys, or other secrets into public repositories.
- URL and email analysis may produce false positives and false negatives.
- A website using HTTPS is not necessarily trustworthy.
- Do not use this project as the sole basis for security-critical decisions.

---

## 🔮 Future Enhancements

- Improve image detection using more diverse and representative datasets.
- Add stronger validation against unseen image sources.
- Extend URL analysis with carefully controlled website-content inspection.
- Improve email authentication and header analysis.
- Add safer deployment configuration, logging, and monitoring.
- Improve report generation and user experience.

---

## 🎓 Project Information

**Project:** DeepShield AI  
**Domain:** Artificial Intelligence, Deep Learning & Cybersecurity  
**Repository:** [github.com/pavindoss/DeepShield-AI](https://github.com/pavindoss/DeepShield-AI)

Developed as an educational project exploring how AI and rule-based analysis can support image authenticity checks and phishing awareness.

---

## ⭐ Support

If you find this project useful for learning or experimentation, consider giving the repository a star.

**Disclaimer:** DeepShield AI is an educational security-support tool. Its outputs are not a substitute for professional security investigation or independent verification.
