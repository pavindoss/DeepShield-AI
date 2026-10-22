from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2
import csv
import json
import re
import io
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import make_response
from reportlab.lib import colors

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
website_history = []

app = Flask(__name__)


def inspect_email_authentication(headers):
    """Parse reported SPF/DKIM/DMARC statuses from supplied email headers.

    This does not independently verify authentication; pasted headers can be forged.
    """
    checks = {
        "SPF": "Not provided",
        "DKIM": "Not provided",
        "DMARC": "Not provided",
    }

    if not headers.strip():
        return checks

    auth_lines = re.findall(
        r"(?im)^authentication-results\s*:\s*([^\r\n]+"
        r"(?:\r?\n[ \t]+[^\r\n]+)*)",
        headers,
    )
    auth_text = " ".join(auth_lines).lower()

    received_spf = re.findall(
        r"(?im)^received-spf\s*:\s*([^\r\n]+)",
        headers,
    )
    spf_text = auth_text + " " + " ".join(received_spf).lower()

    for name, text_to_search in (
        ("SPF", spf_text),
        ("DKIM", auth_text),
        ("DMARC", auth_text),
    ):
        match = re.search(
            rf"\b{name.lower()}\s*=\s*(pass|fail|softfail|neutral|"
            rf"none|temperror|permerror)\b",
            text_to_search,
        )
        if match:
            checks[name] = match.group(1).upper()

    return checks


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "deepshield_model.keras"

IMAGE_SIZE = (128, 128)

THRESHOLD = 0.45

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
REPORT_FOLDER = "static/reports"

HISTORY_FILE = "prediction_history.json"

MAX_HISTORY = 10


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(MODEL_PATH)

print("DeepShield AI model loaded successfully!")


# =========================================================
# FIND GRAD-CAM LAYER
# =========================================================

def find_gradcam_layer():

    try:
        layer = model.get_layer("out_relu")

        print("Grad-CAM layer:", layer.name)

        return layer

    except Exception:
        pass

    for layer in reversed(model.layers):

        try:

            if len(layer.output.shape) == 4:

                print("Grad-CAM layer:", layer.name)

                return layer

        except Exception:
            continue

    raise ValueError(
        "Could not find a suitable Grad-CAM layer."
    )


TARGET_LAYER = find_gradcam_layer()


# =========================================================
# HISTORY FUNCTIONS
# =========================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except Exception as error:

        print(
            "History load error:",
            error
        )

        return []


def save_history(history):

    try:

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:

        print(
            "History save error:",
            error
        )


# =========================================================
# CREATE UNIQUE HISTORY KEY
# =========================================================

def history_key(record):

    filename = str(
        record.get("filename", "")
    ).strip().lower()

    prediction = str(
        record.get("prediction", "")
    ).strip().upper()

    try:

        ai_percentage = round(
            float(
                record.get(
                    "ai_percentage",
                    0
                )
            ),
            2
        )

    except Exception:

        ai_percentage = 0.0

    try:

        real_percentage = round(
            float(
                record.get(
                    "real_percentage",
                    0
                )
            ),
            2
        )

    except Exception:

        real_percentage = 0.0

    return (
        filename,
        prediction,
        ai_percentage,
        real_percentage
    )


# =========================================================
# REMOVE DUPLICATE HISTORY
# =========================================================

def remove_duplicate_history(history):

    unique_history = []

    seen = set()

    for record in history:

        if not isinstance(record, dict):
            continue

        key = history_key(record)

        if key in seen:
            continue

        seen.add(key)

        record["filename"] = str(
            record.get(
                "filename",
                ""
            )
        ).strip()

        record["prediction"] = str(
            record.get(
                "prediction",
                ""
            )
        ).strip().upper()

        try:

            record["ai_percentage"] = round(
                float(
                    record.get(
                        "ai_percentage",
                        0
                    )
                ),
                2
            )

        except Exception:

            record["ai_percentage"] = 0.0

        try:

            record["real_percentage"] = round(
                float(
                    record.get(
                        "real_percentage",
                        0
                    )
                ),
                2
            )

        except Exception:

            record["real_percentage"] = 0.0

        unique_history.append(record)

    return unique_history[:MAX_HISTORY]


# =========================================================
# CHECK DUPLICATE
# =========================================================

def history_contains(history, new_record):

    new_key = history_key(new_record)

    for old_record in history:

        if history_key(old_record) == new_key:

            return True

    return False


# =========================================================
# GENERATE PDF REPORT
# =========================================================

def generate_pdf_report(
    filename,
    result,
    ai_percentage,
    real_percentage,
    threshold,
    reason,
    image_path,
    gradcam_path
):

    report_filename = (
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    story = []

    # Title
    story.append(
        Paragraph(
            "<b>DeepShield AI - Detection Report</b>",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # Image name
    story.append(
        Paragraph(
            f"<b>Image:</b> {filename}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    # Result
    story.append(
        Paragraph(
            f"<b>Final Result:</b> {result}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    # AI probability
    story.append(
        Paragraph(
            f"<b>AI Probability:</b> "
            f"{ai_percentage:.2f}%",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    # Real probability
    story.append(
        Paragraph(
            f"<b>Real Probability:</b> "
            f"{real_percentage:.2f}%",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    # Threshold
    story.append(
        Paragraph(
            f"<b>Decision Threshold:</b> "
            f"{threshold}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # Explanation
    story.append(
        Paragraph(
            "<b>Why this prediction?</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            reason,
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # Original image
    if image_path and os.path.exists(image_path):

        story.append(
            Paragraph(
                "<b>Uploaded Image</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Spacer(1, 8)
        )

        story.append(
            PDFImage(
                image_path,
                width=4 * inch,
                height=3 * inch
            )
        )

        story.append(
            Spacer(1, 20)
        )

    # Grad-CAM
    if gradcam_path and os.path.exists(gradcam_path):

        story.append(
            Paragraph(
                "<b>Grad-CAM Visualization</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Spacer(1, 8)
        )

        story.append(
            PDFImage(
                gradcam_path,
                width=4 * inch,
                height=3 * inch
            )
        )

    story.append(
        Spacer(1, 20)
    )

    # Date and time
    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
            styles["Normal"]
        )
    )

    doc.build(story)

    return report_path.replace(
        "\\",
        "/"
    )


# =========================================================
# GENERATE EXPLANATION
# =========================================================

def generate_reason(
    prediction,
    ai_probability,
    real_probability
):

    confidence = max(
        ai_probability,
        real_probability
    )

    if prediction == "AI":

        if confidence >= 0.85:

            return (
                "The model detected visual patterns that were "
                "strongly associated with AI-generated images "
                "in its training data. The Grad-CAM highlights "
                "the regions that contributed most to this "
                "prediction."
            )

        elif confidence >= 0.60:

            return (
                "The model detected several visual patterns "
                "that were more consistent with AI-generated "
                "images. The confidence is moderate, so this "
                "should be treated as a model prediction rather "
                "than definitive proof."
            )

        else:

            return (
                "The model found some visual patterns associated "
                "with AI-generated images, but the confidence "
                "is relatively low. The image may contain "
                "characteristics shared by both AI and real images."
            )

    else:

        if confidence >= 0.85:

            return (
                "The model detected visual patterns that were "
                "strongly associated with real images in its "
                "training data. The Grad-CAM highlights the "
                "regions that contributed most to this prediction."
            )

        elif confidence >= 0.60:

            return (
                "The model detected several visual patterns "
                "that were more consistent with real images. "
                "The confidence is moderate, so this should "
                "be treated as a model prediction rather than "
                "definitive proof."
            )

        else:

            return (
                "The model found some visual patterns associated "
                "with real images, but the confidence is relatively "
                "low. The image may contain characteristics shared "
                "by both AI and real images."
            )


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    image_array,
    original_image_path,
    predicted_class
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            TARGET_LAYER.output,
            model.output
        ]
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        real_probability = predictions[:, 0]

        if predicted_class == "REAL":

            target_score = real_probability

        else:

            target_score = 1.0 - real_probability

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    original = cv2.imread(
        original_image_path
    )

    if original is None:

        raise ValueError(
            "Could not read uploaded image."
        )

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    gradcam_result = cv2.addWeighted(
        original,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    heatmap_filename = (
        "heatmap_" +
        timestamp +
        ".jpg"
    )

    gradcam_filename = (
        "gradcam_" +
        timestamp +
        ".jpg"
    )

    heatmap_full_path = os.path.join(
        GRADCAM_FOLDER,
        heatmap_filename
    )

    gradcam_full_path = os.path.join(
        GRADCAM_FOLDER,
        gradcam_filename
    )

    cv2.imwrite(
        heatmap_full_path,
        colored_heatmap
    )

    cv2.imwrite(
        gradcam_full_path,
        gradcam_result
    )

    print(
        "Grad-CAM generated successfully!"
    )

    return (
        "gradcam/" + heatmap_filename,
        "gradcam/" + gradcam_filename
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route(
    "/image-detection",
    methods=["GET", "POST"]
)
def index():

    result = None

    percentage = None

    label = None

    image_path = None

    heatmap_path = None

    gradcam_path = None

    report_path = None

    reason = None

    ai_percentage = None

    real_percentage = None

    threshold = THRESHOLD


    # =====================================================
    # LOAD AND CLEAN HISTORY
    # =====================================================

    history = load_history()

    cleaned_history = remove_duplicate_history(
        history
    )

    if cleaned_history != history:

        save_history(
            cleaned_history
        )

    history = cleaned_history


    # =====================================================
    # POST
    # =====================================================

    if request.method == "POST":

        if "image" not in request.files:

            return render_template(
                "index.html",
                history=history,
                threshold=threshold
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "index.html",
                history=history,
                threshold=threshold
            )


        # =================================================
        # SAFE FILE NAME
        # =================================================

        original_filename = secure_filename(
            file.filename
        )

        if not original_filename:

            return render_template(
                "index.html",
                history=history,
                threshold=threshold
            )


        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            timestamp +
            "_" +
            original_filename
        )


        # =================================================
        # SAVE IMAGE
        # =================================================

        full_image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(
            full_image_path
        )


        # =================================================
        # PREPROCESS IMAGE
        # =================================================

        try:

            image = Image.open(
                full_image_path
            ).convert("RGB")

            image = image.resize(
                IMAGE_SIZE
            )

            image_array = np.array(
                image,
                dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

        except Exception as error:

            print(
                "Image processing error:",
                error
            )

            return render_template(
                "index.html",
                history=history,
                threshold=threshold
            )


        # =================================================
        # PREDICTION
        # =================================================

        prediction = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

        print(
            "Model prediction:",
            prediction
        )


        # =================================================
        # PROBABILITIES
        # =================================================

        real_probability = prediction

        ai_probability = (
            1.0 - prediction
        )

        ai_percentage = round(
            ai_probability * 100,
            2
        )

        real_percentage = round(
            real_probability * 100,
            2
        )


        # =================================================
        # CLASSIFICATION
        # =================================================

        if prediction >= THRESHOLD:

            result = "Real Image ✅"

            percentage = real_percentage

            label = "Real Percentage"

            predicted_class = "REAL"

        else:

            result = "AI Generated ❌"

            percentage = ai_percentage

            label = "AI Percentage"

            predicted_class = "AI"


        # =================================================
        # EXPLANATION
        # =================================================

        reason = generate_reason(
            predicted_class,
            ai_probability,
            real_probability
        )


        # =================================================
        # IMAGE PATH
        # =================================================

        image_path = (
            "uploads/" +
            filename
        )


        # =================================================
        # GRAD-CAM
        # =================================================

        try:

            (
                heatmap_path,
                gradcam_path
            ) = generate_gradcam(
                image_array,
                full_image_path,
                predicted_class
            )

        except Exception as error:

            print(
                "Grad-CAM error:",
                error
            )

            heatmap_path = None

            gradcam_path = None


        # =================================================
        # GENERATE PDF REPORT
        # =================================================

        try:

            gradcam_full_path = None

            if gradcam_path:

                gradcam_full_path = os.path.join(
                    "static",
                    gradcam_path
                )

            report_path = generate_pdf_report(
                filename=original_filename,
                result=result,
                ai_percentage=ai_percentage,
                real_percentage=real_percentage,
                threshold=THRESHOLD,
                reason=reason,
                image_path=full_image_path,
                gradcam_path=gradcam_full_path
            )

            print(
                "PDF report generated successfully:",
                report_path
            )

        except Exception as error:

            print(
                "PDF report error:",
                error
            )

            report_path = None


        # =================================================
        # CREATE NEW HISTORY RECORD
        # =================================================

        new_record = {

            "filename":
                original_filename,

            "prediction":
                predicted_class,

            "result":
                (
                    "AI Generated"
                    if predicted_class == "AI"
                    else "Real Image"
                ),

            "ai_percentage":
                ai_percentage,

            "real_percentage":
                real_percentage,

            "threshold":
                THRESHOLD,

            "date":
                datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                )
        }


        # =================================================
        # RELOAD HISTORY
        # =================================================

        history = load_history()

        history = remove_duplicate_history(
            history
        )


        # =================================================
        # DUPLICATE PREVENTION
        # =================================================

        if history_contains(
            history,
            new_record
        ):

            print(
                "Duplicate history entry blocked:",
                original_filename
            )

        else:

            history.insert(
                0,
                new_record
            )

            print(
                "New history entry added:",
                original_filename
            )


        # =================================================
        # CLEAN + KEEP LATEST 20
        # =================================================

        history = remove_duplicate_history(
            history
        )

        save_history(
            history
        )


    # =====================================================
    # SEND DATA TO HTML
    # =====================================================

    return render_template(
        "index.html",

        result=result,

        percentage=percentage,

        label=label,

        image_path=image_path,

        heatmap_path=heatmap_path,

        gradcam_path=gradcam_path,

        report_path=report_path,

        reason=reason,

        ai_percentage=ai_percentage,

        real_percentage=real_percentage,

        threshold=threshold,

        history=history
    )


# =========================================================
# RUN
# =========================================================

# EMAIL DETECTION PAGE

from urllib.parse import urlparse



def analyze_email(sender, subject, content, authentication=None):
    text = f"{subject} {content}".lower()
    findings = []
    score = 0

    # Urgent or threatening language
    urgency_words = [
        "urgent", "immediately", "account suspended",
        "act now", "verify your account", "limited time",
        "within 24 hours"
    ]
    found_urgency = [word for word in urgency_words if word in text]
    if found_urgency:
        score += 20
        findings.append(
            "Urgent or threatening language: " + ", ".join(found_urgency)
        )

    # Requests for sensitive information
    sensitive_phrases = [
        "password", "one time password", "otp",
        "bank details", "credit card", "pin number",
        "social security"
    ]
    found_sensitive = [
        word for word in sensitive_phrases if word in text
    ]
    if found_sensitive:
        score += 25
        findings.append(
            "Sensitive information mentioned: " + ", ".join(found_sensitive)
        )

    
    # URL and domain analysis
    urls = re.findall(r"""https?://[^\s<>"']+""", content, re.I)
    suspicious_urls = []
    url_warnings = []

    shorteners = {
        "bit.ly", "tinyurl.com", "t.co", "is.gd",
        "ow.ly", "buff.ly", "rebrand.ly"
    }

    for url in urls:
        clean_url = url.rstrip(".,!?;:)")
        try:
            parsed = urlparse(clean_url)
            host = (parsed.hostname or "").lower().rstrip(".")

            warnings = []

            if not host:
                warnings.append("invalid or missing hostname")
            else:
                # HTTP is unencrypted; it is a warning, not proof of phishing
                if parsed.scheme.lower() != "https":
                    warnings.append("uses HTTP instead of HTTPS")

                # IP-address-based links
                if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
                    warnings.append("uses an IP address instead of a domain")

                # Internationalized/punycode hostname
                if any(label.startswith("xn--") for label in host.split(".")):
                    warnings.append("contains a punycode hostname")

                # Known URL-shortener domains
                if any(
                    host == shortener or host.endswith("." + shortener)
                    for shortener in shorteners
                ):
                    warnings.append("uses a URL-shortening service")

                # Misleading @ symbol in URL authority
                if "@" in parsed.netloc:
                    warnings.append("contains an @ symbol in the URL authority")

                # Suspiciously long hostname
                if len(host) > 60:
                    warnings.append("has an unusually long hostname")

                # Common brand names embedded in unrelated hostnames
                brand_terms = (
                    "paypal", "microsoft", "google", "apple", "amazon"
                )
                known_domains = {
                    "paypal": ("paypal.com",),
                    "microsoft": ("microsoft.com", "live.com", "office.com"),
                    "google": ("google.com", "gmail.com"),
                    "apple": ("apple.com", "icloud.com"),
                    "amazon": ("amazon.com", "amazon.in"),
                }

                for brand in brand_terms:
                    if brand in host and not any(
                        host == domain or host.endswith("." + domain)
                        for domain in known_domains[brand]
                    ):
                        warnings.append(
                            f"contains '{brand}' in a non-matching domain"
                        )
                        break

            if warnings:
                suspicious_urls.append(clean_url)
                url_warnings.append(
                    f"{clean_url} — " + "; ".join(warnings)
                )

        except (ValueError, TypeError):
            suspicious_urls.append(clean_url)
            url_warnings.append(f"{clean_url} — malformed URL")

    if suspicious_urls:
        score += 30
        findings.append(
            "Potentially suspicious URL(s): " + "; ".join(url_warnings)
        )


    # Sender address checks
    sender = sender.strip().lower()
    if sender:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", sender):
            score += 20
            findings.append("Sender address format looks unusual.")
        elif re.search(r"\d{4,}|[._-]{2,}", sender.split("@")[0]):
            score += 10
            findings.append(
                "Sender name contains unusual numbers or symbols."
            )

    # Common phishing promises
    promise_phrases = [
        "you have won", "claim your prize",
        "free gift", "lottery winner", "you are selected"
    ]
    found_promises = [word for word in promise_phrases if word in text]
    if found_promises:
        score += 20
        findings.append(
            "Unexpected prize or reward language: " + ", ".join(found_promises)
        )

    # Reported SPF / DKIM / DMARC results from supplied headers
    if authentication:
        failed_checks = [
            name for name in ("SPF", "DKIM", "DMARC")
            if str(authentication.get(name, "")).upper()
            in ("FAIL", "SOFTFAIL", "PERMERROR")
        ]

        if failed_checks:
            score += min(15 * len(failed_checks), 30)
            findings.append(
                "Authentication failure reported in supplied headers: "
                + ", ".join(failed_checks)
            )

        temporary_checks = [
            name for name in ("SPF", "DKIM", "DMARC")
            if str(authentication.get(name, "")).upper()
            in ("TEMPERROR", "NEUTRAL", "NONE")
        ]

        if temporary_checks:
            findings.append(
                "Authentication inconclusive or not passed: "
                + ", ".join(temporary_checks)
            )

    score = min(score, 100)

    if score >= 50:
        result = "High Risk — Possible Phishing"
    elif score >= 25:
        result = "Medium Risk — Review Carefully"
    else:
        result = "Low Indicators Found — Still Verify"

    if not findings:
        findings.append(
            "No configured warning patterns were found. "
            "This does not prove the email is safe."
        )

    return {
        "result": result,
        "score": score,
        "findings": findings
    }


email_history = []
@app.route("/email-detection", methods=["GET", "POST"])
def email_detection():
    result = None
    sender = ""
    subject = ""
    content = ""
    headers = ""

    if request.method == "POST":
        sender = request.form.get("sender", "").strip()
        subject = request.form.get("subject", "").strip()
        content = request.form.get("content", "").strip()
        headers = request.form.get("headers", "").strip()

        if not content:
            result = {
                "error": "Please enter the email content before analysis."
            }
        else:
           authentication = inspect_email_authentication(headers)
           result = analyze_email(sender, subject, content, authentication)
           result["authentication"] = authentication
           
           if "error" not in result:
               email_history.insert(0, {
                   "sender": sender,
                   "subject": subject,
                   "score": result["score"],
                   "status": result["result"],
                   "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
               })

               email_history[:] = email_history[:10]



    return render_template(
        "email.html",
        result=result,
        sender=sender,
        subject=subject,
        content=content,
        headers=headers,
        email_history=email_history
    )




def analyze_website_url(url):
    findings = []
    score = 0


    try:
        url = url.strip()
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")

        if (
            parsed.scheme.lower() not in ("http", "https")
            or not host
            or "." not in host
            or any(char.isspace() for char in url)
           ):
            return {
              "score": 0,
              "result": "Invalid URL",
              "findings": [
                "Enter a valid website URL, such as https://example.com."
            ]
        }


        # HTTP is not encrypted; this alone does not prove phishing.
        if parsed.scheme == "http":
            score += 15
            findings.append("Uses HTTP instead of HTTPS.")

        # IP address used as the hostname
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            score += 25
            findings.append("Uses an IP address instead of a domain name.")

        
        # Detect brand names in unrelated hostnames
        known_domains = {
            "paypal": ("paypal.com",),
            "google": ("google.com", "gmail.com"),
            "microsoft": ("microsoft.com", "live.com", "office.com"),
            "apple": ("apple.com", "icloud.com"),
            "amazon": ("amazon.com", "amazon.in"),
        }

        for brand, official_domains in known_domains.items():
            if brand in host and not any(
                host == domain or host.endswith("." + domain)
                for domain in official_domains
            ):
                score += 25
                findings.append(
                    f"Contains '{brand}' in a hostname that does not "
                    "match the configured official domains."
                )
                break

    
        # Detect common URL-shortening services
        shorteners = {
            "bit.ly", "tinyurl.com", "t.co",
            "is.gd", "ow.ly", "buff.ly", "rebrand.ly"
        }

        if any(
            host == domain or host.endswith("." + domain)
            for domain in shorteners
        ):
            score += 20
            findings.append(
                "Uses a URL-shortening service. "
                "The final destination is not verified."
            )

        # Punycode hostname
        if any(part.startswith("xn--") for part in host.split(".")):
            score += 20
            findings.append("Contains a punycode hostname.")

        # URL contains an @ symbol in its authority section
        if "@" in parsed.netloc:
            score += 25
            findings.append("Contains an @ symbol in the URL authority.")

        # Excessively long hostname
        if len(host) > 60:
            score += 15
            findings.append("Has an unusually long hostname.")

        
        # Informational check for sensitive-looking URL paths
        path = parsed.path.lower()
        sensitive_paths = (
            "/login", "/signin", "/verify",
            "/account", "/password", "/reset"
        )

        if any(
            segment in path
            for segment in sensitive_paths
        ):
            findings.append(
                "URL path references a login, account, verification, "
                "or password-related page. This is informational only."
            )

        # Unusual number of subdomains
        if len(host.split(".")) >= 5:
            score += 10
            findings.append("Contains many hostname segments.")

        score = min(score, 100)

        if score >= 50:
            result = "High Risk — Suspicious URL Indicators"
        elif score >= 25:
            result = "Medium Risk — Review Carefully"
        else:
            result = "Low Indicators Found — Still Verify"

        if not findings:
            findings.append(
                "No configured URL warning patterns were found. "
                "This does not prove the website is safe."
            )

        return {
            "score": score,
            "result": result,
            "findings": findings
        }

    except (ValueError, TypeError):
        return {
            "score": 100,
            "result": "Invalid URL",
            "findings": ["The supplied URL could not be parsed."]
        }


@app.route("/website-detection", methods=["GET", "POST"])


def website_detection():
    result = None



    if request.method == "POST":
        website_url = request.form.get("website_url", "").strip()
        result = analyze_website_url(website_url)

        if result.get("result") != "Invalid URL":
            website_history.insert(0, {
                "url": website_url,
                "score": result["score"],
                "status": result["result"],
                "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
            })

            website_history[:] = website_history[:10]




    return render_template(
        "website.html",
        result=result,
        website_history=website_history
    )

  

@app.route("/download-website-history")
def download_website_history():
    import io
    from flask import make_response

    history = website_history

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=25,
        leftMargin=25,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    elements = [
        Paragraph("DeepShield AI - Website Analysis History", styles["Title"]),
        Spacer(1, 18)
    ]

    data = [[
        "Website URL",
        "Risk Score",
        "Status",
        "Analyzed At"
    ]]

    for item in history:
        data.append([
            Paragraph(str(item.get("url", "")), styles["BodyText"]),
            str(item.get("score", "")) + "/100",
            Paragraph(str(item.get("status", "")), styles["BodyText"]),
            str(item.get("time", ""))
        ])

    table = Table(
        data,
        colWidths=[300, 75, 200, 130],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef6ff")]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    elements.append(table)
    doc.build(elements)

    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = (
        "attachment; filename=website_analysis_history.pdf"
    )

    return response


@app.route("/download-history")
def download_image_history():
    import io
    from flask import make_response

    history = load_history()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=25,
        leftMargin=25,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    elements = [
        Paragraph("DeepShield AI - Image Detection History", styles["Title"]),
        Spacer(1, 18)
    ]

    data = [[
        "Filename", "Prediction", "AI %", "Real %",
        "Threshold", "Date"
    ]]

    for item in history:
        data.append([
            str(item.get("filename", "")),
            str(item.get("result", "")),
            str(item.get("ai_percentage", "")),
            str(item.get("real_percentage", "")),
            str(item.get("threshold", "")),
            str(item.get("date", ""))
        ])

    table = Table(
        data,
        colWidths=[150, 100, 55, 55, 65, 120],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef6ff")]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(table)
    doc.build(elements)

    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = (
        "attachment; filename=ai_image_history.pdf"
    )

    return response



@app.route("/download-email-history")
def download_email_history():
    import io
    from flask import make_response

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=25,
        leftMargin=25,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    elements = [
        Paragraph(
            "DeepShield AI - Email Analysis History",
            styles["Title"]
        ),
        Spacer(1, 18)
    ]

    data = [[
        "Sender",
        "Subject",
        "Risk Score",
        "Status",
        "Analyzed At"
    ]]

    for item in email_history:
        data.append([
            Paragraph(
                str(item.get("sender") or "Not provided"),
                styles["BodyText"]
            ),
            Paragraph(
                str(item.get("subject") or "No subject"),
                styles["BodyText"]
            ),
            str(item.get("score", "")) + "/100",
            Paragraph(
                str(item.get("status", "")),
                styles["BodyText"]
            ),
            str(item.get("time", ""))
        ])

    table = Table(
        data,
        colWidths=[125, 180, 65, 115, 125],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef6ff")]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    elements.append(table)
    doc.build(elements)

    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = (
        "attachment; filename=email_analysis_history.pdf"
    )

    return response


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)