import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "deepshield_model.keras"
TEST_PATH = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
REPORT_PATH = "image_evaluation_report.txt"

# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print()

# ==============================
# LOAD TEST DATASET
# ==============================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print()
print("Class mapping:", test_data.class_indices)
print()

# ==============================
# MODEL EVALUATION
# ==============================

print("Evaluating test dataset...")

loss, accuracy = model.evaluate(test_data, verbose=1)

print()
print("==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
accuracy_percent = accuracy * 100
print()

# ==============================
# PREDICTIONS
# ==============================

print("Generating predictions...")

predictions = model.predict(test_data, verbose=1)

# Sigmoid output → class
y_pred = (predictions >= 0.5).astype(int).flatten()

# Actual labels
y_true = test_data.classes

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_true, y_pred)

print()
print("==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

# ==============================
# CLASSIFICATION REPORT
# ==============================

print()
print("==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["AI", "REAL"],
        digits=4
    )
)

print()
print("==============================")
print("EVALUATION COMPLETE")
print("==============================")
print(f"Report saved to: {REPORT_PATH}")

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("DeepShield AI - Image Model Evaluation Report\n")
    file.write("=" * 50 + "\n")
    file.write(f"Test Loss: {loss:.4f}\n")
    file.write(f"Test Accuracy: {accuracy_percent:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(classification_report(
        y_true, y_pred,
        target_names=["AI", "REAL"],
        digits=4
    ))
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm) + "\n")

print(f"Report saved to: {REPORT_PATH}")