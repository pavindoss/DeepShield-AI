import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

MODEL_PATH = "deepShield_model.keras"
VAL_PATH = "dataset/validation"

model = tf.keras.models.load_model(MODEL_PATH)

val_data = ImageDataGenerator(
    rescale=1.0 / 255
).flow_from_directory(
    VAL_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:", val_data.class_indices)
print("Generating validation predictions...\n")

predictions = model.predict(val_data, verbose=1).ravel()
y_true = val_data.classes

print("\nTHRESHOLD RESULTS")
print("=" * 60)

for threshold in np.arange(0.30, 0.71, 0.05):

    y_pred = (predictions >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    ai_recall = tp / (tp + fn) if (tp + fn) else 0
    real_recall = tn / (tn + fp) if (tn + fp) else 0

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"AI Recall: {ai_recall:.4f} | "
        f"REAL Recall: {real_recall:.4f} | "
        f"AI→REAL: {fn} | REAL→AI: {fp}"
    )

print("\nDONE")