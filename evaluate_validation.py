
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

MODEL_PATH = "deepshield_model_experiment.keras"
VALIDATION_PATH = "dataset/validation"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_data = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:", validation_data.class_indices)

print("\nGenerating validation predictions...")
predictions = model.predict(validation_data, verbose=1)

y_pred = (predictions >= 0.5).astype(int).flatten()
y_true = validation_data.classes

accuracy = accuracy_score(y_true, y_pred)

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["AI", "REAL"],
        digits=4,
        zero_division=0
    )
)

print("\nVALIDATION EVALUATION COMPLETE")
