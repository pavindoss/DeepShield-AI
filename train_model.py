import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==============================
# SETTINGS
# ==============================

TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/validation"
TEST_PATH = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

# ==============================
# DATA GENERATORS
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass Mapping:")
print(train_data.class_indices)

# ==============================
# MOBILE NET V2
# ==============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

# ==============================
# CLASSIFICATION HEAD
# ==============================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN
# ==============================

print("\nStarting Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================

model.save("deepshield_model.keras")

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print("Model saved as: deepshield_model.keras")

# ==============================
# TEST EVALUATION
# ==============================

print("\nEvaluating on Test Dataset...\n")

loss, accuracy = model.evaluate(test_data)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)