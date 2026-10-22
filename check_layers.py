import tensorflow as tf

model = tf.keras.models.load_model("deepshield_model.keras")

print("\nMODEL LAYERS")
print("=" * 60)

for layer in model.layers:
    print(layer.name, "->", layer.__class__.__name__)

print("=" * 60)