import subprocess, sys, os

subprocess.run([sys.executable, "-m", "pip", "install", "tf2onnx", "onnx", "-q"], check=True)

import tensorflow as tf

print("Loading Keras model...")
model = tf.keras.models.load_model("model/seizure_model.h5", compile=False)
print(f"Input shape: {model.input_shape}")

saved_path = "model/seizure_model_saved"
print(f"Saving as SavedModel to {saved_path} ...")
model.export(saved_path)

print("Converting SavedModel to ONNX...")
result = subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", saved_path,
    "--output", "model/seizure_model.onnx",
    "--opset", "13"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
else:
    print("Done! model/seizure_model.onnx created.")
    import shutil
    shutil.rmtree(saved_path, ignore_errors=True)
    print(f"Size: {os.path.getsize('model/seizure_model.onnx') / 1024:.1f} KB")
