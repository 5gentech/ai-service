import insightface
import os
import shutil

# Output folder
target_dir = "models"
os.makedirs(target_dir, exist_ok=True)

print("Downloading SCRFD detector...")
detector = insightface.model_zoo.get_model('scrfd_2.5g_bnkps')
scrfd_path = detector.model_path
shutil.copy(scrfd_path, os.path.join(target_dir, "detection.onnx"))

print("Downloading ArcFace embeddings model...")
arc = insightface.model_zoo.get_model('arcface_r100_v1')
arc_path = arc.model_path
shutil.copy(arc_path, os.path.join(target_dir, "arcface.onnx"))

print("DONE — Both models downloaded successfully!")
print("Saved in ./models/")
