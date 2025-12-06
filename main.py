from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import onnxruntime as ort
import cv2

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load ONNX Models
# -------------------------------
print("Loading models...")

detector_session = ort.InferenceSession("models/detection.onnx", providers=["CPUExecutionProvider"])
arcface_session = ort.InferenceSession("models/arcface.onnx", providers=["CPUExecutionProvider"])

print("Models loaded successfully.")

# -------------------------------
# Helper functions
# -------------------------------
def preprocess_face(img):
    """Resize + normalize for ArcFace"""
    img = cv2.resize(img, (112, 112))
    img = img[:, :, ::-1]  # BGR → RGB
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def detect_face(image):
    """Use SCRFD to detect face bounding box"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    blob = cv2.resize(img, (640, 640))
    blob = blob.transpose(2, 0, 1)[None, ...].astype(np.float32)

    outputs = detector_session.run(None, {"input": blob})
    bboxes = outputs[0]
    scores = outputs[1]

    if len(bboxes) == 0:
        return None

    # Best score bbox
    idx = np.argmax(scores)
    x1, y1, x2, y2 = bboxes[idx]

    # Scale back to original size
    x1 = int(x1 / 640 * w)
    y1 = int(y1 / 640 * h)
    x2 = int(x2 / 640 * w)
    y2 = int(y2 / 640 * h)

    return (x1, y1, x2, y2)


def get_embedding(face_img):
    """Generate ArcFace embedding"""
    preprocessed = preprocess_face(face_img)
    embedding = arcface_session.run(None, {"data": preprocessed})[0][0]
    return embedding.tolist()


# -------------------------------
# API Routes
# -------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    """Return embedding vector for face"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    bbox = detect_face(image)

    if bbox is None:
        return {"error": "No face detected"}

    x1, y1, x2, y2 = bbox
    face = image[y1:y2, x1:x2]

    embedding = get_embedding(face)

    return {
        "success": True,
        "embedding": embedding
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
