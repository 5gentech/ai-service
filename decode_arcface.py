import base64

with open("arcface_base64.txt", "r") as f:
    b64 = f.read()

binary_data = base64.b64decode(b64)

with open("arcface.onnx", "wb") as f:
    f.write(binary_data)

print("arcface.onnx created!")
