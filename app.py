from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cpu")

# ----------------------------
# CLASS LABELS
# ----------------------------
classes = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']

# ----------------------------
# LOAD MODEL (RESNET18)
# ----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load("arthritis_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----------------------------
# PDF REPORT GENERATOR
# ----------------------------
def generate_report(prediction, confidence):
    os.makedirs("static", exist_ok=True)

    file_path = "static/medical_report.pdf"
    c = canvas.Canvas(file_path)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, 800, "Arthritis AI Medical Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Diagnosis: {prediction}")
    c.drawString(50, 740, f"Confidence: {confidence}%")
    c.drawString(50, 720, f"Date: {datetime.now()}")

    c.drawString(50, 680, "This report is generated using AI-based X-ray analysis system.")

    c.save()

    return file_path

# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# PREDICT ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Load image
    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item() * 100

    result = classes[predicted_class]

    # Probabilities for graph
    probabilities = [
        round(probs[0].item() * 100, 2),
        round(probs[1].item() * 100, 2),
        round(probs[2].item() * 100, 2),
        round(probs[3].item() * 100, 2),
        round(probs[4].item() * 100, 2)
    ]

    # Generate PDF report
    pdf_file = generate_report(result, round(confidence, 2))

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        probabilities=probabilities,
        pdf_file=pdf_file
    )

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)