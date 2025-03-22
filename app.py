import os
import io
import torch
import torchvision
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
CLASS_DICT = {"PIH": 1, "PIE": 2, "spot": 3}
INV_CLASS_DICT = {1: "PIH", 2: "PIE", 3: "spot"}
NUM_CLASSES = len(CLASS_DICT) + 1  # +1 for background
CHECKPOINT_PATH = "checkpoints/model_epoch_20.pth"
DETECTION_THRESHOLD = 0.3

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Initialize the model
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load the model
model = get_model(NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Transforms for input images
transform = T.Compose([
    T.ToTensor(),
])

def draw_predictions(image, prediction, threshold=DETECTION_THRESHOLD):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    
    # Try to get a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Get predictions above threshold
    keep = prediction["scores"] > threshold
    boxes = prediction["boxes"][keep].cpu().numpy()
    labels = prediction["labels"][keep].cpu().numpy()
    scores = prediction["scores"][keep].cpu().numpy()
    
    # Draw each prediction
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        
        # Create label text
        class_name = INV_CLASS_DICT.get(label, "unknown")
        label_text = f"{class_name}: {score:.2f}"
        
        # Draw label background
        text_size = draw.textbbox((0, 0), label_text, font=font)[2:4]
        draw.rectangle([(x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 4, y1)], 
                      fill="red")
        
        # Draw label text
        draw.text((x1 + 2, y1 - text_size[1] - 2), label_text, fill="white", font=font)
    
    return image

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({"status": "healthy", "model": "Faster R-CNN"})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive an image and return predictions."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Read the image
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        
        # Transform the image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        # Process the prediction
        result_image = draw_predictions(image, prediction)
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Return the image with predictions
        return send_file(img_byte_arr, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Endpoint to receive a base64 encoded image and return predictions as base64."""
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "No image provided in request JSON"}), 400
    
    try:
        # Decode base64 image
        base64_data = request.json['image']
        
        # Remove potential data URL prefix
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
                
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Transform the image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        # Process the prediction
        result_image = draw_predictions(image, prediction)
        
        # Convert result image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Return the base64 encoded result
        return jsonify({
            "result_image": img_str,
            "count": int(sum(prediction["scores"] > DETECTION_THRESHOLD)),
            "detection_threshold": DETECTION_THRESHOLD
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5200))
    app.run(host='0.0.0.0', port=port, debug=False)
