import gradio as gr
import torch
import numpy as np
from PIL import Image

# Load your trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path="runs/train/exp10/weights/best.pt")

# def detect_defects(image):
#     # Perform inference
#     results = model(image)
#     result_img = np.squeeze(results.render())  # Annotated image as NumPy array
#     return Image.fromarray(result_img)
def detect_defects(image):
    # Confidence values
    # 0.05 - Visual debugging
    # 0.2 - Semi-decent initial threshold
    # 0.4–0.6 - Final confident outputs
    # > 0.7 - Low tolerance for false alarms

    model.conf = 0.05  # Set detection confidence threshold on model
    results = model(image)
    print(results.pandas().xyxy[0])
    result_img = np.squeeze(results.render())
    return Image.fromarray(result_img)

# Launch Gradio interface
gr.Interface(fn=detect_defects, 
             inputs=gr.Image(type="pil"), 
             outputs=gr.Image(type="pil")).launch()