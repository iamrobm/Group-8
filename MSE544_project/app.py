import gradio as gr
import torch
import numpy as np
from PIL import Image
import os

# === CONFIG ===
MODEL_DIR = "runs/"  # Put your .pt files here
DEFAULT_CONFIDENCE = 0.10

# === Load available models ===
def get_model_files():
    pt_files = []
    for root, _, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".pt"):
                rel_path = os.path.relpath(os.path.join(root, file), MODEL_DIR)
                pt_files.append(rel_path)
    return pt_files

# === Model cache to avoid reloading ===
loaded_models = {}

def load_model(model_rel_path):
    if model_rel_path not in loaded_models:
        full_path = os.path.join(MODEL_DIR, model_rel_path)
        print(f"Loading model: {model_rel_path}")
        loaded_models[model_rel_path] = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path=full_path
        )
    return loaded_models[model_rel_path]

# === Detection function ===
def detect_defects(image, model_name, conf_threshold):
    model = load_model(model_name)
    model.conf = conf_threshold

    results = model(np.array(image))
    detections = results.pandas().xyxy[0][['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']]
    result_img = np.squeeze(results.render())
    return Image.fromarray(result_img), detections

# === Gradio Interface ===
model_choices = get_model_files()
default_model = 'train/exp10/weights/best.pt' if 'train/exp10/weights/best.pt' in model_choices else (model_choices[0] if model_choices else None)

iface = gr.Interface(
    fn=detect_defects,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=model_choices, label="Choose Model", value=default_model),
        gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=DEFAULT_CONFIDENCE, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(label="Detected Image"),
        gr.Dataframe(label="Detections Table")
    ],
    title="Defect Detection App",
    description="Upload an image, select a model (.pt), and adjust the confidence threshold to detect defects."
)

iface.launch(share=True)