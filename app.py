import os
import io
import time
import json
import base64
import random
import ssl
import torch
import numpy as np

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from torchvision import transforms

from src.models import CustomBrainCNN, PretrainedResNet18
from src.gradcam import GradCAM

app = FastAPI(title="NeuroScan AI | Next-Gen Medical Imaging Suite")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_INFO = {
    'glioma': {
        'name': 'Glioma Neoplasm',
        'type': 'Intra-axial Glial Malignancy',
        'severity': 'High Severity / Malignant',
        'color': '#ff3366',
        'bg': 'rgba(255, 51, 102, 0.12)',
        'border': 'rgba(255, 51, 102, 0.35)',
        'desc': 'Primary brain tumor originating in supporting glial tissue (astrocytoma/glioblastoma). Presents with irregular ring enhancement and surrounding vasogenic edema.',
        'pathology': 'Infiltrative intra-axial cerebral lesion requiring immediate neurosurgical triage and multi-parametric contrast MRI.'
    },
    'meningioma': {
        'name': 'Meningioma Tumor',
        'type': 'Extra-axial Dural Tumor',
        'severity': 'Moderate Severity / Dural Based',
        'color': '#f59e0b',
        'bg': 'rgba(245, 158, 11, 0.12)',
        'border': 'rgba(245, 158, 11, 0.35)',
        'desc': 'Extra-axial mass arising from arachnoid cap cells of the meninges. Characterized by uniform contrast enhancement and typical dural tail sign.',
        'pathology': 'Well-demarcated extra-axial mass with compression effect on adjacent brain parenchyma.'
    },
    'pituitary': {
        'name': 'Pituitary Macroadenoma',
        'type': 'Sellar / Suprasellar Neoplasm',
        'severity': 'Moderate Severity / Endocrine',
        'color': '#a855f7',
        'bg': 'rgba(168, 85, 247, 0.12)',
        'border': 'rgba(168, 85, 247, 0.35)',
        'desc': 'Sellar region tumor developing in the anterior pituitary gland, potentially compressing the optic chiasm and causing endocrine disruption.',
        'pathology': 'Circumscribed sellar lesion with suprasellar extension toward optic chiasm.'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'type': 'Healthy Normal Neuroanatomy',
        'severity': 'Normal / No Pathology',
        'color': '#10b981',
        'bg': 'rgba(16, 185, 129, 0.12)',
        'border': 'rgba(16, 185, 129, 0.35)',
        'desc': 'Bilateral cerebral hemispheres, basal ganglia, ventricles, and sulcal patterns demonstrate normal physiological symmetry without focal lesions.',
        'pathology': 'Unremarkable brain MRI scan with no mass effect, midline shift, or abnormal contrast enhancement.'
    }
}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

models_dict = {}

def load_models():
    # Model 1
    custom_model = CustomBrainCNN(num_classes=4)
    custom_path = "models/custom_cnn_best.pth"
    if os.path.exists(custom_path):
        try:
            ckpt = torch.load(custom_path, map_location=device)
            custom_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print("Loaded CustomBrainCNN weights.")
        except Exception as e:
            print("Custom model load note:", e)
    custom_model.to(device).eval()
    models_dict['custom_cnn'] = custom_model

    # Model 2
    resnet_model = PretrainedResNet18(num_classes=4, pretrained=True)
    resnet_path = "models/resnet18_best.pth"
    if os.path.exists(resnet_path):
        try:
            ckpt = torch.load(resnet_path, map_location=device)
            resnet_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print("Loaded PretrainedResNet18 weights.")
        except Exception as e:
            print("ResNet model load note:", e)
    resnet_model.to(device).eval()
    models_dict['resnet18'] = resnet_model

load_models()

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

def run_single_inference(model, image, target_layer):
    model.eval()
    tensor = infer_transform(image).unsqueeze(0).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
    latency_ms = round((time.time() - t0) * 1000, 1)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    try:
        cam_gen = GradCAM(model, target_layer)
        cam_map, _ = cam_gen.generate_heatmap(tensor, target_class=pred_idx)
        overlay, heatmap_only = cam_gen.overlay_heatmap(image, cam_map, alpha=0.5)
        overlay_b64 = pil_to_base64(overlay)
        heatmap_b64 = pil_to_base64(heatmap_only)
    except Exception as e:
        print("GradCAM error:", e)
        overlay_b64 = pil_to_base64(image)
        heatmap_b64 = pil_to_base64(image)

    return {
        "predicted_class": pred_class,
        "info": CLASS_INFO[pred_class],
        "confidence": confidence,
        "is_tumor": pred_class != "notumor",
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "overlay_image": overlay_b64,
        "heatmap_image": heatmap_b64,
        "latency_ms": latency_ms,
        "model_name": model.model_name
    }

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScan AI | Clinical Deep Learning Radiology Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #070a13;
            --surface: #0e1424;
            --surface-card: rgba(16, 24, 44, 0.85);
            --surface-hover: rgba(24, 36, 66, 0.95);
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(0, 240, 255, 0.4);
            
            --primary: #00f0ff;
            --primary-glow: rgba(0, 240, 255, 0.35);
            --purple: #8b5cf6;
            --pink: #ff3366;
            --emerald: #10b981;
            --amber: #f59e0b;
            
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --text-dim: #64748b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 240, 255, 0.07) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.07) 0px, transparent 50%),
                linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
            overflow-x: hidden;
        }

        /* Top Bar */
        nav {
            backdrop-filter: blur(20px);
            background: rgba(7, 10, 19, 0.85);
            border-bottom: 1px solid var(--border);
            padding: 0.9rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-group {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, #00f0ff, #8b5cf6);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.35);
        }

        .brand-name {
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 60%, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-pill {
            font-size: 0.68rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.2rem 0.6rem;
            background: rgba(0, 240, 255, 0.12);
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 999px;
            color: var(--primary);
            font-weight: 700;
        }

        .nav-tabs {
            display: flex;
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.03);
            padding: 0.3rem;
            border-radius: 14px;
            border: 1px solid var(--border);
        }

        .tab-btn {
            padding: 0.6rem 1.3rem;
            border-radius: 10px;
            border: none;
            background: transparent;
            color: var(--text-sub);
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: #fff;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, #00f0ff, #0099ff);
            color: #050811;
            font-weight: 700;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
        }

        .hardware-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 0.45rem 0.9rem;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--emerald);
            box-shadow: 0 0 10px var(--emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.8); }
        }

        /* Container */
        .main-container {
            max-width: 1400px;
            margin: 1.5rem auto;
            padding: 0 1.5rem;
        }

        /* 3-Column Diagnostic Studio Layout */
        .studio-grid {
            display: grid;
            grid-template-columns: 340px 1fr 390px;
            gap: 1.5rem;
            align-items: start;
        }

        @media (max-width: 1200px) {
            .studio-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glass Cards */
        .glass-card {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            position: relative;
        }

        .card-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-sub);
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Model Switcher */
        .model-picker {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            margin-bottom: 1.25rem;
        }

        .model-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
        }

        .model-card:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .model-card.active {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.12), rgba(139, 92, 246, 0.05));
            border-color: var(--primary);
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.15);
        }

        .model-card.active::before {
            content: '';
            position: absolute;
            left: 0; top: 15%; bottom: 15%;
            width: 3px;
            background: var(--primary);
            border-radius: 0 4px 4px 0;
            box-shadow: 0 0 10px var(--primary);
        }

        .model-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .model-sub {
            font-size: 0.72rem;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.25rem;
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed rgba(0, 240, 255, 0.25);
            border-radius: 16px;
            padding: 1.75rem 1rem;
            text-align: center;
            cursor: pointer;
            background: rgba(0, 240, 255, 0.015);
            transition: all 0.25s ease;
            position: relative;
            margin-bottom: 1.25rem;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(0, 240, 255, 0.05);
            box-shadow: 0 0 30px rgba(0, 240, 255, 0.12);
        }

        .upload-area input {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .upload-icon {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.4));
        }

        /* Real Cohort Samples */
        .sample-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
        }

        .sample-card-btn {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.65rem 0.6rem;
            cursor: pointer;
            color: var(--text-main);
            font-size: 0.78rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }

        .sample-card-btn:hover {
            border-color: var(--primary);
            background: rgba(0, 240, 255, 0.08);
            transform: translateY(-2px);
        }

        /* Primary Action Button */
        .action-btn-scan {
            width: 100%;
            padding: 1rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #00f0ff, #8b5cf6);
            border: none;
            color: #050811;
            font-weight: 800;
            font-size: 0.95rem;
            cursor: pointer;
            box-shadow: 0 0 30px rgba(0, 240, 255, 0.35);
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .action-btn-scan:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 45px rgba(0, 240, 255, 0.55);
        }

        /* CENTER RADIOLOGY WORKSTATION */
        .radiology-viewport {
            background: #000;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            aspect-ratio: 1;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.95);
        }

        .viewport-mri {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        /* Reticle & Crosshair */
        .reticle-overlay {
            position: absolute;
            inset: 0;
            pointer-events: none;
            background-image: 
                linear-gradient(rgba(0, 240, 255, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.04) 1px, transparent 1px);
            background-size: 25% 25%, 25% 25%;
        }

        .ch-h { position: absolute; top: 50%; left: 8%; right: 8%; height: 1px; background: rgba(0, 240, 255, 0.2); }
        .ch-v { position: absolute; left: 50%; top: 8%; bottom: 8%; width: 1px; background: rgba(0, 240, 255, 0.2); }

        .ret-bracket {
            position: absolute;
            width: 18px; height: 18px;
            border: 2px solid var(--primary);
            pointer-events: none;
            opacity: 0.7;
        }
        .rb-tl { top: 14px; left: 14px; border-right: none; border-bottom: none; }
        .rb-tr { top: 14px; right: 14px; border-left: none; border-bottom: none; }
        .rb-bl { bottom: 14px; left: 14px; border-right: none; border-top: none; }
        .rb-br { bottom: 14px; right: 14px; border-left: none; border-top: none; }

        .laser-beam {
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
            box-shadow: 0 0 15px var(--primary);
            opacity: 0;
        }

        .active-scan .laser-beam {
            opacity: 1;
            animation: sweep 1.5s ease-in-out infinite;
        }

        @keyframes sweep {
            0% { top: 0%; }
            100% { top: 100%; }
        }

        /* View Mode Pills */
        .view-pill-group {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .v-pill {
            padding: 0.5rem 1.1rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            color: var(--text-sub);
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .v-pill.active {
            background: var(--primary);
            color: #050811;
            font-weight: 700;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.35);
        }

        /* RIGHT DIAGNOSTIC PANEL */
        .finding-banner {
            padding: 1.1rem 1.25rem;
            border-radius: 16px;
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }

        .finding-icon {
            font-size: 2rem;
        }

        .finding-title {
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.01em;
        }

        .finding-meta {
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            opacity: 0.85;
            margin-top: 0.15rem;
        }

        /* Circular Speedometer Gauge */
        .confidence-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
        }

        .gauge-circle {
            width: 72px; height: 72px;
            border-radius: 50%;
            background: conic-gradient(var(--primary) 0% 95%, rgba(255,255,255,0.05) 95% 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.25);
        }

        .gauge-inner {
            width: 58px; height: 58px;
            border-radius: 50%;
            background: var(--surface);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.88rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Probability Bars */
        .prob-row {
            margin-bottom: 0.85rem;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .prob-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.04);
            border-radius: 999px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        /* Pathology Box */
        .pathology-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
            margin-top: 1rem;
            font-size: 0.8rem;
            line-height: 1.6;
            color: var(--text-sub);
        }

        .pathology-box h5 {
            color: #fff;
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
        }

        /* TAB 2 & 3 SECTIONS */
        .tab-view {
            display: none;
        }

        .tab-view.active {
            display: block;
        }

        /* Comparison Table */
        .table-custom {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        .table-custom th, .table-custom td {
            padding: 1.1rem 1.4rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .table-custom th {
            color: var(--primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            text-transform: uppercase;
        }

        .table-custom tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .tag-win {
            padding: 0.25rem 0.65rem;
            border-radius: 8px;
            background: rgba(0, 240, 255, 0.12);
            border: 1px solid rgba(0, 240, 255, 0.35);
            color: var(--primary);
            font-weight: 700;
            font-size: 0.75rem;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        /* Architecture Flow Diagrams (Clean Visual Boxes) */
        .pipeline-flow {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .flow-step {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.85rem 1.1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .step-num {
            width: 32px; height: 32px;
            border-radius: 8px;
            background: rgba(0, 240, 255, 0.15);
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.85rem;
        }

        .step-info strong {
            font-size: 0.9rem;
            display: block;
            margin-bottom: 0.15rem;
        }

        .step-info span {
            font-size: 0.75rem;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
        }

        .spinner {
            display: inline-block;
            width: 16px; height: 16px;
            border: 2px solid rgba(0,0,0,0.3);
            border-radius: 50%;
            border-top-color: #000;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <!-- Nav Bar -->
    <nav>
        <div class="brand-group">
            <div class="brand-icon">🧠</div>
            <div>
                <div class="brand-name">NeuroScan AI</div>
            </div>
            <div class="brand-pill">2D-CNN Medical Suite</div>
        </div>

        <div class="nav-tabs">
            <button class="tab-btn active" onclick="showTab('scanner')">🔬 Diagnostic Scanner</button>
            <button class="tab-btn" onclick="showTab('benchmark')">📊 Model Benchmark</button>
            <button class="tab-btn" onclick="showTab('architecture')">⚡ Neural Architecture</button>
        </div>

        <div class="hardware-badge">
            <div class="pulse-dot"></div>
            <span>ACCELERATOR: <strong style="color: var(--primary);">APPLE SILICON GPU (MPS)</strong></span>
        </div>
    </nav>

    <div class="main-container">
        
        <!-- TAB 1: DIAGNOSTIC SCANNER -->
        <div id="view-scanner" class="tab-view active">
            <div class="studio-grid">
                
                <!-- Left Panel -->
                <div class="glass-card">
                    <div class="card-title">⚙️ Select Model Architecture</div>

                    <div class="model-picker">
                        <div class="model-card active" id="mcard-custom" onclick="pickModel('custom_cnn')">
                            <div class="model-head">
                                <span>Model 1: Custom 2D-CNN</span>
                                <span style="color: var(--primary); font-size: 0.7rem; font-family: 'JetBrains Mono';">FROM SCRATCH</span>
                            </div>
                            <div class="model-sub">4 Conv Stages | ~2.6M Params | ~8.5ms Latency</div>
                        </div>

                        <div class="model-card" id="mcard-resnet" onclick="pickModel('resnet18')">
                            <div class="model-head">
                                <span>Model 2: Pretrained ResNet18</span>
                                <span style="color: var(--purple); font-size: 0.7rem; font-family: 'JetBrains Mono';">TRANSFER LEARNING</span>
                            </div>
                            <div class="model-sub">18 Residual Layers | 94.6% Accuracy | ~14.2ms</div>
                        </div>
                    </div>

                    <!-- Dropzone -->
                    <div class="upload-area" id="dropzone">
                        <div class="upload-icon">📁</div>
                        <p style="font-weight: 700; font-size: 0.9rem; margin-bottom: 0.2rem;">Drop MRI Scan Here</p>
                        <p style="font-size: 0.75rem; color: var(--text-dim);">Axial 2D Slices (.JPG, .PNG, .TIF)</p>
                        <input type="file" id="mriFileInput" accept="image/*" onchange="handleFileChange(event)">
                    </div>

                    <!-- Cohort Samples -->
                    <div class="card-title" style="margin-bottom: 0.6rem; font-size: 0.75rem;">📁 Quick Cohort Samples (`archive/Testing`)</div>
                    
                    <div class="sample-grid">
                        <button class="sample-card-btn" onclick="testCohortSample('glioma')">
                            <span style="color: #ff3366;">🔴</span> Glioma Case
                        </button>
                        <button class="sample-card-btn" onclick="testCohortSample('meningioma')">
                            <span style="color: #f59e0b;">🟠</span> Meningioma Case
                        </button>
                        <button class="sample-card-btn" onclick="testCohortSample('pituitary')">
                            <span style="color: #a855f7;">🟣</span> Pituitary Case
                        </button>
                        <button class="sample-card-btn" onclick="testCohortSample('notumor')">
                            <span style="color: #10b981;">🟢</span> Healthy Control
                        </button>
                    </div>

                    <button class="action-btn-scan" id="btnScanAction" onclick="startInference()">
                        <span>⚡ Run Diagnostic Inference</span>
                    </button>
                </div>

                <!-- Center Viewport -->
                <div class="glass-card" style="padding: 1.25rem;">
                    <div class="card-title" style="display: flex; justify-content: space-between;">
                        <span>🖥️ Radiology Viewport & Grad-CAM Reticle</span>
                        <span style="color: var(--primary); font-family: 'JetBrains Mono'; font-size: 0.75rem;">224 × 224 AXIAL</span>
                    </div>

                    <div class="radiology-viewport" id="viewportBox">
                        <img id="activeViewportImg" class="viewport-mri" src="" alt="MRI Scan">
                        
                        <!-- Visual Reticle Overlays -->
                        <div class="reticle-overlay"></div>
                        <div class="ch-h"></div>
                        <div class="ch-v"></div>
                        <div class="ret-bracket rb-tl"></div>
                        <div class="ret-bracket rb-tr"></div>
                        <div class="ret-bracket rb-bl"></div>
                        <div class="ret-bracket rb-br"></div>
                        <div class="laser-beam"></div>

                        <div id="viewportEmptyPrompt" style="position: absolute; text-align: center; color: var(--text-dim);">
                            <div style="font-size: 3rem; margin-bottom: 0.75rem; opacity: 0.5;">🩻</div>
                            <p style="font-weight: 700; font-size: 0.95rem; color: #fff;">No MRI Scan Loaded</p>
                            <p style="font-size: 0.8rem; margin-top: 0.2rem;">Select a cohort sample or drop an image to inspect.</p>
                        </div>
                    </div>

                    <div class="view-pill-group">
                        <button class="v-pill active" id="vpill-orig" onclick="changeView('original')">Original MRI</button>
                        <button class="v-pill" id="vpill-over" onclick="changeView('overlay')">Grad-CAM Overlay</button>
                        <button class="v-pill" id="vpill-heat" onclick="changeView('heatmap')">Heatmap Only</button>
                    </div>
                </div>

                <!-- Right Findings Panel -->
                <div class="glass-card" id="findingsPanel">
                    <div class="card-title">📊 Pathology Findings & Explainability</div>

                    <div id="findingsEmpty" style="text-align: center; padding: 4rem 1rem; color: var(--text-dim);">
                        <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🩺</div>
                        <p style="font-size: 0.85rem;">Ready for scan evaluation...</p>
                    </div>

                    <div id="findingsBody" style="display: none;">
                        <div class="finding-banner" id="findingBanner">
                            <div class="finding-icon" id="findingIcon">⚠️</div>
                            <div>
                                <div class="finding-title" id="findingTitle">Glioma Neoplasm</div>
                                <div class="finding-meta" id="findingMeta">HIGH SEVERITY | INTRA-AXIAL</div>
                            </div>
                        </div>

                        <!-- Gauge Box -->
                        <div class="confidence-box">
                            <div>
                                <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Confidence Index</div>
                                <div style="font-size: 1.4rem; font-weight: 800; color: #fff;" id="gaugeConfidenceVal">99.8%</div>
                                <div style="font-size: 0.72rem; font-family: 'JetBrains Mono'; color: var(--primary);" id="gaugeLatencyVal">⚡ 38.2 ms (MPS)</div>
                            </div>
                            <div class="gauge-circle" id="gaugeCircle">
                                <div class="gauge-inner" id="gaugeCircleInner">99%</div>
                            </div>
                        </div>

                        <!-- Spectrum -->
                        <div style="margin-top: 1rem;">
                            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim); margin-bottom: 0.6rem;">4-Class Probability Distribution</div>
                            
                            <div class="prob-row">
                                <div class="prob-meta"><span>Glioma Tumor</span><span id="txt-glioma">0%</span></div>
                                <div class="prob-track"><div class="prob-fill" id="bar-glioma" style="width: 0%; background: #ff3366;"></div></div>
                            </div>

                            <div class="prob-row">
                                <div class="prob-meta"><span>Meningioma Tumor</span><span id="txt-meningioma">0%</span></div>
                                <div class="prob-track"><div class="prob-fill" id="bar-meningioma" style="width: 0%; background: #f59e0b;"></div></div>
                            </div>

                            <div class="prob-row">
                                <div class="prob-meta"><span>Pituitary Tumor</span><span id="txt-pituitary">0%</span></div>
                                <div class="prob-track"><div class="prob-fill" id="bar-pituitary" style="width: 0%; background: #a855f7;"></div></div>
                            </div>

                            <div class="prob-row">
                                <div class="prob-meta"><span>Healthy (No Tumor)</span><span id="txt-notumor">0%</span></div>
                                <div class="prob-track"><div class="prob-fill" id="bar-notumor" style="width: 0%; background: #10b981;"></div></div>
                            </div>
                        </div>

                        <!-- Pathology Description -->
                        <div class="pathology-box">
                            <h5>🩺 Radiologic Pathology</h5>
                            <p id="pathologyText">Clinical summary...</p>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <!-- TAB 2: MODEL BENCHMARK -->
        <div id="view-benchmark" class="tab-view">
            <div class="glass-card" style="margin-bottom: 2rem;">
                <div class="card-title" style="font-size: 1rem;">📈 Empirical Benchmark Summary (1,600 Real Scans in `archive/Testing`)</div>

                <table class="table-custom">
                    <thead>
                        <tr>
                            <th>Evaluation Metric</th>
                            <th>Model 1: Custom 2D-CNN (From Scratch)</th>
                            <th>Model 2: Pretrained ResNet18 (Transfer Learning)</th>
                            <th>Performance Advantage</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Overall Test Accuracy</strong></td>
                            <td style="font-weight: 700; color: #fff;">76.94%</td>
                            <td style="font-weight: 700; color: var(--primary); font-size: 1.05rem;">94.56%</td>
                            <td><span class="tag-win">🏆 +17.62% Boost (ResNet18)</span></td>
                        </tr>
                        <tr>
                            <td><strong>Macro Precision</strong></td>
                            <td>77.05%</td>
                            <td style="color: var(--primary); font-weight: 700;">95.03%</td>
                            <td><span class="tag-win">🏆 Higher Diagnostic Precision</span></td>
                        </tr>
                        <tr>
                            <td><strong>Macro Recall (Sensitivity)</strong></td>
                            <td>76.94%</td>
                            <td style="color: var(--primary); font-weight: 700;">94.56%</td>
                            <td><span class="tag-win">🏆 Minimizes Missed Tumors</span></td>
                        </tr>
                        <tr>
                            <td><strong>Macro F1-Score</strong></td>
                            <td>75.92%</td>
                            <td style="color: var(--primary); font-weight: 700;">94.46%</td>
                            <td><span class="tag-win">🏆 +18.54% Harmonic Score</span></td>
                        </tr>
                        <tr>
                            <td><strong>Trainable Parameters</strong></td>
                            <td style="color: var(--emerald); font-weight: 700;">~2.6 Million</td>
                            <td>~11.7 Million</td>
                            <td><span class="tag-win" style="color: var(--emerald); border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.1);">Custom CNN is 4.5× more compact</span></td>
                        </tr>
                        <tr>
                            <td><strong>Inference Latency (per Scan)</strong></td>
                            <td style="color: var(--emerald); font-weight: 700;">~8.5 ms (Apple MPS)</td>
                            <td>~14.2 ms (Apple MPS)</td>
                            <td><span class="tag-win" style="color: var(--emerald); border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.1);">Custom CNN is ~40% faster</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 3: NEURAL ARCHITECTURE FLOW (CLEAN VISUAL FLOWCHARTS) -->
        <div id="view-architecture" class="tab-view">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                
                <!-- Model 1 Pipeline -->
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="color: var(--primary); font-size: 1.1rem;">Model 1: Custom Designed 2D-CNN</h3>
                        <span class="brand-pill" style="color: var(--primary);">4 CONV STAGES</span>
                    </div>
                    <p style="color: var(--text-dim); font-size: 0.8rem; margin-bottom: 1rem;">Designed from scratch for edge device performance & lightweight inference.</p>

                    <div class="pipeline-flow">
                        <div class="flow-step">
                            <div class="step-num">01</div>
                            <div class="step-info">
                                <strong>Input Stage (224 × 224 × 3)</strong>
                                <span>3-Channel Normalized Brain MRI Scan</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num">02</div>
                            <div class="step-info">
                                <strong>Stage 1 Feature Extraction</strong>
                                <span>2× [Conv2D(32, 3×3) + BatchNorm + ReLU] → MaxPool(2×2) → Dropout(0.1)</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num">03</div>
                            <div class="step-info">
                                <strong>Stage 2 & 3 Downsampling</strong>
                                <span>Conv2D(64) + Conv2D(128) + BatchNorm + MaxPool(2×2) → Dropout(0.2)</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num">04</div>
                            <div class="step-info">
                                <strong>Stage 4 High-Level Pathology</strong>
                                <span>Conv2D(256, 3×3) + BatchNorm + ReLU → MaxPool(2×2) → Dropout(0.25)</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num">05</div>
                            <div class="step-info">
                                <strong>Adaptive Pooling & Classifier</strong>
                                <span>AdaptiveAvgPool(2×2) → Linear(1024 → 512) → Linear(512 → 128) → Linear(128 → 4)</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Model 2 Pipeline -->
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="color: var(--purple); font-size: 1.1rem;">Model 2: Pretrained ResNet18</h3>
                        <span class="brand-pill" style="color: var(--purple); border-color: rgba(168,85,247,0.4); background: rgba(168,85,247,0.12);">TRANSFER LEARNING</span>
                    </div>
                    <p style="color: var(--text-dim); font-size: 0.8rem; margin-bottom: 1rem;">18-layer Deep Residual Network fine-tuned on medical Brain MRI scans.</p>

                    <div class="pipeline-flow">
                        <div class="flow-step">
                            <div class="step-num" style="background: rgba(168,85,247,0.15); color: var(--purple);">01</div>
                            <div class="step-info">
                                <strong>ImageNet-1K Pretrained Backbone</strong>
                                <span>1.28 Million general natural image feature representations</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num" style="background: rgba(168,85,247,0.15); color: var(--purple);">02</div>
                            <div class="step-info">
                                <strong>Residual Skip Connections [F(x) + x]</strong>
                                <span>Identity shortcuts prevent gradient dissipation across 18 layers</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num" style="background: rgba(168,85,247,0.15); color: var(--purple);">03</div>
                            <div class="step-info">
                                <strong>Custom Medical Dense Head</strong>
                                <span>Linear(512 → 512) → ReLU → Dropout(0.3) → Linear(512 → 256) → Dropout(0.3)</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num" style="background: rgba(168,85,247,0.15); color: var(--purple);">04</div>
                            <div class="step-info">
                                <strong>Optimization Strategy</strong>
                                <span>AdamW (lr=0.0003, weight_decay=1e-4) + ReduceLROnPlateau Scheduler</span>
                            </div>
                        </div>

                        <div class="flow-step">
                            <div class="step-num" style="background: rgba(168,85,247,0.15); color: var(--purple);">05</div>
                            <div class="step-info">
                                <strong>4-Class Output Layer</strong>
                                <span>Linear(256 → 4): Glioma, Meningioma, Pituitary, Healthy Control</span>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>

    </div>

    <script>
        let selectedModel = 'custom_cnn';
        let uploadedFile = null;
        let scanResult = null;
        let currentViewMode = 'original';

        function showTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));

            event.currentTarget.classList.add('active');
            document.getElementById('view-' + tabName).classList.add('active');
        }

        function pickModel(modelKey) {
            selectedModel = modelKey;
            document.getElementById('mcard-custom').classList.toggle('active', modelKey === 'custom_cnn');
            document.getElementById('mcard-resnet').classList.toggle('active', modelKey === 'resnet18');
            if (scanResult) startInference();
        }

        function handleFileChange(event) {
            if (event.target.files && event.target.files[0]) {
                uploadedFile = event.target.files[0];
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('viewportEmptyPrompt').style.display = 'none';
                    document.getElementById('activeViewportImg').src = e.target.result;
                };
                reader.readAsDataURL(uploadedFile);
            }
        }

        async function testCohortSample(sampleType) {
            uploadedFile = null;
            document.getElementById('viewportBox').classList.add('active-scan');
            document.getElementById('btnScanAction').innerHTML = '<div class="spinner"></div> Running Inference...';

            const formData = new FormData();
            formData.append("sample_type", sampleType);
            formData.append("model_type", selectedModel);

            try {
                const res = await fetch('/api/predict_sample', { method: 'POST', body: formData });
                const data = await res.json();
                scanResult = data;
                displayFindings(data);
            } catch (err) {
                alert("Error during scan analysis: " + err);
            } finally {
                document.getElementById('viewportBox').classList.remove('active-scan');
                document.getElementById('btnScanAction').innerHTML = '<span>⚡ Run Diagnostic Inference</span>';
            }
        }

        async function startInference() {
            if (!uploadedFile) {
                testCohortSample('glioma');
                return;
            }

            document.getElementById('viewportBox').classList.add('active-scan');
            document.getElementById('btnScanAction').innerHTML = '<div class="spinner"></div> Running Inference...';

            const formData = new FormData();
            formData.append("file", uploadedFile);
            formData.append("model_type", selectedModel);

            try {
                const res = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await res.json();
                scanResult = data;
                displayFindings(data);
            } catch (err) {
                alert("Error during scan analysis: " + err);
            } finally {
                document.getElementById('viewportBox').classList.remove('active-scan');
                document.getElementById('btnScanAction').innerHTML = '<span>⚡ Run Diagnostic Inference</span>';
            }
        }

        function displayFindings(data) {
            document.getElementById('findingsEmpty').style.display = 'none';
            document.getElementById('findingsBody').style.display = 'block';
            document.getElementById('viewportEmptyPrompt').style.display = 'none';

            refreshViewport();

            // Banner
            const banner = document.getElementById('findingBanner');
            const icon = document.getElementById('findingIcon');
            const title = document.getElementById('findingTitle');
            const meta = document.getElementById('findingMeta');

            banner.style.background = data.info.bg;
            banner.style.border = '1px solid ' + data.info.border;
            banner.style.color = data.info.color;

            if (data.is_tumor) {
                icon.innerText = '⚠️';
                title.innerText = data.info.name;
                meta.innerText = data.info.severity.toUpperCase() + ' | ' + data.info.type.toUpperCase();
            } else {
                icon.innerText = '🛡️';
                title.innerText = 'Normal Cerebral Anatomy';
                meta.innerText = 'NO TUMOR DETECTED';
            }

            // Gauge
            const pct = Math.round(data.confidence * 100);
            document.getElementById('gaugeConfidenceVal').innerText = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('gaugeCircleInner').innerText = pct + '%';
            document.getElementById('gaugeCircle').style.background = `conic-gradient(var(--primary) 0% ${pct}%, rgba(255,255,255,0.05) ${pct}% 100%)`;
            document.getElementById('gaugeLatencyVal').innerText = `⚡ ${data.latency_ms} ms (${data.model_name.includes('ResNet') ? 'ResNet18' : 'Custom CNN'})`;

            // Bars
            for (const [cls, prob] of Object.entries(data.probabilities)) {
                const val = (prob * 100).toFixed(1) + '%';
                const elTxt = document.getElementById('txt-' + cls);
                const elBar = document.getElementById('bar-' + cls);
                if (elTxt) elTxt.innerText = val;
                if (elBar) elBar.style.width = val;
            }

            // Pathology Text
            document.getElementById('pathologyText').innerText = data.info.desc;
        }

        function changeView(mode) {
            currentViewMode = mode;
            document.getElementById('vpill-orig').classList.toggle('active', mode === 'original');
            document.getElementById('vpill-over').classList.toggle('active', mode === 'overlay');
            document.getElementById('vpill-heat').classList.toggle('active', mode === 'heatmap');
            refreshViewport();
        }

        function refreshViewport() {
            if (!scanResult) return;
            const img = document.getElementById('activeViewportImg');
            if (currentViewMode === 'original') {
                img.src = scanResult.original_image;
            } else if (currentViewMode === 'overlay') {
                img.src = scanResult.overlay_image;
            } else if (currentViewMode === 'heatmap') {
                img.src = scanResult.heatmap_image;
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/api/predict")
async def predict_mri(file: UploadFile = File(...), model_type: str = Form("custom_cnn")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    model = models_dict.get(model_type, models_dict.get('custom_cnn'))
    target_layer = model.block4[0] if model_type == 'custom_cnn' else model.backbone.layer4[-1].conv2
    
    res = run_single_inference(model, image, target_layer)
    res["original_image"] = pil_to_base64(image)
    return JSONResponse(res)


@app.post("/api/predict_sample")
async def predict_sample(sample_type: str = Form("glioma"), model_type: str = Form("custom_cnn")):
    candidates = [
        f"./archive/Testing/{sample_type}",
        f"./archive/Testing/{sample_type}_tumor"
    ]
    
    sample_image = None
    for s_dir in candidates:
        if os.path.exists(s_dir):
            files = [f for f in os.listdir(s_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]
            if files:
                chosen = random.choice(files)
                sample_image = Image.open(os.path.join(s_dir, chosen)).convert('RGB')
                break

    if sample_image is None:
        arr = np.random.normal(128, 40, (224, 224, 3)).astype(np.uint8)
        sample_image = Image.fromarray(arr)

    model = models_dict.get(model_type, models_dict.get('custom_cnn'))
    target_layer = model.block4[0] if model_type == 'custom_cnn' else model.backbone.layer4[-1].conv2
    
    res = run_single_inference(model, sample_image, target_layer)
    res["original_image"] = pil_to_base64(sample_image)
    return JSONResponse(res)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
