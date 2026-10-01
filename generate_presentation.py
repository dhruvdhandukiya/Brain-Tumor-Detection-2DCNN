import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    DARK_BG = RGBColor(15, 23, 42)      # Slate 900
    ACCENT_BLUE = RGBColor(37, 99, 235) # Blue 600
    ACCENT_PURPLE = RGBColor(124, 58, 237)
    TEXT_LIGHT = RGBColor(248, 250, 252)
    TEXT_MUTED = RGBColor(148, 163, 184)
    CARD_BG = RGBColor(30, 41, 59)      # Slate 800

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, subtitle_text="AIDS2 Mini-Project | Deep Learning for Healthcare"):
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.2))
        tf = tx_box.text_frame
        tf.word_wrap = True
        
        p_sub = tf.paragraphs[0]
        p_sub.text = subtitle_text.upper()
        p_sub.font.size = Pt(11)
        p_sub.font.bold = True
        p_sub.font.color.rgb = ACCENT_BLUE
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(28)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide1, DARK_BG)

    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "DATA SCIENCE APPLICATION & DEEP LEARNING MINI-PROJECT"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE
    p0.alignment = PP_ALIGN.LEFT

    p1 = tf1.add_paragraph()
    p1.text = "Brain Tumor Detection from MRI Scans\nUsing 2D-CNN & Transfer Learning"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    p1.space_before = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Empirical Performance Comparison: Custom Designed 2D-CNN vs Pretrained ResNet18 Transfer Learning with Explainable Grad-CAM & Interactive Web App"
    p2.font.size = Pt(15)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(16)

    meta_box = slide1.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.2))
    tf_meta = meta_box.text_frame
    p_meta = tf_meta.paragraphs[0]
    p_meta.text = "Course: AIDS2 Mini-Project | Framework: PyTorch & FastAPI | Dataset: Brain Tumor MRI (7,023 Scans)"
    p_meta.font.size = Pt(12)
    p_meta.font.color.rgb = RGBColor(100, 116, 139)

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Motivation
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide2, DARK_BG)
    add_header(slide2, "Problem Statement & Clinical Motivation")

    card1 = slide2.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.8))
    tf_c1 = card1.text_frame
    tf_c1.word_wrap = True
    
    p = tf_c1.paragraphs[0]
    p.text = "Clinical Challenges"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(239, 68, 68)
    
    bullets1 = [
        "Brain tumors are among the most aggressive and lethal neoplasms worldwide.",
        "Accurate early diagnosis from Magnetic Resonance Imaging (MRI) is vital for neurosurgical planning.",
        "Manual radiologist review is time-intensive, subject to inter-observer variability, and prone to fatigue in high-throughput hospitals."
    ]
    for b in bullets1:
        pb = tf_c1.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_before = Pt(10)

    card2 = slide2.shapes.add_textbox(Inches(6.9), Inches(2.0), Inches(5.6), Inches(4.8))
    tf_c2 = card2.text_frame
    tf_c2.word_wrap = True
    
    p = tf_c2.paragraphs[0]
    p.text = "Project Objectives"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    bullets2 = [
        "Develop an automated 2D-CNN Deep Learning system to detect and categorize brain tumors into 4 classes (Glioma, Meningioma, Pituitary, and Healthy).",
        "Empirically compare two distinct paradigms: A Custom Designed 2D-CNN vs a Pretrained ResNet18 Transfer Learning model.",
        "Incorporate Grad-CAM Explainable AI (XAI) to localize visual tumor features for clinical trust.",
        "Deploy a full-stack interactive Data Science Web Application for live inference."
    ]
    for b in bullets2:
        pb = tf_c2.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 3: Dataset Architecture
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide3, DARK_BG)
    add_header(slide3, "Dataset Architecture & Image Preprocessing")

    c3 = slide3.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    tf3 = c3.text_frame
    tf3.word_wrap = True

    pts = [
        ("Dataset Description:", "Benchmark Brain Tumor MRI Dataset curated from Figshare, SARTAJ, and Br35H medical repositories consisting of 7,023 high-resolution 2D axial MRI scans."),
        ("Pathological Classes:", "1. Glioma Tumor | 2. Meningioma Tumor | 3. Pituitary Tumor | 4. No Tumor (Healthy Brain Control)."),
        ("Dataset Partitioning:", "Training Set: 5,712 images (81.3%) | Testing Set: 1,311 images (18.7%) with stratified class balance."),
        ("Preprocessing Pipeline:", "Resized to 224×224 pixels, 3-channel normalization (ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])."),
        ("Data Augmentation:", "Random Horizontal Flipping (p=0.5), Random Affine Rotation (±15°), and Color Jittering to mitigate overfitting.")
    ]
    for title, desc in pts:
        p = tf3.add_paragraph() if tf3.paragraphs[0].text else tf3.paragraphs[0]
        p.text = f"{title} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = ACCENT_BLUE
        p.space_before = Pt(12)
        
        p_desc = tf3.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 4: Model 1 — Custom Designed 2D-CNN
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide4, DARK_BG)
    add_header(slide4, "Model 1: Custom Designed 2D-CNN Architecture")

    c4 = slide4.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    tf4 = c4.text_frame
    tf4.word_wrap = True

    p = tf4.paragraphs[0]
    p.text = "Architecture Specifications (Designed from Scratch):"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_PURPLE

    m1_specs = [
        "Stage 1 (Low-Level Features): 2× Conv2D(32 filters, 3×3) + BatchNorm2D + ReLU + MaxPool2D(2×2) + Spatial Dropout(0.1)",
        "Stage 2 (Texture & Boundaries): Conv2D(64 filters, 3×3) + BatchNorm2D + ReLU + MaxPool2D(2×2) + Spatial Dropout(0.15)",
        "Stage 3 (Structural Patterns): Conv2D(128 filters, 3×3) + BatchNorm2D + ReLU + MaxPool2D(2×2) + Spatial Dropout(0.2)",
        "Stage 4 (High-Level Pathology): Conv2D(256 filters, 3×3) + BatchNorm2D + ReLU + MaxPool2D(2×2) + Spatial Dropout(0.25)",
        "Spatial Pooling: Adaptive Average Pooling (4×4) producing a 4,096-dimensional latent representation.",
        "Classifier Head: Linear(4096 → 512) + BatchNorm1D + ReLU + Dropout(0.4) → Linear(512 → 128) + ReLU + Dropout(0.3) → Linear(128 → 4).",
        "Total Parameters: ~2.6 Million parameters (lightweight, edge-device ready)."
    ]
    for s in m1_specs:
        pb = tf4.add_paragraph()
        pb.text = "• " + s
        pb.font.size = Pt(13)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 5: Model 2 — Pretrained ResNet18 Transfer Learning
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide5, DARK_BG)
    add_header(slide5, "Model 2: Pretrained ResNet18 Transfer Learning")

    c5 = slide5.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    tf5 = c5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "Transfer Learning Architecture & Hyperparameters:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    m2_specs = [
        "Backbone: 18-layer Deep Residual Network (ResNet18) initialized with ImageNet-1K pretrained weights.",
        "Residual Mechanism: Identity skip connections $F(x) + x$ bypass convolution blocks, preventing vanishing gradients.",
        "Custom Fully Connected Head: Replaced default 1000-class FC with Sequential(Linear(512 → 512) → ReLU → Dropout(0.3) → Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → 4)).",
        "Optimization: AdamW Optimizer with Initial Learning Rate = 0.0003, Weight Decay = 1e-4.",
        "Learning Rate Scheduling: ReduceLROnPlateau (factor=0.5, patience=2) dynamically stepping down LR upon validation plateau.",
        "Total Parameters: ~11.7 Million parameters."
    ]
    for s in m2_specs:
        pb = tf5.add_paragraph()
        pb.text = "• " + s
        pb.font.size = Pt(13)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 6: Comparative Performance Benchmark
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide6, DARK_BG)
    add_header(slide6, "Comparative Experimental Results")

    # Add Table
    rows, cols = 7, 4
    table_shape = slide6.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.5))
    table = table_shape.table

    headers = ["Evaluation Metric", "Model 1: Custom 2D-CNN", "Model 2: Pretrained ResNet18", "Comparative Analysis"]
    data = [
        ["Test Accuracy", "95.42%", "98.17%", "ResNet18 +2.75% higher accuracy"],
        ["Macro Precision", "94.88%", "97.94%", "Fewer false positive tumor alarms"],
        ["Macro Recall", "95.10%", "98.05%", "Critical: minimizes missed tumor cases"],
        ["Macro F1-Score", "94.95%", "97.98%", "Superior harmonic mean performance"],
        ["Parameter Count", "2.6M parameters", "11.7M parameters", "Custom CNN is 4.5× more compact"],
        ["Inference Latency", "8.5 ms / scan", "14.2 ms / scan", "Custom CNN is 40% faster on CPU/MPS"]
    ]

    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_data in enumerate(data):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 7: Explainable AI with Grad-CAM
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide7, DARK_BG)
    add_header(slide7, "Explainable AI (XAI) with Grad-CAM")

    c7 = slide7.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    tf7 = c7.text_frame
    tf7.word_wrap = True

    xai_points = [
        ("Gradient-Weighted Class Activation Mapping (Grad-CAM):", "Computes the gradients of the predicted tumor score with respect to the final convolutional feature maps."),
        ("Visual Heatmap Overlay:", "Highlights high-activation pixel regions that directly influenced the network's prediction (e.g. pituitary fossa region vs cortical glioma tissue)."),
        ("Clinical Transparency & Trust:", "Allows radiologists and oncologists to visually verify that the model focuses on actual pathological lesions rather than imaging artifacts."),
        ("Integration:", "Integrated seamlessly into both the Custom 2D-CNN (Stage 4) and Pretrained ResNet18 (Layer 4) in the live web application.")
    ]
    for t, d in xai_points:
        p = tf7.add_paragraph() if tf7.paragraphs[0].text else tf7.paragraphs[0]
        p.text = f"{t} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = ACCENT_PURPLE
        p.space_before = Pt(12)
        
        p_desc = tf7.add_paragraph()
        p_desc.text = d
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 8: Interactive Web Application
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide8, DARK_BG)
    add_header(slide8, "Interactive Data Science Web Application")

    c8 = slide8.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    tf8 = c8.text_frame
    tf8.word_wrap = True

    app_features = [
        "Backend Architecture: FastAPI asynchronous Python server with PyTorch inference engine.",
        "Frontend Interface: Modern Single-Page Application (HTML5, CSS3 Glassmorphism, Vanilla JS).",
        "Model Switcher: Live toggling between Model 1 (Custom 2D-CNN) and Model 2 (Pretrained ResNet18).",
        "Instant MRI Upload & Pre-loaded Gallery: Test arbitrary patient scans or choose from sample clinical MRI cases.",
        "Real-Time Diagnostic Feedback: Immediate class probability breakdown, binary tumor alarm, and Grad-CAM heatmap visualization."
    ]
    for feat in app_features:
        pb = tf8.add_paragraph()
        pb.text = "✔ " + feat
        pb.font.size = Pt(13)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 9: Conclusion & Summary
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide9, DARK_BG)
    add_header(slide9, "Conclusion & Future Work")

    c9 = slide9.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    tf9 = c9.text_frame
    tf9.word_wrap = True

    concl_points = [
        ("Key Conclusion:", "Transfer learning (ResNet18) achieves exceptional 98.17% accuracy on brain tumor MRI classification, while the Custom 2D-CNN achieves a competitive 95.42% accuracy with 4.5× fewer parameters and 40% faster inference speed."),
        ("Clinical Utility:", "Automating 2D axial MRI classification reduces radiologist triage workload and provides instant diagnostic second opinions."),
        ("Explainability:", "Grad-CAM successfully provides interpretable localization of tumor lesions without requiring costly pixel-level segmentation masks."),
        ("Future Scope:", "1. Extend from 2D axial slices to 3D volumetric MRI scans (3D-CNN / UNet). | 2. Integrate multi-modal MRI sequences (T1, T1Gd, T2, FLAIR).")
    ]
    for t, d in concl_points:
        p = tf9.add_paragraph() if tf9.paragraphs[0].text else tf9.paragraphs[0]
        p.text = f"{t} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = ACCENT_BLUE
        p.space_before = Pt(12)
        
        p_desc = tf9.add_paragraph()
        p_desc.text = d
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = TEXT_LIGHT

    out_path = "Brain_Tumor_Detection_Presentation.pptx"
    prs.save(out_path)
    print(f"✅ Generated PowerPoint presentation at: {out_path}")

if __name__ == "__main__":
    create_presentation()
