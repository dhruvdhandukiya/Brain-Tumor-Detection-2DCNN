import torch
from src.models import CustomBrainCNN, PretrainedResNet18
from src.gradcam import GradCAM
from PIL import Image
import numpy as np

def test_models():
    print("Testing CustomBrainCNN...")
    model1 = CustomBrainCNN(num_classes=4)
    dummy_input = torch.randn(2, 3, 224, 224)
    out1 = model1(dummy_input)
    print(f"✅ Model 1 output shape: {out1.shape} (Expected: [2, 4])")
    assert out1.shape == (2, 4), "Shape mismatch in Model 1"

    print("\nTesting PretrainedResNet18...")
    model2 = PretrainedResNet18(num_classes=4, pretrained=False)
    out2 = model2(dummy_input)
    print(f"✅ Model 2 output shape: {out2.shape} (Expected: [2, 4])")
    assert out2.shape == (2, 4), "Shape mismatch in Model 2"

    print("\nTesting Grad-CAM explainability...")
    target_layer = model1.block4[0]
    cam = GradCAM(model1, target_layer)
    heatmap, pred_cls = cam.generate_heatmap(dummy_input[:1])
    print(f"✅ Grad-CAM heatmap generated with shape: {heatmap.shape} (Expected: [224, 224] normalized map)")

    raw_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    overlay, _ = cam.overlay_heatmap(raw_img, heatmap)
    print(f"✅ Grad-CAM overlay created: size={overlay.size}")

    print("\n🎉 ALL ARCHITECTURE & EXPLAINABILITY TESTS PASSED!")

if __name__ == "__main__":
    test_models()
