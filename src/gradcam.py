import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

class GradCAM:
    """
    Grad-CAM implementation to compute visual activation heatmaps
    highlighting regions indicative of brain tumors in MRI scans.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

    def overlay_heatmap(self, original_img, cam, colormap=cm.jet, alpha=0.5):
        """
        Overlays the Grad-CAM heatmap on the original PIL image.
        """
        # Resize CAM to image size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(original_img.size, Image.BILINEAR)
        cam_array = np.array(cam_img) / 255.0

        # Apply colormap
        heatmap = colormap(cam_array)[:, :, :3]  # RGB heatmap [0, 1]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap)

        # Blend
        blended = Image.blend(original_img.convert("RGB"), heatmap_img, alpha=alpha)
        return blended, heatmap_img
