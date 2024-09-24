import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load the Model
model_name = 'resnet50'  # Example model
model = timm.create_model(model_name, pretrained=True)
model.eval()
features = None
gradients = None

# Function to register hooks
def get_features_hook(module, input, output):
    global features
    features = output.detach()

def get_gradients_hook(module, input_grad, output_grad):
    global gradients
    gradients = output_grad[0].detach()
# 2. Identify the Last Convolutional Layer
target_layers = [model.layer4[-1]]  # Adjust this based on the model architecture

# Function to register hooks
def get_features_hook(module, input, output):
    global features
    features = output.detach()

def get_gradients_hook(module, input_grad, output_grad):
    global gradients
    gradients = output_grad[0].detach()

# Register hooks
target_layers[0].register_forward_hook(get_features_hook)
target_layers[0].register_backward_hook(get_gradients_hook)

# 3. Preprocess Input Image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# 4. Compute Grad-CAM++
def compute_gradcam_plus_plus(model, image):
    image.requires_grad = True
    output = model(image)
    index = output.argmax(dim=1)
    
    model.zero_grad()
    class_loss = output[0, index].sum()
    class_loss.backward(retain_graph=True)
    
    # Use gradients and features captured by hooks
    global gradients, features
    alpha_numer = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + features.mul(gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_numer.div(alpha_denom+1e-7)
    
    positive_gradients = F.relu(output[0, index].exp() * gradients)  # ReLU(dY/dA)
    weights = (alphas * positive_gradients).sum(dim=(2, 3), keepdim=True)
    
    grad_cam_map = (weights * features).sum(dim=1, keepdim=True)
    grad_cam_map = F.relu(grad_cam_map)
    grad_cam_map = F.interpolate(grad_cam_map, image.shape[2:], mode='bilinear', align_corners=False)
    
    heatmap = grad_cam_map.squeeze().detach().cpu().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-7)
    return heatmap
# 5. Overlay Heatmap
def overlay_heatmap(heatmap, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  #! try uncomment this
    
    original_image = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR)
    if original_image.shape[2] == 1:  # grayscale to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    overlayed_image = heatmap * alpha + original_image * (1 - alpha)
    return cv2.cvtColor(np.uint8(overlayed_image), cv2.COLOR_BGR2RGB)

# Usage Example
img_path = 'Bacterial_Pneumonia_segmentation/person1_bacteria_1.jpg'
image = preprocess_image(img_path)
heatmap = compute_gradcam_plus_plus(model, image)
original_image = cv2.imread(img_path)
overlayed_image = overlay_heatmap(heatmap, original_image)

# Display
plt.imshow(overlayed_image)
plt.axis('off')
plt.show()
