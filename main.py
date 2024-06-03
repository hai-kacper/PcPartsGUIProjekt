import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import random
# Napisz pip install -r requirements.txt w terminalu zeby pobrac wszystko

# Function to load and preprocess the image
def load_image(image):
    image = transform(image).unsqueeze(0)
    return image

# Function to classify the image using the loaded model
def classify_image(image):
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return dataset.classes[predicted.item()]

# Function to handle random image classification from the dataset
def random_test():
    random_image_path = random.choice(dataset.imgs)[0]
    image = Image.open(random_image_path)
    result = classify_image(image)
    return image, f"Predicted Class: {result}"

# Load dataset
dataset_root = "pc_parts"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = ImageFolder(root=dataset_root, transform=transform)
print("Classes in the dataset:", dataset.classes)

# Load model
model_path = 'model.pth'
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=len(dataset.classes))
model.load_state_dict(torch.load(model_path))
model.eval()

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Classifier")

    with gr.Row():
        image_input = gr.Image(type="pil")
        image_output = gr.Label()

    browse_button = gr.Button("Classify Image")
    random_test_button = gr.Button("Random Test")

    browse_button.click(fn=classify_image, inputs=image_input, outputs=image_output)
    random_test_button.click(fn=random_test, inputs=None, outputs=[image_input, image_output])


# Run the Gradio application with external access enabled
demo.launch(share=True, server_port=7860, server_name="0.0.0.0") 