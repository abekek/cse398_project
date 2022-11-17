import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
import torch.nn as nn
 
onnx_model_path = ""
sample_image = ""
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, num_classes = 136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

# https://pytorch.org/hub/pytorch_vision_densenet/
model = Model().to(device)
model.load_state_dict(torch.load('./saved_models/face_landmarks.pth', map_location=device))

# set the model to inference mode
model.eval()
 
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(1, 1, 224, 224).cuda()
torch.onnx.export(model, dummy_input, './saved_models/resnet18.onnx', verbose=True)