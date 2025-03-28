from transformers import AutoModel, AutoImageProcessor
from torch.nn import Module, Linear, BatchNorm1d
import torch
import torchvision.transforms as transforms

class DinoNet(Module):
    def __init__(self, model_path=None):
        super(DinoNet, self).__init__()
        # self.model = AutoModel.from_pretrained("facebook/dino-vits8")
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.fc = Linear(2048, 512, bias=False)
        self.bn = BatchNorm1d(512)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.model(self.transform(x))  # Get feature representation from DINO backbone
        x = self.fc(x)
        embedding = self.bn(x)  # Project to 512-dimensional embedding
        return embedding
    
