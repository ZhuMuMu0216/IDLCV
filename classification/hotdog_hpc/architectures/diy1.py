import torch
import torch.nn as nn
import torch.nn.functional as F

class DIY1(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # ->(64,128,128)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # ->(64,128,128)
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128, kernel_size=5, padding=2),  # ->(128,128,128)
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=5, padding=2),  # ->(128,128,128)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->(128,64,64)        
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*64*64, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x