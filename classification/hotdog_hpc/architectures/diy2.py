import torch
import torch.nn as nn
import torch.nn.functional as F

class DIY2(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, padding=0),  # ->(64,127,127)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, padding=0),  # ->(64,126,126)
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128, kernel_size=2, padding=0),  # ->(128,125,125)
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=2, padding=0),  # ->(128,124,124)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->(128,62,62)

            # add a new convolutional layer with 5by5 kernel size and 2 padding
            nn.Conv2d(128,128, kernel_size=5, padding=2),  # ->(128,62,62)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->(128,31,31)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*31*31, 2048),
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