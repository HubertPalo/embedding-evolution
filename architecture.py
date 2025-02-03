from torch import nn


class MyDIETBackbone(nn.Module):
    def __init__(self, encoding_size: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=8,
                stride=1, bias=False, padding=(8//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, encoding_size)
        )
        
    def forward(self, x):
        return self.model(x)
    
class MyDIETProjectionHead(nn.Module):
    def __init__(
            self,
            encoding_size: int = 128,
            output_size: int = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(encoding_size, output_size, bias=False)
        )

    def forward(self, x):
        return self.model(x)