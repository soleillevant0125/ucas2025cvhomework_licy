import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 卷积层  1 -> 6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层 6 -> 16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 全连接层
        # 28x28 -> conv1(24x24) -> pool(12x12) -> conv2(8x8) -> pool(4x4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 
        
        self.relu = nn.ReLU()
    #前向传播
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 16 * 4 * 4)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x