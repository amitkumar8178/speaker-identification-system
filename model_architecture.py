import torch

class StudentCNN(torch.nn.Module):  
    def __init__(self, num_speakers, input_shape):
        super(StudentCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()

        self.flatten_size = self._get_flatten_size(input_shape)

        self.fc1 = torch.nn.Linear(self.flatten_size, 64)
        self.fc2 = torch.nn.Linear(64, num_speakers)

    def _get_flatten_size(self, input_shape):
        x = torch.randn(1, 1, *input_shape)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.flatten_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
