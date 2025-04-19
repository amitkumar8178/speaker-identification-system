import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_architecture import SpeakerCNN

import torch.nn.functional as F


T = 3.0 
alpha = 0.7  
class StudentCNN(nn.Module):
    def __init__(self, num_classes):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 32 * 32, 64)  
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


train_dataset = SpeakerDataset(split='train') 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


teacher_model = SpeakerCNN()
teacher_model.load_state_dict(torch.load("speaker_recognition_model.pth"))
teacher_model.eval()

student_model = StudentCNN()  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

def distillation_loss(y_student, y_teacher, labels, T, alpha):
    soft_loss = nn.KLDivLoss()(F.log_softmax(y_student/T, dim=1), F.softmax(y_teacher/T, dim=1)) * (T*T)
    hard_loss = criterion(y_student, labels)
    return alpha * soft_loss + (1. - alpha) * hard_loss

# Training loop
for epoch in range(10):
    student_model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        student_outputs = student_model(inputs)
        loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}")

# Save distilled model
torch.save(student_model.state_dict(), "distilled_speaker_model.pth")
