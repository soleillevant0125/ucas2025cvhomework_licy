import torch
import torch.nn as nn
import torch.optim as optim
from model_cnn import SimpleCNN

def train_eval_cnn(train_loader, test_loader, device, epochs=5):


    print(f"device: {device}")
    

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 训练过程 ---
    # print("\n开始训练 CNN...")
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            
            # 200 个 batch 打印一次
            if i % 200 == 199:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.4f}')
                running_loss = 0.0

    # print("CNN训练完成")

    # print("\n评估 CNN")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'CNN 在测试集上的最终准确率: {acc:.2f}%')