import os
import torch
import torch.nn as nn
import torch.optim as optim
from model_cnn import SimpleCNN
from tqdm import tqdm
def train_eval_cnn(train_loader, test_loader, device, epochs=128,checkpoint=None,eval_only=False):


    print(f"device: {device}")
    

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从第 {start_epoch} epoch 继续训练...")
    else:
        start_epoch = 0
    
    # --- 训练过程 ---
    if not eval_only:
        loop = tqdm(range(start_epoch, epochs), total=epochs)
        for epoch in loop:
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
            
            # 2. 计算当前 Epoch 的平均 Loss
            avg_loss = running_loss / len(train_loader)
            # 3. 更新进度条后缀，显示 Loss
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss=avg_loss)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }
            ckpt_save_path = f'checkpoints/last_checkpoint.pth'
            if not os.path.exists(os.path.dirname(ckpt_save_path)):
                os.makedirs(os.path.dirname(ckpt_save_path), exist_ok=True)
            torch.save(checkpoint, ckpt_save_path)
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