import torch
import data_loader
import train_svm
import train_cnn
import os

def main(svm=True, cnn=False):
    print("开始执行图像分类任务...\n")

    # TODO: 请修改这里的路径为你存放 .gz 文件的实际路径
    # 例如: './fashion-mnist/data/fashion'
    LOCAL_DATA_PATH = 'data/fashion' 
     
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"错误: 数据路径 {LOCAL_DATA_PATH} 不存在。")
        return
    if svm:
    # svm
        X_train, y_train, X_test, y_test = data_loader.get_svm_data(
            data_root=LOCAL_DATA_PATH, 
            limit_train=None, 
            limit_test=None
        )
        train_svm.train_eval_svm(X_train, y_train, X_test, y_test)
    if cnn:  
    # CNN
        train_loader, test_loader = data_loader.get_cnn_loaders(
            batch_size=64, 
            data_root=LOCAL_DATA_PATH
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_cnn.train_eval_cnn(train_loader, test_loader, device, epochs=128,checkpoint=None,eval_only=False)
    print('Finished!')
if __name__ == "__main__":
    main(svm=False, cnn=True)