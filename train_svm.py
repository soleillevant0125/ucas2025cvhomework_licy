from sklearn import svm
from sklearn.metrics import accuracy_score
import time

def train_eval_svm(X_train, y_train, X_test, y_test):

    results = {}

    for kernel_type in ['linear', 'rbf']:
        print(f"SVM (Kernel: {kernel_type})...")
        start_time = time.time()
        
        # 定义模型
        clf = svm.SVC(kernel=kernel_type, C=1.0, random_state=42, verbose=True)
        
        # 训练
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 评估
        acc = accuracy_score(y_test, y_pred)
        end_time = time.time()
        
        print(f"Kernel: {kernel_type} | Accuracy: {acc:.4f} | Time: {end_time - start_time:.2f}s")
        results[kernel_type] = acc

    # 简要对比
    print("\n>>> SVM 对比分析:")
    if results['rbf'] > results['linear']:
        print("结论: RBF 核表现更好，适合处理非线性图像数据。")
    else:
        print("结论: 两者表现接近。")