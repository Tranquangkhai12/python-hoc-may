import numpy as np
import matplotlib.pyplot as plt
from mlp_student import load_and_preprocess, initialize_weights, forward, backward, update_params, cross_entropy_loss

def train_with_history(X, Y, n_hidden, learning_rate, max_epochs, stop_loss, seed=42):
    n_input  = X.shape[1]
    n_output = Y.shape[1]

    params = initialize_weights(n_input, n_hidden, n_output, seed=seed)

    loss_history = []
    acc_history = []

    for epoch in range(1, max_epochs + 1):
        # Forward
        A2, cache = forward(X, params)

        # Calculate loss
        loss = cross_entropy_loss(A2, Y)
        loss_history.append(loss)
        
        # Calculate accuracy on the fly
        y_pred = np.argmax(A2, axis=1)
        y_true = np.argmax(Y, axis=1)
        acc = np.mean(y_pred == y_true) * 100
        acc_history.append(acc)

        # Early stopping condition
        if loss <= stop_loss:
            break

        # Backward
        grads = backward(cache, params, Y)

        # Update params
        params = update_params(params, grads, learning_rate)

    return loss_history, acc_history

def main():
    FILEPATH = r'Student performance.csv'
    X, Y, classes = load_and_preprocess(FILEPATH, scale_method='minmax')

    scenarios = [
        {'name': 'Kịch bản 1 (16 nơ-ron, LR=0.05)', 'n_hidden': 16, 'learning_rate': 0.05, 'max_epochs': 1000, 'stop_loss': 1e-4},
        {'name': 'Kịch bản 2 (32 nơ-ron, LR=0.10)', 'n_hidden': 32, 'learning_rate': 0.1, 'max_epochs': 2000, 'stop_loss': 1e-4},
        {'name': 'Kịch bản 3 (64 nơ-ron, LR=0.20)', 'n_hidden': 64, 'learning_rate': 0.2, 'max_epochs': 3000, 'stop_loss': 1.0},
    ]

    results = {}
    for sc in scenarios:
        print(f"Đang huấn luyện {sc['name']}...")
        loss_hist, acc_hist = train_with_history(
            X, Y, sc['n_hidden'], sc['learning_rate'], sc['max_epochs'], sc['stop_loss']
        )
        results[sc['name']] = {'loss': loss_hist, 'acc': acc_hist}

    # 1. Biểu đồ Đường cong Mất mát (Loss Curve)
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['loss'], label=name, linewidth=2)
    plt.title('Biểu đồ Đường cong Mất mát (Loss Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Số vòng lặp (Epochs)', fontsize=12)
    plt.ylabel('Sai số (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    print("Đã lưu biểu đồ Loss Curve thành 'loss_curve.png'.")
    plt.close()

    # 2. Biểu đồ Độ chính xác (Accuracy Curve)
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['acc'], label=name, linewidth=2)
    plt.title('Biểu đồ Độ chính xác (Accuracy Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Số vòng lặp (Epochs)', fontsize=12)
    plt.ylabel('Độ chính xác (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png', dpi=300)
    print("Đã lưu biểu đồ Accuracy Curve thành 'accuracy_curve.png'.")
    plt.close()

if __name__ == '__main__':
    main()
