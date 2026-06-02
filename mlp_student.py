"""
=============================================================================
 BÀI TẬP: XÂY DỰNG MẠNG NƠ-RON ĐA LỚP (MLP) TỪ ĐẦU (FROM SCRATCH)
 Tập dữ liệu: Student Performance
 Thư viện được phép: numpy, pandas, time
=============================================================================
"""

import numpy as np
import pandas as pd
import time

# =============================================================================
# PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================

def load_and_preprocess(filepath, scale_method='minmax'):
    """
    Đọc và tiền xử lý dữ liệu Student Performance.

    Tham số:
      filepath     : đường dẫn đến file CSV
      scale_method : 'minmax' (Min-Max Scaling) hoặc 'zscore' (Z-score)

    Trả về:
      X      : ma trận features đã chuẩn hóa  (numpy array, shape [N, 31])
      Y_onehot: nhãn dạng One-Hot Encoding     (numpy array, shape [N, 8])
      classes : danh sách các lớp (nhãn gốc)
    """
    # --- Đọc CSV ---
    df = pd.read_csv(filepath)

    # --- Bỏ cột STUDENT ID (cột đầu tiên) ---
    # Cột 1..30 là Features (index 1..30 trong DataFrame sau khi bỏ cột ID)
    # Cột 31 là COURSE ID (vẫn là feature)
    # Cột 32 là GRADE (nhãn)
    feature_cols = df.columns[1:32]   # 31 features: cột index 1..31
    label_col    = df.columns[32]     # GRADE

    X_raw = df[feature_cols].values.astype(float)   # shape: [N, 31]
    y_raw = df[label_col].values                    # shape: [N]

    # --- Xử lý Categorical (nếu có) ---
    # Kiểm tra từng cột xem có dữ liệu kiểu chuỗi không
    for i, col in enumerate(feature_cols):
        unique_vals = df[col].unique()
        if df[col].dtype == object:
            # Mã hóa Label Encoding bằng pandas/numpy
            categories = sorted(df[col].astype(str).unique())
            mapping = {v: k for k, v in enumerate(categories)}
            X_raw[:, i] = df[col].astype(str).map(mapping).values.astype(float)

    # --- Chuẩn hóa dữ liệu (chỉ dùng numpy) ---
    if scale_method == 'minmax':
        X_min = X_raw.min(axis=0)
        X_max = X_raw.max(axis=0)
        denom = X_max - X_min
        # Tránh chia cho 0 nếu một cột hằng số
        denom[denom == 0] = 1.0
        X_scaled = (X_raw - X_min) / denom
    elif scale_method == 'zscore':
        X_mean = X_raw.mean(axis=0)
        X_std  = X_raw.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_raw - X_mean) / X_std
    else:
        raise ValueError("scale_method phải là 'minmax' hoặc 'zscore'")

    # --- One-Hot Encoding cho nhãn ---
    classes = sorted(np.unique(y_raw))          # [0,1,2,3,4,5,6,7]
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    N = len(y_raw)
    Y_onehot = np.zeros((N, n_classes), dtype=float)
    for row, label in enumerate(y_raw):
        Y_onehot[row, class_to_idx[label]] = 1.0

    return X_scaled, Y_onehot, classes


# =============================================================================
# PHẦN 2: CÁC HÀM KÍCH HOẠT VÀ HÀM MẤT MÁT
# =============================================================================

def sigmoid(z):
    """
    Hàm kích hoạt Sigmoid được ổn định số học:
      - Với z >= 0: sigmoid = 1 / (1 + exp(-z))
      - Với z <  0: sigmoid = exp(z) / (1 + exp(z))
    Tránh overflow khi z rất âm hoặc rất dương.
    """
    pos_mask = (z >= 0)
    neg_mask = ~pos_mask
    result = np.empty_like(z)

    # z >= 0
    exp_neg = np.exp(-z[pos_mask])
    result[pos_mask] = 1.0 / (1.0 + exp_neg)

    # z < 0
    exp_pos = np.exp(z[neg_mask])
    result[neg_mask] = exp_pos / (1.0 + exp_pos)

    return result

def sigmoid_derivative(a):
    """
    Đạo hàm của Sigmoid tính từ đầu ra a = sigmoid(z):
      sigmoid'(z) = a * (1 - a)
    """
    return a * (1.0 - a)

def softmax(z):
    """
    Hàm Softmax ổn định số học (trừ max để tránh overflow):
      softmax(z_i) = exp(z_i - max) / sum(exp(z_j - max))
    z có shape [N, K] với N là số mẫu, K là số lớp.
    """
    z_shifted = z - z.max(axis=1, keepdims=True)   # ổn định số học
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def cross_entropy_loss(Y_pred, Y_true):
    """
    Hàm mất mát Cross-Entropy:
      L = -1/N * sum_i sum_k [ y_ik * log(y_pred_ik) ]
    Thêm epsilon nhỏ để tránh log(0).
    """
    epsilon = 1e-12
    N = Y_true.shape[0]
    loss = -np.sum(Y_true * np.log(Y_pred + epsilon)) / N
    return loss


# =============================================================================
# PHẦN 3: KHỞI TẠO THAM SỐ MẠNG
# =============================================================================

def initialize_weights(n_input, n_hidden, n_output, seed=42):
    """
    Khởi tạo trọng số bằng phương pháp Xavier/Glorot:
      W ~ Uniform(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))
    Thiên lệch (bias) khởi tạo bằng 0.

    Tham số:
      n_input  : số nơ-ron lớp đầu vào (= số features)
      n_hidden : số nơ-ron lớp ẩn
      n_output : số nơ-ron lớp đầu ra (= số lớp)

    Trả về dictionary chứa:
      W1 : [n_input,  n_hidden]
      b1 : [1, n_hidden]
      W2 : [n_hidden, n_output]
      b2 : [1, n_output]
    """
    rng = np.random.default_rng(seed)

    # Lớp ẩn (Input -> Hidden)
    limit1 = np.sqrt(6.0 / (n_input + n_hidden))
    W1 = rng.uniform(-limit1, limit1, size=(n_input, n_hidden))
    b1 = np.zeros((1, n_hidden))

    # Lớp đầu ra (Hidden -> Output)
    limit2 = np.sqrt(6.0 / (n_hidden + n_output))
    W2 = rng.uniform(-limit2, limit2, size=(n_hidden, n_output))
    b2 = np.zeros((1, n_output))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# =============================================================================
# PHẦN 4: LAN TRUYỀN THUẬN (FORWARD PROPAGATION)
# =============================================================================

def forward(X, params):
    """
    Lan truyền thuận qua MLP 1 lớp ẩn.

    Kiến trúc:
      Input (n_input) -> [W1,b1] -> Z1 -> Sigmoid -> A1
      A1 (n_hidden)  -> [W2,b2] -> Z2 -> Softmax  -> A2 (xác suất)

    Trả về cache chứa các giá trị trung gian cần cho Backprop.
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    # Lớp ẩn
    Z1 = X @ W1 + b1          # [N, n_hidden]
    A1 = sigmoid(Z1)           # [N, n_hidden]

    # Lớp đầu ra
    Z2 = A1 @ W2 + b2          # [N, n_output]
    A2 = softmax(Z2)           # [N, n_output]

    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


# =============================================================================
# PHẦN 5: LAN TRUYỀN NGƯỢC (BACKPROPAGATION)
# =============================================================================

def backward(cache, params, Y_true):
    """
    Lan truyền ngược để tính gradient theo Cross-Entropy + Softmax.

    Đạo hàm Cross-Entropy + Softmax tại lớp ra:
      dZ2 = A2 - Y_true   (công thức đơn giản hóa)

    Đạo hàm tại lớp ẩn (Sigmoid):
      dA1 = dZ2 @ W2^T
      dZ1 = dA1 * sigmoid'(A1) = dA1 * A1 * (1 - A1)

    Trả về gradient cho W1, b1, W2, b2.
    """
    N  = Y_true.shape[0]
    X  = cache['X']
    A1 = cache['A1']
    A2 = cache['A2']

    W2 = params['W2']

    # Gradient tại lớp ra
    dZ2 = (A2 - Y_true) / N     # [N, n_output]
    dW2 = A1.T @ dZ2             # [n_hidden, n_output]
    db2 = dZ2.sum(axis=0, keepdims=True)   # [1, n_output]

    # Gradient tại lớp ẩn
    dA1 = dZ2 @ W2.T             # [N, n_hidden]
    dZ1 = dA1 * sigmoid_derivative(A1)     # [N, n_hidden]
    dW1 = X.T @ dZ1              # [n_input, n_hidden]
    db1 = dZ1.sum(axis=0, keepdims=True)   # [1, n_hidden]

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


# =============================================================================
# PHẦN 6: CẬP NHẬT THAM SỐ (GRADIENT DESCENT)
# =============================================================================

def update_params(params, grads, learning_rate):
    """
    Cập nhật tham số theo Gradient Descent tiêu chuẩn:
      W = W - lr * dW
      b = b - lr * db
    """
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    return params


# =============================================================================
# PHẦN 7: HUẤN LUYỆN MLP (TRAINING)
# =============================================================================

def train(X, Y, n_hidden, learning_rate, max_epochs, stop_loss, seed=42, verbose=True):
    """
    Huấn luyện MLP với các tham số tùy chỉnh.

    Tham số:
      X            : ma trận features [N, n_input]
      Y            : nhãn One-Hot [N, n_output]
      n_hidden     : số nơ-ron lớp ẩn
      learning_rate: tốc độ học
      max_epochs   : số vòng lặp tối đa
      stop_loss    : ngưỡng dừng sớm (dừng khi loss <= stop_loss)
      seed         : hạt giống ngẫu nhiên (để tái lập kết quả)
      verbose      : in thông tin trong quá trình huấn luyện

    Trả về:
      params      : tham số mạng đã huấn luyện
      history     : danh sách loss theo từng epoch
      actual_epochs: số epoch thực tế đã chạy
      elapsed_time : thời gian huấn luyện (giây)
    """
    n_input  = X.shape[1]
    n_output = Y.shape[1]

    # Khởi tạo tham số
    params = initialize_weights(n_input, n_hidden, n_output, seed=seed)

    history      = []
    actual_epochs = 0
    start_time   = time.time()

    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch

        # Forward
        A2, cache = forward(X, params)

        # Tính loss
        loss = cross_entropy_loss(A2, Y)
        history.append(loss)

        # In tiến trình (mỗi 100 epoch)
        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"  Epoch {epoch:5d}/{max_epochs} | Loss: {loss:.6f}")

        # Kiểm tra điều kiện dừng sớm
        if loss <= stop_loss:
            if verbose:
                print(f"  [Dừng sớm] Loss {loss:.6f} <= ngưỡng {stop_loss} tại epoch {epoch}")
            break

        # Backward
        grads = backward(cache, params, Y)

        # Cập nhật tham số
        params = update_params(params, grads, learning_rate)

    elapsed_time = time.time() - start_time
    return params, history, actual_epochs, elapsed_time


# =============================================================================
# PHẦN 8: DỰ ĐOÁN VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC
# =============================================================================

def predict(X, params):
    """
    Dự đoán nhãn cho tập dữ liệu X.
    Trả về chỉ số lớp có xác suất cao nhất.
    """
    A2, _ = forward(X, params)
    return np.argmax(A2, axis=1)

def accuracy(X, Y_onehot, params):
    """
    Tính độ chính xác (Accuracy) trên tập dữ liệu.
    """
    y_pred = predict(X, params)
    y_true = np.argmax(Y_onehot, axis=1)
    return np.mean(y_pred == y_true)


# =============================================================================
# PHẦN 9: CÁC KỊCH BẢN THỬ NGHIỆM
# =============================================================================

if __name__ == '__main__':

    FILEPATH = r'Student performance.csv'   # Đặt cùng thư mục với file .py

    print("=" * 65)
    print("  MẠNG NƠ-RON ĐA LỚP (MLP) FROM SCRATCH")
    print("  Tập dữ liệu: Student Performance")
    print("=" * 65)

    # --- Nạp và tiền xử lý dữ liệu ---
    print("\n[1] Đang tải và tiền xử lý dữ liệu...")
    X, Y, classes = load_and_preprocess(FILEPATH, scale_method='minmax')
    print(f"    Số mẫu      : {X.shape[0]}")
    print(f"    Số features : {X.shape[1]}")
    print(f"    Số lớp      : {len(classes)}  → {classes}")
    print(f"    Phân bố nhãn: {dict(zip(classes, Y.sum(axis=0).astype(int)))}")

    # -----------------------------------------------------------------
    # ĐỊNH NGHĨA 3 KỊCH BẢN (SCENARIOS)
    # -----------------------------------------------------------------
    scenarios = [
        {
            'name'         : 'Kịch bản 1',
            'description'  : 'Nhỏ, LR thấp',
            'n_hidden'     : 16,
            'learning_rate': 0.05,
            'max_epochs'   : 1000,
            'stop_loss'    : 1e-4,
        },
        {
            'name'         : 'Kịch bản 2',
            'description'  : 'Vừa, LR trung bình',
            'n_hidden'     : 32,
            'learning_rate': 0.1,
            'max_epochs'   : 2000,
            'stop_loss'    : 1e-4,
        },
        {
            'name'         : 'Kịch bản 3',
            'description'  : 'Lớn, LR cao, dừng sớm ở loss=1.0',
            'n_hidden'     : 64,
            'learning_rate': 0.2,
            'max_epochs'   : 3000,
            'stop_loss'    : 1.0,
        },
    ]

    # -----------------------------------------------------------------
    # CHẠY THỬ NGHIỆM
    # -----------------------------------------------------------------
    results = []

    for sc in scenarios:
        print(f"\n{'=' * 65}")
        print(f"  {sc['name']} — {sc['description']}")
        print(f"  Cấu hình: n_hidden={sc['n_hidden']}, "
              f"lr={sc['learning_rate']}, "
              f"max_epochs={sc['max_epochs']}, "
              f"stop_loss={sc['stop_loss']}")
        print(f"{'=' * 65}")

        params, history, actual_epochs, elapsed = train(
            X=X,
            Y=Y,
            n_hidden      = sc['n_hidden'],
            learning_rate = sc['learning_rate'],
            max_epochs    = sc['max_epochs'],
            stop_loss     = sc['stop_loss'],
            seed          = 42,
            verbose       = True,
        )

        final_loss = history[-1]
        acc        = accuracy(X, Y, params)

        results.append({
            'Kịch bản'          : sc['name'],
            'n_hidden'           : sc['n_hidden'],
            'Learning Rate'      : sc['learning_rate'],
            'Max Epochs'         : sc['max_epochs'],
            'Stop Loss'          : sc['stop_loss'],
            'Epochs thực tế'     : actual_epochs,
            'Thời gian (s)'      : round(elapsed, 4),
            'Loss cuối'          : round(final_loss, 6),
            'Accuracy (%)'       : round(acc * 100, 2),
        })

        print(f"\n  → Kết quả: Epochs={actual_epochs} | "
              f"Loss={final_loss:.6f} | "
              f"Acc={acc*100:.2f}% | "
              f"Thời gian={elapsed:.4f}s")

    # -----------------------------------------------------------------
    # IN BẢNG SO SÁNH DẠNG MARKDOWN
    # -----------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("  BẢNG SO SÁNH KẾT QUẢ CÁC KỊCH BẢN")
    print("=" * 65)

    # Header
    header = (
        "| Kịch bản | Kiến trúc | LR | Max Epochs | Stop Loss"
        " | Epochs TT | Thời gian (s) | Loss cuối | Accuracy (%) |"
    )
    sep = (
        "|---|---|---|---|---|---|---|---|---|"
    )
    print(header)
    print(sep)

    for r in results:
        arch = f"31-{r['n_hidden']}-8"
        row = (
            f"| {r['Kịch bản']} | {arch}"
            f" | {r['Learning Rate']}"
            f" | {r['Max Epochs']}"
            f" | {r['Stop Loss']}"
            f" | {r['Epochs thực tế']}"
            f" | {r['Thời gian (s)']}"
            f" | {r['Loss cuối']}"
            f" | {r['Accuracy (%)']}"
            f" |"
        )
        print(row)

    print("\n[Hoàn thành] Tất cả kịch bản đã chạy xong.")
