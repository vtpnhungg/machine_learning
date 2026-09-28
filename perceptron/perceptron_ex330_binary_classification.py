import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Re-use lớp Perceptron chuẩn từ Bài 3.29
class Perceptron:
    """Lớp Perceptron tự cài đặt cho bài toán phân loại nhị phân."""

    def __init__(self, eta=0.01, n_iter=100, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ================= PHẦN THỰC THI BÀI 3.30 =================
if __name__ == "__main__":
    print("=" * 15, "BÀI 3.30: PHÂN LOẠI NHỊ PHÂN VÀ ĐÁNH GIÁ MÔ HÌNH", "=" * 15)

    # 1. Giả lập tập dữ liệu Phát hiện giao dịch gian lận (Binary Classification)
    # 1000 giao dịch, 8 đặc trưng (số tiền, thời gian, vị trí, v.v.)
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        weights=[0.85, 0.15],  # 85% hợp lệ (-1), 15% gian lận (1)
        random_state=42
    )

    # Chuyển nhãn về dạng -1 (giao dịch thường) và 1 (gian lận)
    y = np.where(y == 0, -1, 1)

    # 2. Chia tập dữ liệu thành Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Chuẩn hóa dữ liệu (Standardization) - Bước quan trọng để Perceptron hội tụ tốt nhất
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Khởi tạo và huấn luyện mô hình Perceptron
    model = Perceptron(eta=0.01, n_iter=150, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Dự báo trên tập Test
    y_pred = model.predict(X_test_scaled)

    # 6. Tính toán các độ đo đánh giá (Accuracy, Precision, Recall, F1-score)
    # Nhãn tích cực (pos_label) là 1 (Gian lận)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ ĐỘ ĐO ---")
    print(f"1. Accuracy  (Độ chính xác) : {acc * 100:.2f}%")
    print(f"2. Precision (Độ chuẩn xác) : {prec * 100:.2f}%")
    print(f"3. Recall    (Độ nhạy)      : {rec * 100:.2f}%")
    print(f"4. F1-score                 : {f1 * 100:.2f}%")

    print("\n--- BẢNG MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    print(f"True Positive (Dự báo đúng Gian lận)  : {cm[0, 0]}")
    print(f"False Negative (Bỏ sót Gian lận)       : {cm[0, 1]}")
    print(f"False Positive (Báo nhầm Gian lận)     : {cm[1, 0]}")
    print(f"True Negative (Dự báo đúng Thường)    : {cm[1, 1]}")

    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")
    print(classification_report(y_test, y_pred, target_names=["Thường (-1)", "Gian lận (1)"]))