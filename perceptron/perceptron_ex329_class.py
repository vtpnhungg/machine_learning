import numpy as np


class Perceptron:
    """Lớp cài đặt thuật toán Perceptron cho bài toán phân loại nhị phân.

    Parameters:
    -----------
    eta : float
        Tốc độ học (Learning rate, nằm trong khoảng 0.0 đến 1.0).
    n_iter : int
        Số lần lặp (epochs) đi qua tập dữ liệu huấn luyện.
    random_state : int
        Hạt giống ngẫu nhiên để khởi tạo trọng số w.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Huấn luyện mô hình Perceptron trên tập dữ liệu.

        Parameters:
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Tập dữ liệu đầu vào.
        y : array-like, shape = [n_samples]
            Nhãn thực tế (dạng -1 hoặc 1).

        Returns:
        --------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        # Khởi tạo trọng số w ngẫu nhiên theo phân phối chuẩn với độ lệch chuẩn nhỏ
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0  # Khởi tạo bias b bằng 0
        self.errors_ = []  # Lưu số lượng mẫu bị phân loại sai ở mỗi epoch

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Công thức cập nhật: update = eta * (y - y_hat)
                # - Nếu dự đoán đúng: update = 0 (không đổi trọng số)
                # - Nếu dự đoán sai: update khác 0 (tiến hành cập nhật w và b)
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Tính tổ hợp tuyến tính z = w^T * x + b"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Dự báo nhãn của dữ liệu mới.

        Trả về 1 nếu net_input >= 0, ngược lại trả về -1.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ================= TEST THỬ LỚP PERCEPTRON =================
if __name__ == "__main__":
    print("=" * 15, "BÀI 3.29: THỬ NGHIỆM LỚP PERCEPTRON", "=" * 15)

    # Tạo dữ liệu kiểm thử nhỏ (Dạng AND gate)
    # X gồm 4 mẫu, 2 đặc trưng
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Nhãn target dạng -1 và 1
    y_test = np.array([-1, -1, -1, 1])

    # Khởi tạo và huấn luyện mô hình
    ppn = Perceptron(eta=0.1, n_iter=10, random_state=42)
    ppn.fit(X_test, y_test)

    # Dự báo nhãn mới
    y_pred = ppn.predict(X_test)

    print("Trọng số w học được :", ppn.w_)
    print("Bias b học được     :", ppn.b_)
    print("Nhãn thực tế        :", y_test)
    print("Nhãn dự báo         :", y_pred)
    print(
        "Kết quả kiểm thử    :",
        "THÀNH CÔNG" if np.array_equal(y_test, y_pred) else "THẤT BẠI",
    )