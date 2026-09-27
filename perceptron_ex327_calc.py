import numpy as np

# Đề bài: Cho w = [1, 2, -10]^T, x = [3, 4, 1]^T
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_true = -1  # Nhãn thực tế
# 1. Tính w^T * x
wTx = np.dot(w, x)
print(f"1. Giá trị w^T * x = {wTx}")

# 2. Xác định nhãn dự đoán của điểm dữ liệu
# Hàm kích hoạt Perceptron: y_hat = 1 nếu w^T * x >= 0, ngược lại -1
y_pred = 1 if wTx >= 0 else -1
print(f"2. Nhãn dự đoán (y_hat) = {y_pred}")

# 3. Kiểm tra xem điểm dữ liệu có bị phân lớp sai không với y = -1
is_misclassified = y_pred != y_true
print(f"3. Nhãn thực tế y = {y_true}")
if is_misclassified:
    print(
        "   => Kết luận: Điểm dữ liệu BỊ PHÂN LỚP SAI (do nhãn dự đoán khác nhãn thực tế)."
    )
else:
    print("   => Kết luận: Điểm dữ liệu ĐƯỢC PHÂN LỚP ĐÚNG.")