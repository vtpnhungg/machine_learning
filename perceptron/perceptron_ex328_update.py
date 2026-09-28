import numpy as np

# Đề bài: Cho w = [-2, 1, 0]^T, x = [2, 3, 1]^T, y = 1 (đã thêm bias)
w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y_true = 1
eta = 1.0  # Tốc độ học (Learning rate mặc định)
# 1. Kiểm tra mẫu có bị phân lớp sai hay không
wTx = np.dot(w, x)
y_pred = 1 if wTx >= 0 else -1
is_misclassified = y_pred != y_true

print(f"1. Kiểm tra mẫu ban đầu:")
print(f"   - w^T * x = {wTx}")
print(f"   - Nhãn dự đoán (y_hat) = {y_pred}")
print(f"   - Nhãn thực tế (y)     = {y_true}")

if is_misclassified:
    print("   => Kết luận: Mẫu BỊ PHÂN LỚP SAI.\n")

    # 2. Thực hiện một bước cập nhật Perceptron: w_new = w + eta * y * x
    w_new = w + eta * y_true * x
    print(f"2. Thực hiện cập nhật trọng số:")
    print(f"   - Công thức: w_new = w + eta * y * x")
    print(f"   - Trọng số w mới sau cập nhật = {w_new}\n")

    # 3. Tính lại giá trị w^T * x sau cập nhật
    wTx_new = np.dot(w_new, x)
    y_pred_new = 1 if wTx_new >= 0 else -1
    print(f"3. Giá trị w^T * x sau cập nhật:")
    print(f"   - w_new^T * x = {wTx_new}")
    print(f"   - Nhãn dự đoán mới = {y_pred_new}")
    print(
        "   => Kết luận: Sau cập nhật, điểm dữ liệu đã được PHÂN LỚP ĐÚNG."
    )
else:
    print(
        "   => Kết luận: Mẫu ĐƯỢC PHÂN LỚP ĐÚNG (không cần cập nhật trọng số)."
    )