import matplotlib.pyplot as plt
import numpy as np
# 1. Định nghĩa hàm số f(x) và đạo hàm f'(x)
def cost(x):
    return x**2 - 4 * x + 5
def grad(x):
    return 2 * x - 4
# 2. Khởi tạo các tham số
x_0 = 5.0  # Điểm khởi tạo x(0)
eta = 0.2  # Learning rate
num_steps = 4  # Số bước cập nhật

# Khởi tạo danh sách lưu trữ lịch sử các giá trị
x_history = [x_0]
f_history = [cost(x_0)]

# 3. Vòng lặp Gradient Descent
x_current = x_0
print(f"{'Step (t)':<10}{'x^(t)':<12}{'f\'(x^(t))':<15}{'f(x^(t))':<12}")
print("-" * 50)
# In bước ban đầu (t = 0)
print(f"{0:<10}{x_0:<12.4f}{grad(x_0):<15.4f}{cost(x_0):<12.4f}")

for t in range(1, num_steps + 1):
    gd = grad(x_current)  # Tính đạo hàm tại vị trí x hiện tại
    x_next = x_current - eta * gd  # Cập nhật x mới
    f_next = cost(x_next)  # Tính giá trị f(x) tại x mới
    gd_next = grad(x_next)  # Tính đạo hàm tương ứng tại x mới

    # Lưu lại lịch sử
    x_history.append(x_next)
    f_history.append(f_next)

    # In ra màn hình từng bước (hiển thị đúng x_next và đạo hàm gd_next tương ứng)
    print(f"{t:<10}{x_next:<12.4f}{gd_next:<15.4f}{f_next:<12.4f}")

    # Cập nhật x cho bước tiếp theo
    x_current = x_next

# 4. Trực quan hóa quá trình hội tụ
x_vals = np.linspace(-1, 6, 400)
y_vals = cost(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, "b-", label="f(x) = x² - 4x + 5")
plt.plot(
    x_history,
    f_history,
    "ro--",
    markersize=8,
    linewidth=2,
    label="Các bước Gradient Descent",
)

# Đánh số các điểm bước nhảy trên đồ thị
for i, (x_val, y_val) in enumerate(zip(x_history, f_history)):
    plt.annotate(
        f"  t={i} (x={x_val:.2f})",
        (x_val, y_val),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
    )

plt.title("Mô phỏng Gradient Descent tiến về cực tiểu (x* = 2, f(x*) = 1)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axvline(
    x=2, color="gray", linestyle=":", label="Điểm cực tiểu lý thuyết (x=2)"
)
plt.grid(True)
plt.legend()
plt.show()

# 5. In nhận xét sự hội tụ
print("\n" + "=" * 50)
print("NHẬN XÉT SỰ HỘI TỤ CỦA THUẬT TOÁN:")
print(
    f"1. Vị trí x: Giảm dần từ {x_history[0]} -> {x_history[-1]:.4f}, tiến gần về điểm cực tiểu x* = 2."
)
print(
    f"2. Giá trị f(x): Giảm liên tục từ {f_history[0]} -> {f_history[-1]:.4f}, tiến dần về min f(x*) = 1."
)
print(
    f"3. Đạo hàm |f'(x)|: Giảm dần độ lớn từ {abs(grad(x_history[0])):.4f} xuống {abs(grad(x_history[-1])):.4f}."
)
print(
    "=> Kết luận: Thuật toán hội tụ tốt và ổn định với tốc độ học eta = 0.2."
)