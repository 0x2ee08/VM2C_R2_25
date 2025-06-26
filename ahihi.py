import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Tên cổ phiếu
stocks = ["VIC", "REE", "GAS", "HPG", "MWG", "VCB", "VNM", "PNJ", "FPT", "VJC"]
n = len(stocks)

# Lợi suất kỳ vọng (mu%) chuyển về dạng số thực
mu = np.array([-18.9311, 20.3524, 7.9976, 14.5745, 10.2352,
               10.5293, -13.7006, 17.0632, 36.0089, -2.5574]) / 100

# Ma trận hiệp phương sai (covariance matrix)
Sigma = np.array([
    [0.1213, 0.0154, 0.0187, 0.0415, 0.0084, 0.0088, 0.0055, 0.0106, 0.0125, 0.0080],
    [0.0154, 0.1082, 0.0484, 0.0394, 0.0558, 0.0164, 0.0156, 0.0384, 0.0466, 0.0057],
    [0.0187, 0.0484, 0.1111, 0.0387, 0.0434, 0.0308, 0.0089, 0.0300, 0.0370, 0.0067],
    [0.0415, 0.0394, 0.0387, 0.1697, 0.0675, 0.0467, 0.0217, 0.0343, 0.0403, 0.0156],
    [0.0084, 0.0558, 0.0434, 0.0675, 0.1523, 0.0398, 0.0226, 0.0531, 0.0510, 0.0135],
    [0.0088, 0.0164, 0.0308, 0.0467, 0.0398, 0.0680, 0.0194, 0.0201, 0.0254, 0.0095],
    [0.0055, 0.0156, 0.0089, 0.0217, 0.0226, 0.0194, 0.0441, 0.0222, 0.0281, 0.0077],
    [0.0106, 0.0384, 0.0300, 0.0343, 0.0531, 0.0201, 0.0222, 0.0821, 0.0392, 0.0158],
    [0.0125, 0.0466, 0.0370, 0.0403, 0.0510, 0.0254, 0.0281, 0.0392, 0.0713, 0.0110],
    [0.0080, 0.0057, 0.0067, 0.0156, 0.0135, 0.0095, 0.0077, 0.0158, 0.0110, 0.0516]
])
print(Sigma)
# Tạo model
model = gp.Model("Portfolio Optimization")

# Biến quyết định: trọng số đầu tư w_i vào từng cổ phiếu
w = model.addVars(n, lb=0.0, ub=1.0, name="w")

# Mục tiêu: tối đa hóa lợi suất kỳ vọng danh mục
model.setObjective(gp.quicksum(w[i] * mu[i] for i in range(n)), GRB.MAXIMIZE)

# Ràng buộc 1: tổng trọng số bằng 1 (toàn bộ vốn đầu tư)
model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1, name="Budget")

# Ràng buộc 2: độ lệch chuẩn danh mục không vượt quá 20%
# Tức là phương sai danh mục <= 0.2^2
portfolio_variance = gp.QuadExpr()
for i in range(n):
    for j in range(n):
        portfolio_variance += w[i] * w[j] * Sigma[i, j]
model.addQConstr(portfolio_variance <= 0.2 ** 2, name="RiskBound")

# Tối ưu hóa
model.optimize()

# In kết quả
if model.status == GRB.OPTIMAL:
    print("\nTỷ trọng tối ưu vào từng cổ phiếu:")
    for i in range(n):
        if w[i].X > 1e-4:
            print(f"{stocks[i]}: {w[i].X * 100:.2f}%")
    expected_return = sum(w[i].X * mu[i] for i in range(n))
    variance = sum(w[i].X * w[j].X * Sigma[i, j] for i in range(n) for j in range(n))
    print(f"\nLợi suất kỳ vọng danh mục: {expected_return * 100:.2f}%")
    print(f"Độ lệch chuẩn danh mục (rủi ro): {np.sqrt(variance) * 100:.2f}%")
else:
    print("Không tìm được nghiệm tối ưu.")
