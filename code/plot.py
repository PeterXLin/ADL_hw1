import matplotlib.pyplot as plt

# 假設有一些數據點
x = [1, 2, 3, 4, 5]  # X 軸數據
y = [83.117, 83.151, 83.749, 83.716, 84.015]  # 對應的 Y 軸數據

# 使用 Matplotlib 畫線性圖
plt.figure(figsize=(8, 6))  # 設置圖表大小（可選）
plt.plot(x, y, marker='o', color='b', label='EM')  # 畫線，帶有標記點和藍色線條，並標上標籤
plt.xlabel('epoch')  # X 軸標籤
plt.ylabel('EM')  # Y 軸標籤
plt.title('exact match')  # 圖表標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格（可選）

# 顯示圖表
plt.show()


x = [1, 2, 3, 4, 5]  # X 軸數據
y = [0.8502, 0.4354, 0.2964, 0.247, 0.2343]  # 對應的 Y 軸數據

# 使用 Matplotlib 畫線性圖
plt.figure(figsize=(8, 6))  # 設置圖表大小（可選）
plt.plot(x, y, marker='o', color='b',
         label='training loss')  # 畫線，帶有標記點和藍色線條，並標上標籤
plt.xlabel('epoch')  # X 軸標籤
plt.ylabel('training loss')  # Y 軸標籤
plt.title('training loss')  # 圖表標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格（可選）

# 顯示圖表
plt.show()
