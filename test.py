import pandas as pd

# 定义 r 和 alpha 的值
r_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
alpha_values = [0.2, 0.4, 0.6, 0.8]

# 创建一个 DataFrame
df = pd.DataFrame(index=alpha_values, columns=r_values)

# 填充 DataFrame (可以设置具体的值或者保持为空)
# 这里我们可以填充为空或者设置某些默认值
df[:] = " "  # 例如所有单元格都填充 'value'

# 将 DataFrame 导出为 Excel 文件
df.to_excel("hyperparameters.xlsx", index=True)

print("Excel 文件已创建！")
