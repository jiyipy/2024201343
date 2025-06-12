import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV文件（根据实际路径修改）
df = pd.read_csv('./NEU-DET/NEU-DET_labels.csv')

# 1. 基础数据统计
print("数据概览:")
print(f"总样本数: {len(df)}")
print(f"唯一图像数量: {df['image_id'].nunique()}")
print(f"类别分布:\n{df['class_name'].value_counts()}\n")

# 计算边界框宽高和面积
df['bbox_width'] = df['width']
df['bbox_height'] = df['height']
df['bbox_area'] = df['bbox_width'] * df['bbox_height']
df['aspect_ratio'] = df['bbox_width'] / df['bbox_height']

# 2. 统计每张图像中的目标数量
objects_per_image = df.groupby('image_id').size()
print("每张图像目标数量统计:")
print(objects_per_image.describe())

# 3. 数据可视化
plt.figure(figsize=(15, 12))

# 3.1 类别分布柱状图
plt.subplot(2, 2, 1)
sns.countplot(data=df, y='class_name', order=df['class_name'].value_counts().index)
plt.title('Class Distribution')
plt.xlabel('Count')
plt.ylabel('Class Name')

# 3.2 边界框面积分布
plt.subplot(2, 2, 2)
sns.histplot(df['bbox_area'], bins=50, kde=True)
plt.title('Bounding Box Area Distribution')
plt.xlabel('Area')
plt.xscale('log')  # 使用对数坐标因为面积变化范围大

# 3.3 宽高比分布
plt.subplot(2, 2, 3)
sns.histplot(df['aspect_ratio'], bins=30, kde=True)
plt.title('Aspect Ratio Distribution')
plt.xlabel('Width/Height Ratio')
plt.axvline(1, color='r', linestyle='--', alpha=0.5)  # 标记正方形位置

# 3.4 中心点位置分布
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='center_x', y='center_y', alpha=0.4)
plt.title('Center Point Distribution')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().invert_yaxis()  # 反转Y轴以匹配图像坐标系

plt.tight_layout()
plt.savefig('overall_analysis.png', dpi=300)
plt.show()

# 4. 箱线图分析
plt.figure(figsize=(12, 6))

# 4.1 边界框尺寸分布
plt.subplot(1, 2, 1)
sns.boxplot(data=df[['bbox_width', 'bbox_height']])
plt.title('Bounding Box Size Distribution')
plt.ylabel('Normalized Units')

# 4.2 按类别的宽高比分布
plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='class_name', x='aspect_ratio', showfliers=False)
plt.title('Aspect Ratio by Class')
plt.xlabel('Width/Height Ratio')
plt.axvline(1, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('boxplots_analysis.png', dpi=300)
plt.show()

# 5. 打印关键统计量
print("\n关键统计量:")
print(f"平均目标数量/图像: {objects_per_image.mean():.2f}")
print(f"边界框平均宽度: {df['bbox_width'].mean():.4f}")
print(f"边界框平均高度: {df['bbox_height'].mean():.4f}")
print(f"边界框平均面积: {df['bbox_area'].mean():.4f}")
print(f"平均宽高比: {df['aspect_ratio'].mean():.4f}")

# 6. 异常值检测（示例）
print("\n可能的异常值检测:")
high_aspect = df[df['aspect_ratio'] > 10]
low_aspect = df[df['aspect_ratio'] < 0.1]
print(f"极端宽高比(>10): {len(high_aspect)}条")
print(f"极端宽高比(<0.1): {len(low_aspect)}条")