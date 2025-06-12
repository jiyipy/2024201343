import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取标签文件并添加特征
df = pd.read_csv('./NEU-DET/NEU-DET_labels.csv')
df['area'] = df['width'] * df['height']
df['aspect_ratio'] = df['width'] / df['height']

# 1. 相关性分析
pearson_corr = df[['width','height','area','aspect_ratio','center_x','center_y']].corr(method='pearson')
spearman_corr = df[['width','height','area','aspect_ratio','center_x','center_y']].corr(method='spearman')
print("Pearson相关系数：\n", pearson_corr)
print("Spearman相关系数：\n", spearman_corr)

# 绘制热力图
plt.figure(figsize=(6,5))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Pearson相关系数热力图")
plt.show()

# 2. 可视化（Pairplot）
sns.pairplot(df[['width','height','area','aspect_ratio']], corner=True)
plt.show()

# 3. 缺陷类别关联分析
group_stats = df.groupby('class_name')[['width','height','area','aspect_ratio','center_x','center_y']].mean()
print("各类别特征均值：\n", group_stats)

# 4. 中心位置散点图
plt.figure(figsize=(5,5))
colors = {'Rolled-in_Scale':'red','Crazing':'blue','Patches':'green',
          'Pitted_Surface':'orange','Inclusion':'purple','Scratches':'brown'}
for name, group in df.groupby('class_name'):
    plt.scatter(group['center_x'], group['center_y'], alpha=0.3, label=name, color=colors[name], s=5)
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('center_x'); plt.ylabel('center_y')
plt.title("不同缺陷类别的中心坐标分布")
plt.show()

# 5. 箱线图：面积和长宽比
plt.figure(figsize=(6,4))
sns.boxplot(x='class_name', y='area', data=df)
plt.xticks(rotation=45); plt.title("不同缺陷类别的面积分布")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='class_name', y='aspect_ratio', data=df)
plt.xticks(rotation=45); plt.title("不同缺陷类别的长宽比分布")
plt.show()

# 6. Anchor聚类示例（YOLO Anchor优化）
X = df[['width','height']].values
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(X)
print("聚类得到的锚框尺寸（宽度, 高度）：")
print(kmeans.cluster_centers_)
