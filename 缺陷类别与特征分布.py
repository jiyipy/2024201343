# 统计各类别的特征均值
group_stats = df.groupby('class_name')[['width','height','area','aspect_ratio','center_x','center_y']].mean()
print("各类别特征均值：\n", group_stats)

# 绘制不同类别的面积分布箱线图
plt.figure(figsize=(6,4))
sns.boxplot(x='class_name', y='area', data=df)
plt.xticks(rotation=45)
plt.title("不同缺陷类别的面积分布")
plt.show()

# 绘制不同类别的长宽比分布箱线图
plt.figure(figsize=(6,4))
sns.boxplot(x='class_name', y='aspect_ratio', data=df)
plt.xticks(rotation=45)
plt.title("不同缺陷类别的长宽比分布")
plt.show()
