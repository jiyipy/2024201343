# 准备数据集结构文件

# 根目录创建1个文件夹(可自定义名称)
# 下面创建再2个文件夹(images和labels)
# images和labels 下再分别创建2个文件夹(train和val)
# images下的train和val 放入训练图片(png、jpg)
# labels下的train和val 放入图片标注(txt)


# 准备数据集配置文件

# 创建1个yaml格式的文件(可自定义名称)
# 配置数据集信息、用于训练模型
from ultralytics import YOLO

def main():
    # 训练模型
    a_1 = YOLO(".\yolo11s.pt")
    a_1.train(
        data="data.yaml",
        epochs=500,
        batch=16,
        imgsz=640,
        # 其他训练参数
    )

if __name__ == '__main__':
    main()

