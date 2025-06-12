import os

def check(directory):
    image_files = set(f[:-4] for f in os.listdir(os.path.join(directory, 'images')) if f.endswith('.jpg'))
    label_files = set(f[:-4] for f in os.listdir(os.path.join(directory, 'labels')) if f.endswith('.txt') or f.endswith('.xml'))

    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    num_images = len(image_files)
    num_labels = len(label_files)

    print(f"在 {directory} 目录下:")
    print(f"图片数量: {num_images}")
    print(f"标签数量: {num_labels}")
    print("没有标签的图片：", missing_labels)
    print("没有图片的标签：", missing_images)

if __name__ == '__main__':
    check('./NEU-DET/train')
    check('./NEU-DET/valid')
