import os
import cv2

def check_images(folder_path):
    bad_images = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            count += 1
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                bad_images.append(filename)
            elif img.shape != (200, 200):
                print(f"尺寸异常：{filename} -> {img.shape}")
    print(f"已读取{count}张图片,无法读取的图片：", bad_images)

if __name__ == '__main__':
    check_images("./NEU-DET/train/images")
    check_images("./NEU-DET/valid/images")
