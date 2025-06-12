import hashlib
import os

def get_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def check_duplicates(directory):
    hashes = {}
    num_images = 0
    for filename in os.listdir(os.path.join(directory, 'images')):
        path = os.path.join(directory, 'images', filename)
        h = get_hash(path)
        if h in hashes:
            print(f"重复图片：{filename} 和 {hashes[h]}")
        else:
            hashes[h] = filename
        num_images += 1

    print(f"目录 {directory} 下共有 {num_images} 张图片")

if __name__ == '__main__':
    check_duplicates('./NEU-DET/train')
    check_duplicates('./NEU-DET/valid')
