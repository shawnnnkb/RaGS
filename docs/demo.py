from PIL import Image
import imageio
import os
import numpy as np
import cv2

def get_valid_prefixes(text_file):
    # 从文本文件中读取有效的前缀
    with open(text_file, 'r') as file:
        valid_prefixes = [line.strip() for line in file.readlines() if line.strip()]
    return valid_prefixes

def create_gif(image_folder, output_gif, valid_prefixes, duration=0.1, resize_dim=(600, 600)): # 242*5, 162*5
    images = []
    
    # 遍历有效的前缀
    for i, prefix in enumerate(valid_prefixes):
        img_path = os.path.join(image_folder, prefix + '.jpg')
        
        if os.path.exists(img_path):  # 如果文件存在
            img = Image.open(img_path)  # 使用Pillow打开图像
            img = img.resize(resize_dim)  # 调整图像大小为 (162, 242)
            images.append(img)  # 将图像添加到列表
            
            print(f"Processing {i+1}/{len(valid_prefixes)}: {prefix}")
    
    # 使用Pillow保存为GIF
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF saved as {output_gif}")

def create_gif_on_file(image_folder, output_gif, prefix, duration=0.005, resize_dim=(600, 600)):
    images = []
    files = sorted(os.listdir(image_folder))
    
    for i, filename in enumerate(files):
        if filename.endswith("png") and prefix in filename:
            index = int(filename.split('_')[1])  # 提取并转换为整数
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path)  # 使用Pillow打开图像
            image = image.resize(resize_dim)  # 调整图像大小为 (162, 242)
            image = np.array(image)
            images.append(dict(index=index, image=image))
            print(f"Processing {i+1}/{len(files)}: {filename}")
    
    images_sorted = sorted(images, key=lambda x: x['index'])
    sorted_images = [item['image'] for item in images_sorted]
    imageio.mimsave(output_gif, sorted_images, duration=duration)
    print(f"GIF saved as {output_gif}")

def create_mp4_on_file(image_folder, output_mp4, prefix, duration=0.005, resize_dim=(600, 600)):
    images = []
    files = sorted(os.listdir(image_folder))
    
    for i, filename in enumerate(files):
        if filename.endswith("png") and prefix in filename:
            index = int(filename.split('_')[1])  # 提取并转换为整数
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path)  # 使用Pillow打开图像
            image = image.convert("RGB")
            image = image.resize(resize_dim)  # 调整图像大小为 (600, 600)
            image = np.array(image, dtype=np.uint8)[:,:,:3][:,:,(2,1,0)]
            images.append(dict(index=index, image=image))
            print(f"Processing {i+1}/{len(files)}: {filename}")
    
    images_sorted = sorted(images, key=lambda x: x['index'])
    sorted_images = [item['image'] for item in images_sorted]
    
    # 获取图像尺寸
    height, width, _ = sorted_images[0].shape
    
    # 使用cv2.VideoWriter创建一个MP4视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用MP4编码 mp4v *'MJPG'
    video_writer = cv2.VideoWriter(output_mp4, fourcc, 1 / duration, (width, height))
    
    for image in sorted_images:
        video_writer.write(image)  # 写入每一帧
    
    video_writer.release()  # 释放视频写入对象
    print(f"MP4 video saved as {output_mp4}")

if __name__ == "__main__":

    ##################################################################################
    # text_file = './data/VoD/radar_5frames/ImageSets/val.txt'  # 你上传的文件路径
    # image_folder = "./data/VoD/radar_5frames/training/image_2"  # 图像文件夹路径
    # output_gif = "./output.gif"  # 输出的 GIF 文件
    # valid_prefixes = get_valid_prefixes(text_file)
    # create_gif(image_folder, output_gif, valid_prefixes, duration=0.1)
    
    ##################################################################################
    image_folder = 'work_dirs/vod-RadarGS_4x4_24e_detect3d/figures_path/test/det3d'
    create_mp4_on_file(image_folder, "./BEV_Det3D_VoD.avi", prefix='paper_radar', duration=0.1, resize_dim=(640, 640))
    create_mp4_on_file(image_folder, "./Image_Det3D_VoD.avi", prefix='split_pred', duration=0.1, resize_dim=(1024, 640))

    image_folder = 'work_dirs/TJ4D-RadarGS_4x4_24e_detect3d/figures_path/test/det3d'
    create_mp4_on_file(image_folder, "./BEV_Det3D_TJ4D.avi", prefix='paper_radar', duration=0.1, resize_dim=(640, 640))
    create_mp4_on_file(image_folder, "./Image_Det3D_TJ4D.avi", prefix='split_pred', duration=0.1, resize_dim=(1024, 640))
