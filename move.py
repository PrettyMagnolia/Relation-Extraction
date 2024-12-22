import os
import shutil

# 遍历runs 目录下的每一个文件夹，文件夹中 会有一个 pth 文件，将其拷贝到 BERTem文件夹中，并以文件夹的名字重名名 pth文件的名字
source_dir = '/home/yifei/code/NLP/BERT-EM/runs'
destination_dir = '/home/yifei/code/NLP/BERT-EM/BERTem'

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pth.tar'):
                source_file = os.path.join(folder_path, file_name)
                destination_file = os.path.join(destination_dir, f"{folder_name}.pth.tar")
                shutil.copy(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")