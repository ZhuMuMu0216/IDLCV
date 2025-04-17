import os

# 定义文件夹路径
data_path1= ['hotdog_nothotdog/train/hotdog',
'hotdog_nothotdog/train/nothotdog',
'hotdog_nothotdog/test/hotdog',
'hotdog_nothotdog/test/nothotdog']
for data_path in data_path1:
    # 遍历文件夹中的所有文件
    for filename in os.listdir(data_path):
        # 检查是否是 .Zone.Identifier 文件
        if filename.endswith("Zone.Identifier"):
            file_path = os.path.join(data_path, filename)
            # 删除文件
            os.remove(file_path)
            print(f"Deleted {file_path}")

    print("Finished cleaning up .Zone.Identifier files.")
