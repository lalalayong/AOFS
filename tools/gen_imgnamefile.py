import os

directory = r'dataset\evaluation\labelTxt'  # 将此行代码中的路径替换为目标目录的路径
filename = r'\dataset\imgnamefile.txt'

with open(filename, 'w') as f:
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            filename_without_ext = os.path.splitext(file)[0]
            f.write(filename_without_ext + '\n')
