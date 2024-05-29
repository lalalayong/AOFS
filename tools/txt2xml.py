import os

nwpu_classes = ['airplane',
                'ship',
                'storage-tank',
                'baseball-diamond',
                'tennis-court',
                'basketball-court',
                'ground-track-field',
                'harbor',
                'bridge',
                'vehicle']

# 遍历文件夹
folder_path = r"dataset\evaluation\annotations"

for filename in os.listdir(folder_path):
    # 判断文件类型是否为txt文件
    if filename.endswith('.txt'):
        # 读取txt文件内容
        with open(os.path.join(folder_path, filename), 'r') as f:
            lines = f.readlines()

        # 写入同名的xml文件中
        with open(os.path.join(folder_path, filename.replace('.txt', '.xml')), 'w') as f:
            filename = filename.replace('.txt', '.jpg')
            f.write('<annotation>\n')
            f.write(f'  <filename>{filename}</filename>\n')
            f.write(f'  <source>\n')
            f.write(f'      <database>NWPU VHR-10</database>\n')
            f.write(f'  </source>\n')
            f.write(f'  <segmented>0</segmented>\n')
            for line in lines:
                xmin, ymin, xmax, ymax, classes_id = line.split()
                classes = nwpu_classes[int(classes_id)-1]
                f.write(f'  <object>\n')
                f.write(f'      <name>{classes}</name>\n')
                f.write(f'      <pose>Unspecified</pose>\n')
                f.write(f'      <bndbox>\n')
                f.write(f'          <xmin>{xmin}</xmin>\n')
                f.write(f'          <ymin>{ymin}</ymin>\n')
                f.write(f'          <xmax>{xmax}</xmax>\n')
                f.write(f'          <ymax>{ymax}</ymax>\n')
                f.write(f'      </bndbox>\n')
                f.write(f'  </object>\n')
            f.write('</annotation>\n')
