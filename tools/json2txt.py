import json
import os

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_txt_file(file_path, data):
    with open(file_path.replace('.json', '.txt'), 'w') as f:
        for item in data['shapes']:
            label = item['label']
            x1, y1, x2, y2, x3, y3, x4, y4 = item['points'][0][0], item['points'][0][1], item['points'][1][0], item['points'][1][1],item['points'][2][0],item['points'][2][1],item['points'][3][0],item['points'][3][1]
            f.write(f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {label} 0\n')

def main():
    folder_path = r'dataset\evaluation\labels'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            data = read_json_file(file_path)
            write_txt_file(file_path, data)

if __name__ == '__main__':
    main()
