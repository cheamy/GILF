import pandas as pd

# 从文本文件读取数据
with open('RGB.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
data = {'scene': [], 'PSNR': [], 'SSIM': []}
current_scene = None

for line in lines:
    if line.startswith('Dataset----'):
        parts = line.split(',')
        current_scene = parts[0].split('----')[-1].strip()
        psnr = float(parts[1].split('---')[-1].strip())
        ssim = float(parts[2].split('---')[-1].strip())
        data['scene'].append(current_scene)
        data['PSNR'].append(psnr)
        data['SSIM'].append(ssim)

# 创建DataFrame
df = pd.DataFrame(data)

# 将DataFrame写入Excel文件
df.to_excel('output.xlsx', index=False)