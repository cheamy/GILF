import h5py
import scipy.io as sci
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

root_path = 'F:\stanford_half'
# mode = 'val'
image_path = os.path.join(root_path)
mat_path = os.path.join(root_path + '_mat')
if not os.path.exists(mat_path):
    os.mkdir(mat_path)
# scenes_list = os.listdir(image_path)
# scenes_list = ['bracelet', 'knights', 'tarot', 'tarot_small']
scenes_list = ['beans', 'bulldozer', 'bunny', 'chess', 'flowers', 'gem', 'treasure', 'truck']
for scene in tqdm(scenes_list):
    scene_path = os.path.join(image_path, scene)
    SAIs = os.listdir(scene_path)

    sample = Image.open(os.path.join(scene_path, SAIs[0]))
    width, height = sample.size

    LF_data = np.zeros((17, 17, height, width, 3))
    for SAI in SAIs:
        u = int(SAI.split('_')[1])
        v = int(SAI.split('_')[2])
        SAI_data = np.array(Image.open(os.path.join(scene_path, SAI))) / 255.0
        LF_data[u, v, :, :, :] = SAI_data
    LF_data = LF_data[::-1, ::-1, :, :, :]
    mat_dict = {'LF':LF_data}
    sci.savemat(os.path.join(mat_path, '{}.mat'.format(scene)), mat_dict)


