import time
import numpy as np
from deeplab import DeeplabV3
import os

if __name__ == '__main__':
    deeplab = DeeplabV3()
    mode = 'predict'
    count = True
    name_classes = ['background', 'peak']
    simplify = True
    onnx_save_path = 'model_data/models.onnx'
    data_path = './experi_data'
    with open(os.path.join(data_path, "val", "flood_name.txt"), 'r') as f:
        test_flood_lines = f.readlines()

    if mode == 'predict':
        for i in range(len(test_flood_lines)):
            with open('result_best.txt', 'a') as f:
                f.write(test_flood_lines[i].split()[0]+'.npy'+"\n")
            print(test_flood_lines[i].split()[0]+'.npy')
            img = np.load(os.path.join(data_path, "val", "flood_GY_npy_ori", test_flood_lines[i].split()[0]+'.npy'))
            r_image = deeplab.detect_image(img, count=count, name_classes=name_classes)
            np.save(os.path.join('./predict_val', test_flood_lines[i].split()[0]+'.npy'), r_image)
