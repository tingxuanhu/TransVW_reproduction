import os
import numpy as np
import pydicom
from numba import jit


def list_file(directory):
    file_inner = os.listdir(directory)
    file_list = []
    for file in file_inner:
        dcm_file = os.path.join(directory, file)
        file_list.append(dcm_file)
    return file_list


def hu_convert(slices):
    # z, y, x
    images = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    images[images == -2048] = 0

    for slice_num in range(len(slices)):
        intercept = slices[slice_num].RescaleIntercept
        slope = slices[slice_num].RescaleSlope
        if slope != 1:
            images[slice_num] = slope * images[slice_num].astype(np.float64)
            images[slice_num] = images[slice_num].astype(np.int16)
        images[slice_num] = images[slice_num] + np.int16(intercept)
    return images


@jit(nopython=True)
def calc(img_temp, rows, cols, min_val, max_val):
    for i in np.arange(rows):
        for j in np.arange(cols):
            # 避免除以0的报错
            if max_val - min_val == 0:
                result = 1
            else:
                result = max_val - min_val
            img_temp[i, j] = int((img_temp[i, j] - min_val) / result * 255)


def set_ww_and_wc(img_data, win_width, win_center):
    min_val = (2 * win_center - win_width) / 2.0 + 0.5
    max_val = (2 * win_center + win_width) / 2.0 + 0.5
    for index in range(len(img_data)):
        img_temp = img_data[index]
        rows, cols = img_temp.shape
        calc(img_temp, rows, cols, min_val, max_val)
        img_temp[img_temp < 0] = 0
        img_temp[img_temp > 255] = 255
        img_data[index] = img_temp
    return img_data


def save_as_dcm_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_and_hu_convert(path_to_load, path_to_save, win_width, win_center):
    slices = [pydicom.read_file(os.path.join(path_to_load, s), force=True) for s in os.listdir(path_to_load)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    patient_slices_hu = hu_convert(slices)
    patient_pixels = set_ww_and_wc(patient_slices_hu, win_width, win_center)
    for idx, ith_value in enumerate(slices):
        ith_value.Pixel_data = patient_pixels[idx].tobytes()
        ith_value.save_as(path_to_save + f"/I_00{idx}.dcm")
    # return patient_pixels


# 纵隔窗窗宽窗位
win_width, win_center = 300, 40

# DICOM文件夹下存放的是待转化的原始像素dcm文件
dcm_path = "/home/data/tingxuan/DICOM"
dcm_list = list_file(dcm_path)
print(dcm_list)    # dcm_list -->  ['/home/data/tingxuan/DICOM/UDC0HSKF', '/home/data/tingxuan/DICOM/RWRRPZ3V', ...]

for directory in dcm_list:
    # 存新的dcm文件的文件夹(形式上与原来像素格式的文件保持一致)
    dir_for_save = directory.split("/")[-1].split("/")[0]
    print(dir_for_save)

    if not os.path.exists(os.path.join('/home/data/tingxuan/HU/', dir_for_save)):
        os.makedirs(os.path.join('/home/data/tingxuan/HU/', dir_for_save))

    path_to_save = os.path.join('/home/data/tingxuan/HU/', dir_for_save)

    # patient0_slices
    load_and_hu_convert(directory, path_to_save, win_width, win_center)



