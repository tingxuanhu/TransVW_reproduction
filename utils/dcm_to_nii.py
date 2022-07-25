import os
import numpy as np
import pydicom
from numba import jit
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import glob
import nibabel as nib


def list_file(directory):
    file_inner = os.listdir(directory)
    file_list = []
    for file in file_inner:
        dcm_file = os.path.join(directory, file)
        file_list.append(dcm_file)
    return file_list


# def list_scans(file_list):
#     patient_scan_list = [[] for i in file_list]
#     for idx, scans in enumerate(file_list):
#         patient_list_scans = list_file(scans)
#         # print(patient_list_scans)
#         for scan in patient_list_scans:
#             patient_scan_list[idx].append(scan)
#     return patient_scan_list


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


# def save_as_dcm(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


def load_and_hu_convert(path, win_width, win_center):
    slices = [pydicom.read_file(os.path.join(path, s), force=True) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    patient_slices_hu = hu_convert(slices)
    patient_pixels = set_ww_and_wc(patient_slices_hu, win_width, win_center)

    for idx, ith_value in enumerate(slices):
        ith_value.Pixel_data = patient_pixels[idx].tobytes()
        ith_value.save_as(f"/home/data/tingxuan/dcmProcess/I_new_00{idx}.dcm")

    return patient_pixels


# 纵隔窗窗宽窗位
win_width = 300
win_center = 40
dcm_path = "/home/data/tingxuan/DICOM"
dcm_list = list_file(dcm_path)
print(dcm_list)
""" dcm_list
["/home/data/tingxuan/Research/guoyong/1/VVP0RFAZ/U23MAA4W/, ..., ..."]
"""

if __name__ == "__main__":
    patient0_slices = load_and_hu_convert(dcm_list[0], win_width, win_center)
    print(patient0_slices)










