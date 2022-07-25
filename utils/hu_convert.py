import os
import numpy as np
import pydicom as pyd
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk


def list_file(directory):
    file_inner = os.listdir(directory)
    file_list = []
    for file in file_inner:
        dcm_file = os.path.join(directory, file)
        file_list.append(dcm_file)
    return file_list


def list_scans(file_list):
    patient_scan_list = [[] for i in file_list]
    for idx, scans in enumerate(file_list):
        patient_list_scans = list_file(scans)
        # print(patient_list_scans)

        for scan in patient_list_scans:
            patient_scan_list[idx].append(scan)

    return patient_scan_list


dcm_path = "/home/data/tingxuan/DICOM"
# dcm_path = "/home/data/tingxuan/DICOM/AI313ROF"
dcm_list = list_file(dcm_path)
print(dcm_list)


scans_list = list_scans(dcm_list)
print(scans_list)

# rows, cols = list_scans(dcm_list)
# print(rows, cols)

# for file in dcm_list:
#     print(file)
#     img = pyd.read_file(file)
#     img_array = sitk.GetArrayFromImage(sitk.ReadImage(file))
#     print(img_array)
#     print(np.shape(img_array))
#     print('-----------------------------------------')
#
#     print(img.RescaleSlope)
#     print(img.RescaleIntercept)
#     HU = np.dot(img_array, img.RescaleSlope) + img.RescaleSlope
#     print(HU)
#
#     break
