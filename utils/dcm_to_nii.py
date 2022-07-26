import nibabel as nib
import numpy as np
import SimpleITK as sitk
from numba import jit


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


def dcm2nii(dcms_path, nii_path, ww, wc):

    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()

    # 2.将整合后的数据转为array，并获取dicom文件基本信息(经过测试  已经转化为HU值了  只需做窗宽窗位调整即可)
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    # print(np.max(image_array), np.min(image_array))   # --> 3071， -1024

    image_array = set_ww_and_wc(image_array, ww, wc)
    # print(np.max(image_array), np.min(image_array))  # --> 255, 0

    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z

    # 3.将array转为img，并保存为.nii
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)



if __name__ == '__main__':

    # 纵隔窗窗宽窗位
    win_width, win_center = 300, 40

    dcms_path = "/home/data/tingxuan/DICOM/UDC0HSKF/"

    nii_path = "/home/data/tingxuan/NII/UDC0HSKF.nii"

    dcm2nii(dcms_path, nii_path, win_width, win_center)

    img = nib.load(nii_path)
    img = img.get_fdata()



