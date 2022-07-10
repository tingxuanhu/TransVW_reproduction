import SimpleITK as sitk
import numpy as np
import os

reader = sitk.ImageSeriesReader()  # Read series of image files into a SimpleITK image.

parent_dir = "/home/data/tingxuan/DatasetForTest"
dicom_dir = os.listdir(parent_dir)
print(dicom_dir)

for scans in dicom_dir:
    if scans != 'train' and scans != 'test' and scans != 'validation':
        path = os.path.join(parent_dir, scans)
        print(path)
        dicom_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        # print(type(image_array))  # <class 'numpy.ndarray'>
        # print(image_array.shape)  # z, y, x

        image_copy = image_array.copy()
        image_copy = np.transpose(image_copy, (2, 1, 0))
        # print(image_copy.shape)  # x, y, z

        # change HU range and normalize the distribution
        hu_min = -1000.0
        hu_max = 1000.0

        image_copy[image_copy < hu_min] = hu_min
        image_copy[image_copy > hu_max] = hu_max

        image_copy = (image_copy - hu_min) / (hu_max - hu_min)
        # print(image_copy)
        print(image_copy.shape)
        # print(image_copy.shape[0])
        print('--------------------------------------------------------------------------')

        # save npy file
        path_to_save_npy = f'/home/data/tingxuan/DatasetForTest/train/{scans}_{image_copy.shape[2]}.npy'
        np.save(path_to_save_npy, image_copy)
