import SimpleITK as sitk
import numpy as np
import os

reader = sitk.ImageSeriesReader()  # Read series of image files into a SimpleITK image.

parent_dir = "/home/data/tingxuan/DatasetForTest"
dicom_dir = os.listdir(parent_dir)
print(dicom_dir)

for type in dicom_dir:
    path = os.path.join(parent_dir, type)
    print(path)
    dicom = os.listdir(path)
    print(dicom)

    count = 0

    for dicom_name in dicom:
        dicom_names = os.path.join(path, dicom_name)
        print(dicom_names)
        target_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dicom_names)
        reader.SetFileNames(target_names)
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

        # imageCHWD = np.expand_dims(image_copy, axis=0)
        # print(imageCHWD.shape)
        # print('--------------------------------------------------------------------------')

        # save npy file
        if type == '0':
            path_to_save_npy = f'/home/data/tingxuan/demo/0/data_{type}_00{count}_{image_copy.shape[2]}.npy'
        elif type == '1':
            path_to_save_npy = f'/home/data/tingxuan/demo/1/data_{type}_00{count}_{image_copy.shape[2]}.npy'
        else:
            path_to_save_npy = f'/home/data/tingxuan/demo/data_{type}_00{count}_{image_copy.shape[2]}.npy'

        count += 1

        print(path_to_save_npy)
        np.save(path_to_save_npy, image_copy)
        print('--------------------------------------------------------------------------')


