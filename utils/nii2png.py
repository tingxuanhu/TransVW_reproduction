import nibabel as nib
import numpy as np
import os
import imageio


def list_file(directory):
    file_inner = os.listdir(directory)
    file_list = []
    for file in file_inner:
        dcm_file = os.path.join(directory, file)
        file_list.append(dcm_file)
    return file_list


def nii2img(nii_path, nii_list, img_path_to_save):
    for idx, nii in enumerate(nii_list):
        filename = nii.split("/")[-1].split(".")[0]
        path_to_save = os.path.join(img_path_to_save, filename)
        print(path_to_save)

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        nii_path_to_load = os.path.join(nii_path, nii)
        nii_img = nib.load(nii_path_to_load)
        nii_fdata = nii_img.get_fdata().astype(np.uint8)

        # change nii file to png file
        (x, y, z) = nii_img.shape

        for i in range(z):
            slice = nii_fdata[:, :, i]
            imageio.imwrite(os.path.join(path_to_save, f'patient{idx}_slice{i}.png'), slice)


nii_path = "/home/data/tingxuan/NII"
nii_list = list_file(nii_path)
print(nii_list)

img_path_to_save = '/home/data/tingxuan/PNG'
nii2img(nii_path, nii_list, img_path_to_save)

