import os
import shutil

path = '/home/data/tingxuan/Research'

path_to_save = '/home/data/tingxuan/DatasetForTest'

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

directory = os.listdir(path)
# print(directory)

for i in directory:
    if not i.endswith('.DS_Store'):
        people = os.path.join(path, i)
        num_person = os.listdir(people)
        # print(num_person)

        for num in num_person:
            if num == '1':
                image_file = os.path.join(people, num)
                # print(image_file)

                for img in os.listdir(image_file):
                    if img != '.DS_Store' and img != 'VERSION' and img != 'LOCKFILE' and img != 'DICOMDIR':
                        image_files = os.path.join(image_file, img)
                        # print(image_files)

                        for img_ in os.listdir(image_files):
                            if img_ != '.DS_Store' and img_ != 'VERSION':
                                # print(img_)

                                image_end = os.path.join(image_files, img_)
                                # print(image_end)

                                path_to_save_images = os.path.join(path_to_save, img_)
                                # print(path_to_save_images)

                                for figure in os.listdir(image_end):
                                    if figure != 'VERSION' and figure != '.DS_Store':
                                        # print(figure)
                                        target_figure = os.path.join(image_end, figure)

                                        if not os.path.exists(path_to_save_images):
                                            os.makedirs(path_to_save_images)

                                        judge_if_exist = os.path.join(path_to_save_images, figure)
                                        # print(judge_if_exist)

                                        if not os.path.exists(judge_if_exist):
                                            shutil.copy(target_figure, path_to_save_images)

                                print(img_ + 'is finished!')







