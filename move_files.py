import os
import shutil

dir_path_1 = '/home/local-admin/Documents/projects/CLAM/features_TM1_512_level_0/pt_files'
dir_path_ori = '/media/local-admin/Elements/data_MAIA_2/TM/HC_MAIA_boite2_TM'
dir_path_2 = '/home/local-admin/Documents/projects/CLAM/features_TM2_512_level_0/pt_files'

for file_path in os.listdir(dir_path_1):
    data_name = file_path.split('.')[:-1][0]
    print(data_name)
    # check if current file_path is a file
    if os.path.isfile(os.path.join(dir_path_ori, data_name + '.svs')):
        shutil.move(os.path.join(dir_path_1, file_path), os.path.join(dir_path_2, file_path))
