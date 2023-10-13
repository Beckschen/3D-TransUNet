import os
nnUNet_raw_data_path="/opt/tiger/project/data/nnUNet_raw_data"

# only support single phase now
def convert(compressed_path):
    os.system('tar -xvf {}'.format(compressed_path))
    os.system('mv {} {}'.format(compressed_path.replace('.tar', ''), nnUNet_raw_data_path))
    task_name = os.path.basename(compressed_path).replace('.tar', '')
    new_task_name = task_name.replace('Task', 'Task0')
    task_id = new_task_name.split('_')[0].replace('Task', '')
    os.system('mv {} {}'.format(nnUNet_raw_data_path+'/'+task_name, nnUNet_raw_data_path+'/'+new_task_name))
    os.system('rm {}/._*'.format(nnUNet_raw_data_path+'/'+new_task_name))
    imagesTr_dir = nnUNet_raw_data_path+'/'+new_task_name+'/imagesTr'
    imagesTs_dir = nnUNet_raw_data_path+'/'+new_task_name+'/imagesTs'
    labelsTr_dir = nnUNet_raw_data_path+'/'+new_task_name+'/labelsTr'
    os.system('rm {}/._*'.format(imagesTr_dir))
    os.system('rm {}/._*'.format(imagesTs_dir))
    os.system('rm {}/._*'.format(labelsTr_dir))
    for name in os.listdir(imagesTr_dir):
        name2 = name.replace('.nii.gz', '_0000.nii.gz')
        os.system('mv {} {}'.format(imagesTr_dir+'/'+name, imagesTr_dir+'/'+name2))
    
    for name in os.listdir(imagesTs_dir):
        name2 = name.replace('.nii.gz', '_0000.nii.gz')
        os.system('mv {} {}'.format(imagesTs_dir+'/'+name, imagesTs_dir+'/'+name2))



c_path = "/opt/tiger/project/data/MSD/Task08_HepaticVessel.tar"
# c_path = "/opt/tiger/project/data/MSD/Task07_Pancreas.tar"
# c_path = "/opt/tiger/project/data/MSD/Task03_Liver.tar"
# c_path = "/opt/tiger/project/data/MSD/Task01_BrainTumour.tar"
convert(c_path)