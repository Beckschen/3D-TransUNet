#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
default_num_threads = 4
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 4
# do not modify these unless you know what you are doing
my_output_identifier = "UNet_IN_NANFang"
default_plans_identifier = "nnUNetPlansv2.1"
default_data_identifier = 'nnUNetData_plans_v2.1'
default_trainer = "nnUNetTrainer"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
# nnUNet_raw_data_base='/data/nnUNetFrame/DATASET/nnUNet_raw'
# nnUNet_preprocessed='/data/nnUNetFrame/DATASET/nnUNet_preprocessed'
# RESULTS_FOLDER='/data/nnUNetFrame/DATASET/nnUNet_trained_models'
nnUNet_raw_data_base=  os.getenv('nnUNet_raw_data_base')# +'/nnUNet_raw_data' # '/mnt/lustre/luoxiangde.vendor/projects/nnUNetFrame/DATASET/nnUNet_raw'
nnUNet_preprocessed= os.getenv('nnUNet_preprocessed')# '/mnt/lustre/luoxiangde.vendor/projects/nnUNetFrame/DATASET/nnUNet_preprocessed'
RESULTS_FOLDER= os.getenv('RESULTS_FOLDER') # '/mnt/lustre/luoxiangde.vendor/projects/nnUNetFrame/DATASET/nnUNet_trained_models'

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

base = nnUNet_raw_data_base
preprocessing_output_dir = nnUNet_preprocessed
network_training_output_dir_base = RESULTS_FOLDER

if base is not None:
    nnUNet_raw_data = join(base, "nnUNet_raw_data")
    nnUNet_cropped_data = join(base, "nnUNet_cropped_data")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
else:
    print("nnUNet_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    nnUNet_cropped_data = nnUNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information on how to set this "
          "up.")
    network_training_output_dir = None
