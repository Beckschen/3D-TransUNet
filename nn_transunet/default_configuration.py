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


from nn_transunet.trainer.nnUNetTrainerV2 import nnUNetTrainerV2
from nn_transunet.trainer.nnUNetTrainer import nnUNetTrainer
from nn_transunet.trainer.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP

from batchgenerators.utilities.file_and_folder_operations import *
from nn_transunet.configuration import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
import os


def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    folder = folder[len(network_training_output_dir):]
    if folder.startswith("/"):
        folder = folder[1:]

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=(os.getenv('nnUNet_codebase'), "training", "network_training"),
                              base_module='nnunet.training.network_training',
                              hdfs_base='', plan_update=''):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"
    
    dataset_directory = join(preprocessing_output_dir, task)
    if network == '2d':
        plans_file = join(preprocessing_output_dir, task,
                          plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task,
                          plans_identifier + "_plans_3D.pkl")

    if plan_update: 
        plans_file = os.path.join('./plans', plan_update)
    
    
    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]
    # stage is 1 for 3d_fullres
    print([join(*search_in)], network_trainer, base_module)

    network_trainer_dict = {'nnUNetTrainer': nnUNetTrainer, 'nnUNetTrainerV2': nnUNetTrainerV2, 'nnUNetTrainerV2_DDP': nnUNetTrainerV2_DDP }
    if network_trainer ==  'nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm':
        from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2020.nnUNetTrainerV2BraTSRegions_moreDA import nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm
        network_trainer_dict['nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm'] = nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm
    
    trainer_class = network_trainer_dict[network_trainer]
    # trainer_class = nnUNetTrainer
    # trainer_class = recursive_find_python_class([join(*search_in)], network_trainer, current_module=base_module)
    output_folder_name = join(network_training_output_dir, task, network_trainer + "__" + plans_identifier, hdfs_base)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    # summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(
        dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
