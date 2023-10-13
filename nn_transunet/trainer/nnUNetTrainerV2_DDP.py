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



#installed package
from genericpath import exists
import os
import shutil
from _warnings import warn
from collections import OrderedDict
from multiprocessing import Pool
from time import sleep, time
from typing import Tuple
import numpy as np
import torch
import torch.distributed as dist
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.utilities.distributed import awesome_allgather_function
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from torch import nn, distributed
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from tqdm import trange

from  ..trainer.nnUNetTrainerV2 import nnUNetTrainerV2, InitWeights_He


from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, subfiles, isfile, load_pickle, \
    save_json
from ..data.data_augmentation_moreDA import get_moreDA_augmentation
from ..data.dataset_loading import unpack_dataset
from ..data.default_data_augmentation import default_2D_augmentation_params, get_patch_size, default_3D_augmentation_params



from ..networks.transunet3d_model import Generic_TransUNet_max_ppbp





class nnUNetTrainerV2_DDP(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False, 
                 model="Generic_UNet", 
                 input_size=(64, 160, 160),
                 args=None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)
        self.init_args = (
            plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
            deterministic, distribute_batch_size, fp16)
        assert args is not None
        self.args = args
        if self.args.config.find('500Region') != -1:
            self.regions = {"whole tumor": (1, 2, 3),
                            "tumor core": (2, 3),
                            "enhancing tumor": (3,) # correct
                            }
            if self.args.config.find('500RegionFix') != -1:
                self.regions = {"whole tumor": (1, 2, 3),
                                "tumor core": (2, 3),
                                "enhancing tumor": (2,) # fig 1: the innermost tumor, but this is a bug!!
                                }
            self.regions_class_order = (1, 2, 3)
        

        self.layer_decay = args.layer_decay
        self.lr_scheduler_name = args.lrschedule # [ TO DO ]
        self.reclip = args.reclip
        self.warmup_epochs = args.warmup_epochs
        self.min_lr = args.min_lr
        self.is_spatial_aug_only = args.is_spatial_aug_only
        if "model_params" in args:
            self.model_params = args.model_params
        else:
            self.model_params = {}
        


        self.optim_name = args.optim_name
        self.find_zero_weight_decay = args.find_zero_weight_decay
        self.model = args.model
        self.resume = args.resume
        self.input_size=input_size
        self.disable_ds=args.disable_ds
        self.max_num_epochs = args.max_num_epochs # set 8 gpu training
        self.initial_lr = args.initial_lr # 8 * 0.01
        self.weight_decay = args.weight_decay # 3e-5 in nnUNetTrainer.py
        self.save_every = 1 # prev 50

        self.distribute_batch_size = distribute_batch_size

        np.random.seed(local_rank)
        torch.manual_seed(local_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_rank)
        self.local_rank = local_rank

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # dist.init_process_group(backend='nccl', init_method='env://') # init outside

        self.loss = None
        self.ce_loss = RobustCrossEntropyLoss()

        self.global_batch_size = None  # we need to know this to properly steer oversample

    def setup_DA_params_BraTSRegions(self):
        # nnUNetTrainerV2.setup_DA_params(self)
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-90. / 360 * 2. * np.pi, 90. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-90. / 360 * 2. * np.pi, 90. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-90. / 360 * 2. * np.pi, 90. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["p_rot"] = 0.3

        self.data_aug_params["scale_range"] = (0.65, 1.6)
        self.data_aug_params["p_scale"] = 0.3
        self.data_aug_params["independent_scale_factor_for_each_axis"] = True
        self.data_aug_params["p_independent_scale_per_axis"] = 0.3

        self.data_aug_params["do_elastic"] = True
        self.data_aug_params["p_eldef"] = 0.3 # LMH 0.2 -> 0.3 according to paper
        self.data_aug_params["eldef_deformation_scale"] = (0, 0.25)

        self.data_aug_params["do_additive_brightness"] = True
        self.data_aug_params["additive_brightness_mu"] = 0
        self.data_aug_params["additive_brightness_sigma"] = 0.2
        self.data_aug_params["additive_brightness_p_per_sample"] = 0.3
        self.data_aug_params["additive_brightness_p_per_channel"] = 0.5

        self.data_aug_params['gamma_range'] = (0.5, 1.6)

        self.data_aug_params['num_cached_per_thread'] = 4

    def set_batch_size_and_oversample(self):
        batch_sizes = []
        oversample_percents = []

        world_size = self.args.world_size# dist.get_world_size()
        my_rank = self.args.rank # dist.get_rank() # not local_rank

        if self.args.total_batch_size: # actually it is global_batch_size
            # reset the batch_size per gpu accordingly
            self.batch_size = self.args.total_batch_size // world_size

        
        # if self.args.local_rank == 0:
        #     print("total_batch_size: %d, updated batch_size per gpu %d, world_size %d" % (self.args.total_batch_size, self.batch_size, world_size))
        if self.distribute_batch_size: # set total batch_size to 16 will be fine...
            self.global_batch_size = self.batch_size
        else:
            self.global_batch_size = self.batch_size * world_size

        batch_size_per_GPU = np.ceil(self.batch_size / world_size).astype(int)  # probably 1
        
        for rank in range(world_size):
            if self.distribute_batch_size:
                if (rank + 1) * batch_size_per_GPU > self.batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - self.batch_size)
                else:
                    batch_size = batch_size_per_GPU
            else:
                batch_size = self.batch_size

            batch_sizes.append(batch_size)

            sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
            sample_id_high = np.sum(batch_sizes)

            if sample_id_high / self.global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percents.append(0.0)
            elif sample_id_low / self.global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percents.append(1.0)
            else:
                percent_covered_by_this_rank = sample_id_high / self.global_batch_size - sample_id_low / self.global_batch_size
                oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                sample_id_low / self.global_batch_size) / percent_covered_by_this_rank)
                oversample_percents.append(oversample_percent_here)

        print("worker", my_rank, "oversample", oversample_percents[my_rank])
        print("worker", my_rank, "batch_size", batch_sizes[my_rank])
        # batch_sizes [self.batch_size]*world_size
        self.batch_size = batch_sizes[my_rank] 
        self.oversample_foreground_percent = oversample_percents[my_rank]

    def save_checkpoint(self, fname, save_optimizer=True):
        if self.local_rank == 0:
            super().save_checkpoint(fname, save_optimizer)

    def plot_progress(self):
        if self.local_rank == 0:
            super().plot_progress()

    def print_to_log_file(self, *args, also_print_to_console=True):
        if self.local_rank == 0:
            super().print_to_log_file(*args, also_print_to_console=also_print_to_console)

    def process_plans(self, plans):
        super().process_plans(plans)
        if (self.patch_size != self.args.crop_size).any():
            self.patch_size = self.args.crop_size
        
        self.set_batch_size_and_oversample()

        if self.args.config.find('500Region') != -1:
            self.num_classes = len(self.regions) # only care about foreground (compatible with sigmoid)

    def initialize(self, training=True, force_load_plans=False):
        """
        :param training:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            if self.args.config.find('500Region') != -1: # BraTSRegions_moreDA
                self.setup_DA_params_BraTSRegions()

            
            if hasattr(self.args, 'deep_supervision_scales') and len(self.args.deep_supervision_scales)>0:
                self.deep_supervision_scales = self.args.deep_supervision_scales # overwrite setup_DA_params() from nnUNetTrainerV2
            

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] + 
                                                    "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    distributed.barrier()
                else:
                    # distributed.barrier()
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # setting weights for deep supervision losses
                if not self.model.startswith("Generic") and self.args.fix_ds_net_numpool:
                    # here is a bug, which need to be fixed!
                    net_numpool = len(self.deep_supervision_scales)
                else:
                    net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
                weights[~mask] = 0
                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                if self.disable_ds:
                    self.ds_loss_weights[0]=1
                    self.ds_loss_weights[1:]=0

                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
                print("seeds train", seeds_train)
                print("seeds_val", seeds_val)
                # add more transform into dataloader

                if self.reclip:
                    lb, ub, means, stds = self.reclip[0], self.reclip[1], self.intensity_properties[0]['mean'], self.intensity_properties[0]['sd']
                    self.reclip = [lb, ub, means, stds]


                if self.args.config.find('500Region') != -1: # BraTSRegions_moreDA
                    from nnunet.training.data_augmentation.data_augmentation_insaneDA2 import get_insaneDA_augmentation2
                    self.tr_gen, self.val_gen = get_insaneDA_augmentation2(
                                                    self.dl_tr, self.dl_val,
                                                    self.data_aug_params[
                                                        'patch_size_for_spatialtransform'],
                                                    self.data_aug_params,
                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                    pin_memory=self.pin_memory,
                                                    regions=self.regions
                                                ) # such that we can get val
                else:
                    self.tr_gen, self.val_gen = get_moreDA_augmentation(
                                                self.dl_tr, self.dl_val,
                                                    self.data_aug_params[
                                                        'patch_size_for_spatialtransform'],
                                                    self.data_aug_params,
                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                    seeds_train=seeds_train,
                                                    seeds_val=seeds_val,
                                                    pin_memory=self.pin_memory,
                                                    is_spatial_aug_only=self.is_spatial_aug_only,
                                                    reclip=self.reclip
                                                )
                
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                # in network_trainer.py tr_keys = val_keys = list(self.dataset.keys()) if fold=='all'
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False) 
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)
            if self.local_rank==0:
                print(self.network)
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """

        if self.model.startswith("Generic"):
            if self.threeD:
                conv_op = nn.Conv3d
                dropout_op = nn.Dropout3d
                norm_op = nn.InstanceNorm3d

            else:
                conv_op = nn.Conv2d
                dropout_op = nn.Dropout2d
                norm_op = nn.InstanceNorm2d

            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
            dropout_op_kwargs = {'p': 0, 'inplace': True}
            net_nonlin = nn.LeakyReLU # nnunet v1, not softmax..., interesting..., but compute_loss has consider the softmax..
            net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
            do_ds = not self.disable_ds
            if not do_ds: print("disable ds")

            if self.model == 'Generic_TransUNet_max_ppbp':
                self.network = Generic_TransUNet_max_ppbp(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, do_ds, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, 
                                    convolutional_upsampling= False if ('is_fam' in self.model_params.keys() and self.model_params['is_fam']) else True, #  default True,
                                    patch_size=self.args.crop_size, 
                                    **self.model_params)
                

            
            self.network.inference_apply_nonlin = nn.Sigmoid() if self.model == 'Generic_UNet_large' else softmax_helper


        else:

            raise NotImplementedError

        total = sum([param.nelement() for param in self.network.parameters()])
        print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))
        # assert 1==2, self.network
        if torch.cuda.is_available():
            self.network.cuda()


    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if self.optim_name == 'adam':
            self.optimizer= torch.optim.Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay,)
        elif self.optim_name == 'adamw':
            self.optimizer= torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay,)
        else:
            self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        print("initialized optimizer ", self.optim_name)
        if self.lr_scheduler_name == 'warmup_cosine':
            print("initialized lr_scheduler ", self.lr_scheduler_name)
            from ..optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
            self.lr_scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_num_epochs, eta_min=self.min_lr)
        else:
            self.lr_scheduler = None

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        
        if self.lr_scheduler is not None:
            from torch.optim import lr_scheduler
            # LinearWarmupCosineAnnealingLR inherit _LRScheduler
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                if self.epoch > 0:  # otherwise self.train_loss_MA is None
                    self.lr_scheduler.step(self.train_loss_MA)
            else:
                # print("maybe_update_lr(): update lr through lr_scheduler for self.epoch+1")
                self.lr_scheduler.step(self.epoch + 1)
            
        else:
            if self.warmup_epochs is not None:
                from network_trainer import warmup_poly_lr
                self.optimizer.param_groups[0]['lr'] = warmup_poly_lr(ep, self.max_num_epochs, self.warmup_epochs, self.initial_lr, 0.9)
            else:
                from network_trainer import poly_lr
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))


    def on_after_backward(self):
        # added by jieneng: https://github.com/PyTorchLightning/pytorch-lightning/issues/4956
        valid_gradients = True
        for name, param in self.network.parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            self.optimizer.zero_grad()


    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        if self.args.merge_femur:
            target[target==16] = 15

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=None)
            target = to_cuda(target, gpu_id=None)

        self.optimizer.zero_grad()


        if self.fp16:
            with torch.autograd.set_detect_anomaly(True):
                with autocast():
                    is_c2f = self.args.model.find('C2F') != -1
                    is_max = self.args.model_params.get('is_max', self.args.model.find('max') != -1)
                    is_max_hungarian = ('is_max_hungarian' in self.args.model_params.keys() and self.args.model_params['is_max_hungarian'])
                    is_max_cls = ('is_max_cls' in self.args.model_params.keys() and self.args.model_params['is_max_cls'])
                    is_max_ds =('is_max_ds' in self.args.model_params.keys() and self.args.model_params['is_max_ds'])
                    point_rend =('point_rend' in self.args.model_params.keys() and self.args.model_params['point_rend'])
                    num_point_rend = self.args.model_params['num_point_rend'] if point_rend else None
                    no_object_weight = self.args.model_params['no_object_weight'] if 'no_object_weight' in self.args.model_params.keys() else None

                    if is_c2f:
                        output = self.network(data, target[0] if is_c2f else None)
                    else:
                        output = self.network(data) # transunet output [2, 17, 64, 160, 160]
                    del data
                    if is_max and not ('is_masking_argmax' in self.args.model_params.keys() and self.args.model_params['is_masking_argmax']):
                        self.args.is_sigmoid = True
                    
                    if self.disable_ds:
                        if not (is_max or is_c2f):
                            if isinstance(output, (tuple, list)):
                                output = output[0]
                        if isinstance(target, (tuple, list)):
                            target = target[0]


                    l = self.compute_loss(output, target, is_max, is_c2f, self.args.is_sigmoid, is_max_hungarian, is_max_ds, point_rend, num_point_rend, no_object_weight)

                if do_backprop:
                    self.amp_grad_scaler.scale(l).backward()
                    """
                    for name, param in self.network.named_parameters():
                        if param.grad is None:
                            print("unused paramter found in ", name)
                    """
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    if self.args.skip_grad_nan:
                        self.on_after_backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
            
        else:
            output = self.network(data)
            del data
            l = self.compute_loss(output, target)

            if do_backprop:
                l.backward()
                # self.on_after_backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        
        if run_online_evaluation:
            with torch.no_grad():
                self.run_online_evaluation(output, target) # compute dice for train sample?

        del target

        return l.detach().cpu().numpy()

    def compute_loss(self, output, target, is_max=False, is_c2f=False, is_sigmoid=False, is_max_hungarian=False, is_max_ds=False, point_rend=False, num_point_rend=None, no_object_weight=None):
        total_loss, smooth, do_fg = None, 1e-5, False
        if self.args.config.find('500Region') != -1:
            assert is_sigmoid, "BraTS region should be compatible with sigmoid activation"
            smooth = 0
            do_fg = True
        
        if isinstance(output, (tuple, list, dict)):
            len_ds = 1+len(output['aux_outputs']) if isinstance(output, dict) else len(output)
            if self.args.max_loss_cal == 'exp': # for max_ds or ds
                max_ds_loss_weights = np.array([1 / (2 ** i) for i in range(len_ds)])
                max_ds_loss_weights = max_ds_loss_weights / max_ds_loss_weights.sum()
            else:
                max_ds_loss_weights = [1] * (len_ds) # previous had a bug with exp weight for 'v0' ..

        if is_max and is_max_hungarian:
            # output: a dict of ['pred_logits', 'pred_masks', 'aux_outputs']
            if not self.disable_ds:
                output_ds, target_ds = output[1:], target[1:]
                output, target = output[0], target[0]

            aux_outputs = output['aux_outputs'] # a list of dicts of ['pred_logits', 'pred_masks']
            
            if self.args.config.find('500Region') != -1:
                target_onehot = target
            else:
                target_onehot = torch.zeros_like(target.repeat(1, self.num_classes, 1, 1, 1), device=target.device)
                target_onehot.scatter_(1, target.long(), 1)

            target_sum = target_onehot.flatten(2).sum(dim=2) # (b, 3)
            targets = []
            for b in range(len(target_onehot)):
                target_mask = target_onehot[b][target_sum[b] > 0] # (K, D, H, W)
                target_label = torch.nonzero(target_sum[b] > 0).squeeze(1) # (K)
                targets.append({'labels':target_label, 'masks':target_mask})

            from ..networks.transunet3d_model import HungarianMatcher3D, compute_loss_hungarian
            if ('cost_weight' in self.args.model_params.keys() and self.args.model_params['cost_weight']):
                self.cost_weight = self.args.model_params['cost_weight']
            else:
                self.cost_weight = [2.0, 5.0, 5.0]
            matcher = HungarianMatcher3D(
                    cost_class=self.cost_weight[0], # 2.0
                    cost_mask=self.cost_weight[1],
                    cost_dice=self.cost_weight[2],
                )
            outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}
            loss_list = []
            loss_final = compute_loss_hungarian(outputs_without_aux, targets, 0, matcher, self.num_classes, point_rend, num_point_rend, no_object_weight=no_object_weight, cost_weight=self.cost_weight)
            loss_list.append(max_ds_loss_weights[0] * loss_final)
            if is_max_ds and "aux_outputs" in output:
                for i, aux_outputs in enumerate(output["aux_outputs"][::-1]): # reverse order
                    loss_aux = compute_loss_hungarian(aux_outputs, targets, i+1, matcher, self.num_classes, point_rend, num_point_rend, no_object_weight=no_object_weight, cost_weight=self.cost_weight)
                    loss_list.append(max_ds_loss_weights[i+1] *loss_aux)
            else:
                return loss_final

            if self.args.max_loss_cal == '':
                total_loss = sum(loss_list) / len(loss_list)
            elif self.args.max_loss_cal == 'v1':
                total_loss = (loss_list[0] + sum(loss_list[1:])/len(loss_list[1:])) / 2
            elif self.args.max_loss_cal in ['v0', 'exp']:
                total_loss = sum(loss_list) # weighted with 1.0 or exp
            
            if not self.disable_ds:
                total_loss = self.ds_loss_weights[0] * total_loss # replace with loss from Transformer decoder
                # print("i {} loss {}".format(0, total_loss))
                for i in range(len(output_ds)):
                    axes = tuple(range(2, len(output_ds[i].size()))) # (2,3,4)

                    with autocast(enabled=False):
                        output_act = output_ds[i].sigmoid() if is_sigmoid else softmax_helper(output_ds[i]) # bug occurs here..
                    tp, fp, fn, _ = get_tp_fp_fn_tn(output_act, target_ds[i], axes, mask=None) # target_ds[i] is one-hot, tp: (b, n_class)
                    if do_fg: # 
                        nominator = 2 * tp # already fg
                        denominator = 2 * tp + fp + fn
                    else:
                        nominator = 2 * tp[:, 1:] # do fg
                        denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]

                    if self.batch_dice:
                        # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                        nominator = awesome_allgather_function.apply(nominator) # (b * n_gpu, n_class)
                        denominator = awesome_allgather_function.apply(denominator)
                        nominator = nominator.sum(0)
                        denominator = denominator.sum(0)
                    
                    dice_loss = (- (nominator + smooth) / (denominator + smooth)).mean()

                    if is_sigmoid:
                        if self.args.config.find('500Region') != -1:
                            target_onehot = target_ds[i]
                        else:
                            target_onehot = torch.zeros_like(output_ds[i], device=output_ds[i].device)
                            target_onehot.scatter_(1, target_ds[i].long(), 1) # target is a tuple
                            assert (torch.argmax(target_onehot, dim=1) == target_ds[i][:, 0].long()).all()
                        ce_loss = F.binary_cross_entropy_with_logits(output_ds[i], target_onehot)
                    else:
                        ce_loss = self.ce_loss(output_ds[i], target_ds[i][:, 0].long())
                    
                    # print("i {} loss {}".format(i+1, self.ds_loss_weights[i+1] * (ce_loss + dice_loss) ))
                    total_loss += self.ds_loss_weights[i+1] * (0.5*ce_loss + 0.5*dice_loss) # NOTE: weight shift..

            return total_loss

        if self.disable_ds and (is_max or is_c2f) and isinstance(output, (tuple, list)): # mask2former deep supervision per layer
            # output is a tuple of aux_output, but not for target

            loss_list = []
            assert isinstance(target, torch.Tensor)

            for i in range(len(output)):
                axes = tuple(range(2, len(output[i].size())))
                output_act = nn.Sigmoid()(output[i]) if is_sigmoid else softmax_helper(output[i])
                # output_sigmoid (b, k, d, h, w), target (b, 1, d, h, w)
                # get the tp, fp and fn terms we need
                tp, fp, fn, _ = get_tp_fp_fn_tn(output_act, target, axes, mask=None) # e.g. tp = output * y_onehot
                # for dice, compute nominator and denominator so that we have to accumulate only 2 instead of 3 variables
                nominator = 2 * tp[:, 1:]
                denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]
                if self.batch_dice:
                    # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                    nominator = awesome_allgather_function.apply(nominator)
                    denominator = awesome_allgather_function.apply(denominator)
                    nominator = nominator.sum(0)
                    denominator = denominator.sum(0)

                if is_sigmoid:
                    target_onehot = torch.zeros_like(output[i], device=output[i].device)
                    target_onehot.scatter_(1, target.long(), 1)
                    assert (torch.argmax(target_onehot, dim=1) == target[:, 0].long()).all()
                    ce_loss = F.binary_cross_entropy_with_logits(output[i], target_onehot)
                else:
                    ce_loss = self.ce_loss(output[i], target[:, 0].long())

                dice_loss = (- (nominator + smooth) / (denominator + smooth)).mean() # we smooth by 1e-5 to penalize false positives if tp is 0

                cur_loss = max_ds_loss_weights[i] * (ce_loss + dice_loss)

                loss_list.append(cur_loss)

                if total_loss is None:
                    total_loss = cur_loss
                else:
                    total_loss += cur_loss
                
                if i==0:
                    final_loss = (ce_loss + dice_loss)
            
            # should have a self.ds_loss_weights for max?
            if self.args.max_loss_cal == '':
                total_loss /= len(output)
            elif self.args.max_loss_cal == 'v1': # bug
                total_loss /= len(output)
                total_loss += final_loss
            elif self.args.max_loss_cal == 'v2':
                del total_loss
                total_loss = (loss_list[0] + sum(loss_list[1:])) / 2
            elif self.args.max_loss_cal in ['v0', 'exp']:
                pass

            return total_loss

        elif self.disable_ds and not isinstance(output, (tuple, list)):
            # assert 1==2, torch.sum(output[0,:,10,50,50])
            # Starting here it gets spicy!
            axes = tuple(range(2, len(output.size())))
            # network does not do softmax. We need to do softmax for dice
            if is_sigmoid:
                output_softmax = nn.Sigmoid()(output)
            else:
                output_softmax = softmax_helper(output)
            # get the tp, fp and fn terms we need
            tp, fp, fn, _ = get_tp_fp_fn_tn(output_softmax, target, axes, mask=None)
            # for dice, compute nominator and denominator so that we have to accumulate only 2 instead of 3 variables
            # do_bg=False in nnUNetTrainer -> [:, 1:]
            nominator = 2 * tp[:, 1:]
            denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]

            if self.batch_dice:
                # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                nominator = awesome_allgather_function.apply(nominator)
                denominator = awesome_allgather_function.apply(denominator)
                nominator = nominator.sum(0)
                denominator = denominator.sum(0)
            else:
                pass
            
            if is_sigmoid:
                target_onehot = torch.zeros_like(output, device=output.device)
                target_onehot.scatter_(1, target.long(), 1)
                assert (torch.argmax(target_onehot, dim=1) == target[:, 0].long()).all()
                ce_loss = F.binary_cross_entropy_with_logits(output, target_onehot)
            else:
                ce_loss = self.ce_loss(output, target[:, 0].long())

            dice_loss = (- (nominator + smooth) / (denominator + smooth)).mean()
            total_loss = ce_loss + dice_loss
            return total_loss
        else:
            for i in range(len(output)):
                # Starting here it gets spicy!
                axes = tuple(range(2, len(output[i].size())))

                # network does not do softmax. We need to do softmax for dice
                output_act = nn.Sigmoid()(output[i]) if is_sigmoid else softmax_helper(output[i])

                # get the tp, fp and fn terms we need
                tp, fp, fn, _ = get_tp_fp_fn_tn(output_act, target[i], axes, mask=None)
                # for dice, compute nominator and denominator so that we have to accumulate only 2 instead of 3 variables
                # do_bg=False in nnUNetTrainer -> [:, 1:]
                if do_fg: # 
                    nominator = 2 * tp # already fg
                    denominator = 2 * tp + fp + fn
                else:
                    nominator = 2 * tp[:, 1:] # do fg
                    denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]
                
                if self.batch_dice:
                    # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                    nominator = awesome_allgather_function.apply(nominator)
                    denominator = awesome_allgather_function.apply(denominator)
                    nominator = nominator.sum(0)
                    denominator = denominator.sum(0)
                else:
                    pass

                if is_sigmoid:
                    if self.args.config.find('500Region') != -1:
                        target_onehot = target[i]
                    else:
                        target_onehot = torch.zeros_like(output[i], device=output[i].device)
                        target_onehot.scatter_(1, target[i].long(), 1) # target is a tuple
                        assert (torch.argmax(target_onehot, dim=1) == target[i][:, 0].long()).all()
                    ce_loss = F.binary_cross_entropy_with_logits(output[i], target_onehot)
                else:
                    ce_loss = self.ce_loss(output[i], target[i][:, 0].long())

                dice_loss = (- (nominator + smooth) / (denominator + smooth)).mean()
                if total_loss is None:
                    total_loss = self.ds_loss_weights[i] * (ce_loss + dice_loss)
                else:
                    total_loss += self.ds_loss_weights[i] * (ce_loss + dice_loss)
            return total_loss
        
    def run_online_evaluation(self, output, target):
        if self.disable_ds:
            if ('is_max_hungarian' in self.args.model_params.keys() and self.args.model_params['is_max_hungarian']):
                # semantic_inference
                mask_cls, mask_pred = output["pred_logits"], output["pred_masks"]
                mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # filter out non-object class
                mask_pred = mask_pred.sigmoid()
                output = torch.einsum("bqc,bqdhw->bcdhw", mask_cls, mask_pred)

            elif self.args.model.find('max') != -1 or self.args.model.find('C2F') != -1:
                output = output[0]

            output = output.unsqueeze(0) # previously do_ds return a list for evaluation
            target = target.unsqueeze(0)
        else:
            # self.args.model_params['max_infer'] = 'general'
            if ('is_max_hungarian' in self.args.model_params.keys() and self.args.model_params['is_max_hungarian']):
                if 'max_infer' in self.args.model_params.keys() and self.args.model_params['max_infer']=='general':
                    mask_cls, mask_pred = output[0]["pred_logits"], output[0]["pred_masks"]
                    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # filter out non-object class, (b, q, n_cls)
                    mask_pred = mask_pred.sigmoid()
                    mask_cls_max, mask_cls_idx = torch.max(mask_cls, dim=-1) # (b, q)
                    B, _, D, H, W = mask_pred.shape
                    segm = torch.zeros((B, self.num_classes, D, H, W)).to(mask_cls.device)

                    mask_pred_hard = (mask_pred>0.5).long() # DEBUG !!!!!

                    mask_pred_q = torch.zeros_like(mask_pred_hard).to(mask_pred_hard.device)
                    for b in range(len(mask_cls)): # per batch
                        mask_cls_b_max, mask_cls_b_idx = mask_cls_max[b], mask_cls_idx[b]
                        # if b ==0:
                        #     print("----------", mask_cls_b_idx)
                        #     print("----------", mask_cls_b_max)

                        for q in range(20):
                            cur_cls = mask_cls_b_idx[q]
                            mask_pred_q[b][q][mask_pred_hard[b][q]==1] = cur_cls+1
                            
                        for i in range(self.num_classes):
                            mask_cls_b_max_i = mask_cls_b_max[mask_cls_b_idx==i] # (s) for each gt class, there might be several queries which are most likely to the class
                            
                            if len(mask_cls_b_max_i) > 0:
                                mask_cls_b_max_i_maxq, mask_cls_b_max_i_idxq = torch.max(mask_cls_b_max_i, dim=-1) # select one most likely query
                                segm[b, i] = mask_pred[b][mask_cls_b_max_i_idxq]
                            else:
                                assert 1==2, "len(mask_cls_b_max_i) == 0"
                    ###################################################
                    """
                    viz_output_folder = os.path.join(self.output_folder, 'viz_q')
                    if self.local_rank == 0:
                        import SimpleITK as sitk
                        os.makedirs(viz_output_folder, exist_ok=True)
                        for i in range(20):
                            pred = mask_pred_q[0, i].cpu().numpy()
                            pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
                            sitk.WriteImage(pred_sitk, os.path.join(viz_output_folder, str(i)+'_pred.nii.gz'))
                        
                        targ_hot = target[0][0] # (3, d, h, w)
                        targ = torch.zeros(targ_hot.shape[1:]).to(targ_hot.device)
                        for kk in range(3):
                            targ[targ_hot[kk]==1] = kk+1
                        targ = targ.cpu().numpy()
                        targ_sitk = sitk.GetImageFromArray(targ.astype(np.uint8))
                        sitk.WriteImage(targ_sitk, os.path.join(viz_output_folder, 'targ.nii.gz'))
                    """
                    ###################################################



                else: # semantic_inference
                    mask_cls, mask_pred = output[0]["pred_logits"], output[0]["pred_masks"]
                    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # filter out non-object class
                    mask_pred = mask_pred.sigmoid()
                    output[0] = torch.einsum("bqc,bqdhw->bcdhw", mask_cls, mask_pred) # replace 


        if self.args.config.find('500Region') != -1: # only care about foreground
            if 'max_infer' in self.args.model_params.keys() and self.args.model_params['max_infer']=='general':
                out_hard = (segm > 0.5).float()
            elif 'max_infer' in self.args.model_params.keys() and self.args.model_params['max_infer'].startswith('thres'):
                out_sigmoid = torch.sigmoid(output[0])
                out_hard = (out_sigmoid > float(self.args.model_params['max_infer'].split('_')[-1])).float()
            elif self.args.model.find('max') != -1:
                # out_sigmoid = torch.sigmoid(output[0]) #  max w/o activation, but semantic_inference has.
                #  out_sigmoid[0,:,50,50,50] -> [0.5, 0.5, 0.5], if this pixel is background, then the channel will predict 0.5?
                # out_hard = (out_sigmoid > 0.6).float() # 0.6 perform the best!!!
                out_hard = (output[0] > 0.5).float()
            else:
                out_sigmoid = torch.sigmoid(output[0])
                out_hard = (out_sigmoid > 0.5).float()
            """
            viz_output_folder = os.path.join(self.output_folder, 'viz')
            if self.local_rank == 0:
                import SimpleITK as sitk
                os.makedirs(viz_output_folder, exist_ok=True)
                # print("viz_output_folder", viz_output_folder)
                # for j in range(20):
                #     mask_pred = (mask_pred > 0.5).float()
                #     pred = mask_pred[0, j].cpu().numpy()
                for i in range(3):
                    pred = out_sigmoid[0, i].cpu().numpy()
                    targ = target[0][0, i].cpu().numpy()
                    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
                    sitk.WriteImage(pred_sitk, os.path.join(viz_output_folder, str(i)+'_pred.nii.gz'))
                    targ_sitk = sitk.GetImageFromArray(targ.astype(np.uint8))
                    sitk.WriteImage(targ_sitk, os.path.join(viz_output_folder, str(i)+'_targ.nii.gz'))
                assert 1==2
            """
            axes = (2, 3, 4)
            tp_hard, fp_hard, fn_hard, _ = get_tp_fp_fn_tn(out_hard, target[0], axes=axes) # tp:(b, n_class)

        else:
            # with torch.no_grad():
            num_classes = output[0].shape[1]
            output_seg = output[0].argmax(1) # not applicable to 500
            target = target[0][:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)
            # tp_hard, fp_hard, fn_hard = get_tp_fp_fn((output_softmax > (1 / num_classes)).float(), target,
            #                                         axes, None)
            # print_if_rank0("before allgather", tp_hard.shape)


        tp_hard = tp_hard.sum(0, keepdim=False)[None] # (1, n_fg)
        fp_hard = fp_hard.sum(0, keepdim=False)[None]
        fn_hard = fn_hard.sum(0, keepdim=False)[None]
        tp_hard = awesome_allgather_function.apply(tp_hard) # (n_gpu, n_fg)
        fp_hard = awesome_allgather_function.apply(fp_hard)
        fn_hard = awesome_allgather_function.apply(fn_hard)
        tp_hard = tp_hard.detach().cpu().numpy().sum(0)
        fp_hard = fp_hard.detach().cpu().numpy().sum(0)
        fn_hard = fn_hard.detach().cpu().numpy().sum(0)
        # dice1 = (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)
        self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """

        if self.local_rank == 0:
            self.save_debug_information()

        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        if not self.disable_ds:
            net.do_ds = True

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            
            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True) # run_online_evaluation=True
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end() # network_trainer: plot_progress() maybe_update_lr(self.epoch) maybe_save_checkpoint() update_eval_criterion_MA()
            
            torch.distributed.barrier()
            if self.resume == 'auto':
                from ..utils.dist_utils import check_call_hdfs_command, mkdir_hdfs
                hdfs_folder=self.output_folder.replace("/opt/tiger/project/data/nnUNet_trained_models/UNet_IN_NANFang/", "") # with fold
                hdfs_path="/home/byte_arnold_hl_vc/user/jienengchen/projects/transunet/nnunet/snapshots/" + hdfs_folder # a folder name, with fold
                if self.local_rank==0:
                    mkdir_hdfs(hdfs_path)
                torch.distributed.barrier()
                local_path = self.output_folder + '/model_latest.model'
                put_cmd = f"-put -f {local_path} {hdfs_path}"
                if self.local_rank==0:
                    check_call_hdfs_command(put_cmd)
                local_best_path = self.output_folder + '/model_best.model'
                if self.local_rank==0 and os.path.exists(local_best_path):
                    put_cmd = f"-put -f {local_best_path} {hdfs_path}"
                    check_call_hdfs_command(put_cmd)

            
            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1 # previous saving will also +1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: 
            if self.local_rank==0: print("saving final...")
            self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))

        if self.local_rank == 0:
            # now we can delete latest as it will be identical with final
            pass

        net.do_ds = ds

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError(
                    "We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        all_keys = list(self.dataset_val.keys())
        my_keys = all_keys[self.local_rank::dist.get_world_size()]
        # we cannot simply iterate over all_keys because we need to know pred_gt_tuples and valid_labels of all cases
        # for evaluation (which is done by local rank 0)
        for k in my_keys:
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])
            if k in my_keys:
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']

                    print(k, data.shape)
                    data[-1][data[-1] == -1] = 0

                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                         do_mirroring=do_mirroring,
                                                                                         mirror_axes=mirror_axes,
                                                                                         use_sliding_window=use_sliding_window,
                                                                                         step_size=step_size,
                                                                                         use_gaussian=use_gaussian,
                                                                                         all_in_gpu=all_in_gpu,
                                                                                         mixed_precision=self.fp16)[1]

                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating obejcts
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                             ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                               properties, interpolation_order,
                                                               self.regions_class_order,
                                                               None, None,
                                                               softmax_fname, None, force_separate_z,
                                                               interpolation_order_z),
                                                              )
                                                             )
                                   )

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        distributed.barrier()

        if self.local_rank == 0:
            # evaluate raw predictions
            self.print_to_log_file("evaluation of raw predictions")
            task = self.dataset_directory.split("/")[-1]
            job_name = self.experiment_name
            _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                                 json_output_file=join(output_folder, "summary.json"),
                                 json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                                 json_author="Fabian",
                                 json_task=task, num_threads=default_num_threads)

            if run_postprocessing_on_folds:
                # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
                # except the largest connected component for each class. To see if this improves results, we do this for all
                # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
                # have this applied during inference as well
                self.print_to_log_file("determining postprocessing")
                determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                         final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
                # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
                # They are always in that folder, even if no postprocessing as applied!

            # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
            # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
            # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
            # be used later
            gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
            maybe_mkdir_p(gt_nifti_folder)
            for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
                success = False
                attempts = 0
                e = None
                while not success and attempts < 10:
                    try:
                        shutil.copy(f, gt_nifti_folder)
                        success = True
                    except OSError as e:
                        attempts += 1
                        sleep(1)
                if not success:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    if e is not None:
                        raise e

        self.network.train(current_mode)
        net.do_ds = ds

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel, DDP))
        assert isinstance(self.network, tuple(valid))
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False
        try:
            ret = net.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                use_sliding_window=use_sliding_window, step_size=step_size,
                                patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                mixed_precision=mixed_precision)
        except:
            print("run?")
        net.do_ds = ds
        return ret

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                print("duh")
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        try:
            self.network.load_state_dict(new_state_dict)
        except:
            print("************ loaded error, switch strict=False")
            self.network.load_state_dict(new_state_dict, strict=False)

        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            
            if optimizer_state_dict is not None:
                print("optimizer_state_dict loaded")
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                print("load_checkpoint_ram: step lr_scheduler to update lr for self.epoch!")
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]


    def on_epoch_end(self):

        # appear in nnUNetTrainer, accumulate "Average global foreground Dice:" from results in self.run_online_evaluation
        # to update all_val_eval_metrics, which can update val_eval_criterion_MA to update best_val_eval_criterion_MA
        # and then save_best_checkpoint; so the model_best is the best results of self.val_gen
        self.finish_online_evaluation()

        self.plot_progress()

        self.maybe_update_lr(self.epoch)

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience() # save model_best here

        return continue_training



