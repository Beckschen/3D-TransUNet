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


import argparse
import os
import pickle

import torch
import torch.distributed
import yaml

from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights

from nn_transunet.default_configuration import get_default_configuration


def main():
    parser = argparse.ArgumentParser()
    # change batch_size in nnUNetTrainer.py self.batch_size = stage_plans['batch_size']; can change batch_size=1 if world_size=16
    parser.add_argument("--network", default="3d_fullres", type=str)
    parser.add_argument("--network_trainer", default="nnUNetTrainerV2_DDP")
    parser.add_argument("--task", default="Task801_WORD", help="can be task name or task id")
    parser.add_argument("--task_pretrained", default="Task801_WORD", help="option Task801_WORD, Task850_ABD1K")
    
    parser.add_argument("--fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("--model", default="Generic_UNet", type=str)
    parser.add_argument("--disable_ds", default=False, type=bool)
    parser.add_argument("--resume", default='local_latest', type=str) # auto
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--dbs", required=False, default=False, action="store_true", help="distribute batch size. If "
                                                                                          "True then whatever "
                                                                                          "batch_size is in plans will "
                                                                                          "be distributed over DDP "
                                                                                          "models, if False then each "
                                                                                          "model will have batch_size "
                                                                                          "for a total of "
                                                                                          "GPUs*batch_size")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the vlaidation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--valbest", required=False, default=False, action="store_true", help="")
    parser.add_argument("--vallatest", required=False, default=False, action="store_true", help="")
    
    parser.add_argument("--find_lr", required=False, default=False, action="store_true", help="")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files. Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    parser.add_argument('--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_num_epochs', default=1000, type=int)
    parser.add_argument('--initial_lr', default=0.1, type=float)
    parser.add_argument('--min_lr', default=0, type=float)
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8), from MAE ft')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default), from MAE ft')
    parser.add_argument('--weight_decay', default=3e-5, type=float)

    parser.add_argument("--local-rank", type=int) # must pass
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--total_batch_size', default=None, type=int, help='node rank for distributed training')
    
    parser.add_argument('--hdfs_base', default='', type=str)
    parser.add_argument('--optim_name', default='', type=str) # sgd as default, otherwise will have effect in nnUNetTrainerV2_DDP
    parser.add_argument('--lrschedule', default='', type=str) # polylr as default
    parser.add_argument('--warmup_epochs', default=None, type=int)

    parser.add_argument("--val_final", default=False, action="store_true", help="")
    parser.add_argument("--is_ssl", default=False, action="store_true", help="SSL pretraining")
    parser.add_argument("--is_spatial_aug_only", default=False, action="store_true", help="SSL pretraining")
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--loss_name', default='', type=str)
    parser.add_argument('--plan_update', default='', type=str)
    parser.add_argument('--crop_size', nargs='+', type=int, default=None,
                        help='input to network')
    parser.add_argument('--reclip', nargs='+', type=int)
    parser.add_argument("--pretrained", default=False, action="store_true", help="")
    parser.add_argument("--disable_decoder", default=False, action="store_true", help="disable decoder of mae network")
    parser.add_argument("--model_params", default={})
    parser.add_argument('--layer_decay', default=1.0, type=float, help="layer-wise dacay for lr")
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1), drop_path=0 for MAE pretrain')
    parser.add_argument("--find_zero_weight_decay", default=False, action="store_true", help="")

    parser.add_argument('--n_class', default=17, type=int, help="17 for WORD including background")
    parser.add_argument('--deep_supervision_scales', nargs='+', type=int, default=[], help='remember to align with pat_emb_stride for z')
    parser.add_argument("--fix_ds_net_numpool", default=False, action="store_true", help="")
    parser.add_argument("--skip_grad_nan", default=False, action="store_true", help="skip_grad_nan in nnUNetTrainerV2_DDP")
    parser.add_argument("--merge_femur", default=False, action="store_true", help="merge class-15 and class-16 (head of femur) during training")
    parser.add_argument("--is_sigmoid", default=False, action="store_true", help="is_sigmoid for output instead of softmax")
    parser.add_argument('--max_loss_cal', default='', type=str, help="v0, v1")
    
    
    
    # debug

    args_config, _ = parser.parse_known_args() # expect return 'remaining' standing for the namspace from launch? but not...
   
    # if args_config.config:
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    model_params = cfg.get("model_params", {})
    
    args = parser.parse_args() # update args from yaml

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data
    deterministic = args.deterministic
    valbest = args.valbest
    vallatest = args.vallatest
    find_lr = args.find_lr
    val_folder = args.val_folder

    fp32 = args.fp32
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    # init DDP, in favor of multi-node training
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    torch.distributed.barrier()

    if fold.startswith('all'):
        pass
    else:
        fold = int(fold)


    if not args.hdfs_base:
        args.hdfs_base = network + '_' + args.model
    plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier, hdfs_base=args.hdfs_base, plan_update=args.plan_update)
    resolution_index = 1

    if args.config.find('500Region') != -1:
        batch_dice = True
        resolution_index = 0

    if '005' in plans_file or '004' in plans_file or '002' in plans_file or '001' in plans_file:
        resolution_index = 0

    info = pickle.load(open(plans_file, "rb"))
    plan_data = {}
    plan_data["plans"] = info
    patch_size = plan_data['plans']['plans_per_stage'][resolution_index]['patch_size']
    if args.crop_size is None:
        args.crop_size = patch_size

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in meddec.model_training")

    if args.pretrained: 
        fold_name = 'all' if isinstance(fold, str) and fold.startswith('all') else 'fold_'+str(fold)
        init_ckpt_base = model_params['init_ckpt']

        pretrained_output_folder =  output_folder_name.replace(args.hdfs_base, init_ckpt_base) + '/' + fold_name
        pretrained_ckpt_path = pretrained_output_folder +  "/model_latest.model" # check network_trainer.load_latest_checkpoint()

        if args.task_pretrained!= args.task:
            pretrained_output_folder = pretrained_output_folder.replace(args.task, args.task_pretrained)

            pretrained_ckpt_path = pretrained_ckpt_path.replace(args.task, args.task_pretrained)
        os.makedirs(pretrained_output_folder, exist_ok=True)

        if args.local_rank==0:
            downloaded = pretrained_ckpt_path if os.path.exists(pretrained_ckpt_path) else False
            if not downloaded:
                print("pretrained weights not existed in both local and remote")
            else:
                print("pretrained weights downloaded to remote")

        torch.distributed.barrier() # make sure each rank has updated model_params
        model_params['init_ckpt'] = pretrained_ckpt_path
        print("###########update model_params['init_ckpt']: ", model_params['init_ckpt'])

    trainer = trainer_class(plans_file, fold, local_rank=args.local_rank, output_folder=output_folder_name,
                            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                            unpack_data=decompress_data, deterministic=deterministic, fp16=not fp32,
                            distribute_batch_size=args.dbs, 
                            # model=args.model, disable_ds=args.disable_ds, resume=args.resume,
                            input_size=args.crop_size,
                            args=args) # for V2

    if args.disable_saving:
        trainer.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = False  # whether or not to save checkpoint_latest
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint

    trainer.initialize(not validation_only)

    resume_epoch = 0
    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
                # trainer.load_checkpoint_ram()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                # new training without pretraine weights, do nothing
                pass
            
            if args.resume == 'auto':
                fold_name = fold if isinstance(fold, str) and fold.startswith('all') else 'fold_'+str(fold)
                output_folder =  output_folder_name + '/' + fold_name
                assert trainer.output_folder == output_folder, "output_folder path are not consistent!" # check if consistent!
                if args.local_rank == 0: os.makedirs(output_folder, exist_ok=True)
                ckpt_path = output_folder +  "/model_latest.model" # check network_trainer.load_latest_checkpoint()
                if args.local_rank == 0: # downloaded for each node
                    resume = ckpt_path if os.path.exists(ckpt_path) else False
                torch.distributed.barrier()
                resume = ckpt_path if os.path.exists(ckpt_path) else False # set resume flag for every process
                if resume: # will find ckpt_path (find 1. best 2. final 3. latest) in network_trainer
                    print("### resume, load_latest_checkpoint")
                    trainer.load_latest_checkpoint() # load ckpt, opt, amp, epoch, plot.... check network_trainer.load_latest_checkpoint(), which will call nnUNetTrainerV2_DDP.load_checkpoint_ram()
                    resume_epoch = trainer.epoch
            elif args.resume == 'local_latest':
                fold_name = fold if isinstance(fold, str) and fold.startswith('all') else 'fold_'+str(fold)
                output_folder =  output_folder_name + '/' + fold_name
                assert trainer.output_folder == output_folder, "output_folder path are not consistent!" # check if consistent!
                if args.local_rank == 0: os.makedirs(output_folder, exist_ok=True)
                torch.distributed.barrier()
                ckpt_path = output_folder +  "/model_latest.model" # check network_trainer.load_latest_checkpoint()
                resume = ckpt_path if os.path.exists(ckpt_path) else False # set resume flag for every process

                if resume: # will find ckpt_path (find 1. best 2. final 3. latest) in network_trainer
                    print("### resume, load_latest_checkpoint")
                    trainer.load_latest_checkpoint() # load ckpt, opt, amp, epoch, plot.... check network_trainer.load_latest_checkpoint(), which will call nnUNetTrainerV2_DDP.load_checkpoint_ram()
                    resume_epoch = trainer.epoch
            trainer.run_training()
            
        else:
            
            if valbest:
                trainer.load_best_checkpoint(train=False)
            elif vallatest:
                trainer.load_latest_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

        trainer.network.eval()

        # predict validation !!!!!!
        if args.val_final or vallatest:
            trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                         run_postprocessing_on_folds=not disable_postprocessing_on_folds)
        

        if network == '3d_lowres':
            raise NotImplementedError
    
    # torch.distributed.barrier()
    
    print("######### run_training_DDP done!")
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
