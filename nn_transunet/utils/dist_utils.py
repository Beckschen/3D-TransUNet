import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import logging
import subprocess

import os
import random
import multiprocessing

import torch.distributed as t_dist
from collections import OrderedDict
# from flop_count.flop_count import flop_count
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import List

_logger = logging.getLogger(__name__)

class WarmupPolyLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """
    def __init__(
            self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int, warmup_start_lr: float = 0.0, last_epoch: int = -1
        ) -> None:
            """
            Args:
                optimizer: wrapped optimizer.
                warmup_epochs: number of warmup iterations.
                max_epochs: total number of training iterations.
                last_epoch: the index of last epoch.
            Returns:
                None
            """
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.warmup_start_lr = warmup_start_lr
            super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))

        return [(base_lr * (1.0 - progress)) for base_lr in self.base_lrs]


    # def lr_lambda(self, step):
    #     """
    #     new lr = lr * lambda 
    #     return lambda
    #     """
    #     if step < self.warmup_steps:
    #         return float(step) / float(max(1.0, self.warmup_steps))
    #     progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
    #     return (1.0 - progress) ** 0.9


def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == 'agc':
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."
        
class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)

def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


def data_raw_convert_Synapse13_To_8(data_dir):
    """convert the label of 13 (14) class to 8 (9) class"""
    cvt_dict = [8, 4, 3, 2, 6, 11, 1, 7]
    for folder in ["labelsTr"]:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            print("convert ", filename)
            vol_path = os.path.join(data_dir, folder, filename)
            label = sitk.ReadImage(vol_path)
            spacing = label.GetSpacing()
            size = label.GetSize()
            origin = label.GetOrigin()
            direction = label.GetDirection()
            # print("size {} origin {} direction {}".format(size, origin, direction))

            label = sitk.GetArrayFromImage(label)
            label_new = np.zeros_like(label)

            for dst_idx, src_idx_val in enumerate(cvt_dict):
                label_new[label==src_idx_val] = dst_idx+1

            lab_itk = sitk.GetImageFromArray(label_new)
            lab_itk.SetSpacing(spacing)
            # lab_itk.SetSize(size)
            lab_itk.SetOrigin(origin)
            lab_itk.SetDirection(direction)
            os.makedirs(os.path.join(data_dir, folder+"_8"), exist_ok=True)
            sitk.WriteImage(lab_itk, os.path.join(data_dir, folder+"_8", filename))

def postpro_maxcc():
    # to do: get max K connected components as postprocessing
    pass


def get_flops(model, test_data):
    batch_size = test_data.shape[0]
    flop_dict, _ = flop_count(model, (test_data,))
    msg = 'model_flops' + '\t' + str(sum(flop_dict.values()) / batch_size) + 'G'+ '\t params:' + str(
        sum([m.numel() for m in model.parameters()])) + '\n-----------------'
    return msg
    # logging.info("counting: {}".format(msg))
    # assert 1==2, msg



def mkdir_hdfs(dirpath, raise_exception=False):
    """mkdir hdfs directory"""
    try:
        cmd = '-mkdir -p {}'.format(dirpath)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = 'Failed to mkdir {} in HDFS: {}'.format(dirpath, e)
        if raise_exception:
            raise ValueError(msg)
        else:
            logging.error(msg)
        return False

def dist_init(port=2333):
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    rank = int(os.environ['SLURM_PROCID'])
    world_size = os.environ['SLURM_NTASKS']
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = world_size
    os.environ['RANK'] = str(rank)

    t_dist.init_process_group(backend='nccl')

    return rank, int(world_size), gpu_id


def init_device(args):
    # Random seed setting
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True


    args.device = int(os.environ['LOCAL_RANK'])

    print(f"device: {args.device}")
    torch.cuda.set_device(args.device)
    t_dist.init_process_group(backend='nccl', init_method='env://')

    rank = t_dist.get_rank()
    size = t_dist.get_world_size()
    device = torch.device(f'cuda:{args.device}')
    print(f"rank: {rank}, world_size: {size}")

    print(f'=> Device: running distributed training with world size:{size}')
    # PyTorch version record
    print(f'=> PyTorch Version: {torch.__version__}\n')
    t_dist.barrier()
    return rank, size, device



_HADOOP_COMMAND_TEMPLATE = 'hadoop fs {command}'
_SUPPORTED_HDFS_PATH_PREFIXES = ('hdfs://', 'ufs://')


def _get_hdfs_command(command):
    """return hadoop fs command"""
    return _HADOOP_COMMAND_TEMPLATE.format(command=command)


def check_call_hdfs_command(command):
    """check call hdfs command"""
    import shlex
    hdfs_command = _get_hdfs_command(command)
    subprocess.check_call(shlex.split(hdfs_command))


def download_from_hdfs(src_path, dst_path, raise_exception=False):
    """download src_path from hdfs to local dst_path"""
    if not has_hdfs_path_prefix(src_path):
        raise ValueError(
            'Input src_path {} is not a valid hdfs path'.format(src_path))
    if has_hdfs_path_prefix(dst_path):
        raise ValueError(
            'Input dst_path {} is a hdfs path, not a path for local FS'.format(dst_path))

    try:
        cmd = '-get {} {}'.format(src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = 'Failed to download src {} to dst {}: {}'.format(src_path, dst_path, e)
        if raise_exception:
            raise ValueError(msg)
        else:
            logging.error(msg)
        return False


def upload_to_hdfs(src_path, dst_path, overwrite=False, raise_exception=False):
    """Upload src_path to hdfs dst_path"""
    if not os.path.exists(src_path):
        raise IOError('Input src_path {} not found in local storage'.format(src_path))
    if not has_hdfs_path_prefix(dst_path):
        raise ValueError('Input dst_path {} is not a hdfs path'.format(dst_path))

    try:
        cmd = '-put -f' if overwrite else '-put'
        cmd = '{} {} {}'.format(cmd, src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = 'Failed to upload src {} to dst {}: {}'.format(src_path, dst_path, e)
        if raise_exception:
            raise ValueError(msg)
        else:
            logging.error(msg)
        return False


def has_hdfs_path_prefix(filepath):
    """Check if input filepath has hdfs prefix"""
    for prefix in _SUPPORTED_HDFS_PATH_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True, is_resume_metric=False):
    # timm helpers
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            # if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
            #     if log_info:
            #         _logger.info('Restoring AMP loss scaler state from checkpoint...')
                # loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key]) # for NativeScaler
            if loss_scaler is not None and 'amp_scaler' in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                else:
                    print('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint['amp_scaler'])
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        if 'metric_values' in checkpoint and is_resume_metric:
            return resume_epoch, checkpoint['metric_values']
        else:
            return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()



class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        # loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
        self._scaler.scale(loss).backward(create_graph=create_graph)
        # if clip_grad is not None:
        #     assert parameters is not None
        #     self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        #     dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score, target, dice_score=False):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        if dice_score:
            return loss
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, pp=False, pp_up=False, pp_cc=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            if pp or pp_up:
                out = torch.softmax(outputs, dim=1)
                if x != patch_size[0] or y != patch_size[1]:
                    out = nn.Upsample(size=x, mode='bicubic')(out)
                out = torch.argmax(out, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred=out
            else:
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) # can do upsample here
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
            prediction[ind] = pred
    
    if pp or pp_cc: # postprocessing in whole volume
        from skimage import measure
        pred_new = np.zeros_like(prediction).astype(np.uint8)
        for k in range(1, 9):
            pred = (prediction==k).astype(np.uint8)
            pred_ccs, num = measure.label(pred, return_num=True)
            max_label = 0
            max2_label = 0
            max_num = 0
            for i in range(1, num+1):  
                # 计算面积，保留最大面积对应的索引标签，然后返回二值化最大连通域
                if np.sum(pred_ccs == i) > max_num:
                    max_num = np.sum(pred_ccs == i)
                    max2_label = max_label
                    max_label = i
            if max2_label != 0:
                lcc = (pred_ccs == max_label) | (pred_ccs == max2_label)
            else:
                lcc = (pred_ccs == max_label)
            # assert 1==2, [np.shape(lcc), np.shape(prediction)]
            pred_new += (lcc.astype(np.uint8) * k)
        prediction = pred_new

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def test_single_volume_parallel(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # To Do
    dice_metric = DiceLoss(classes)
    image, label = image.squeeze(0), label.squeeze(0)
    test_batch_size = 8
    if len(image.shape) == 3:
        prediction = torch.zeros_like(label)
        for start_ind in range(0, image.shape[0], test_batch_size):
            slice = image[start_ind: start_ind+test_batch_size, :, :]
            # print("start_id {} total {} slice {}".format(start_ind, image.shape[0], slice.shape))
            x, y = slice.shape[1], slice.shape[2]
            input = slice.unsqueeze(1).cuda()
            if x != patch_size[0] or y != patch_size[1]:
                input = nn.Upsample(size=(patch_size[0], patch_size[1]), mode='bilinear')(input)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                if x != patch_size[0] or y != patch_size[1]:
                    outputs = nn.Upsample(size=(x, y), mode='bilinear')(outputs)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1) 
                prediction[start_ind: start_ind+test_batch_size] = out
    else:
        assert 1==2, "shape not matched in test_single_volume_parallel!"
    metric_list = []
    # label, prediction = label.cpu().detach().numpy(), prediction.cpu().detach().numpy()
    for i in range(1, classes):
        metric_list.append(dice_metric._dice_loss(prediction == i, label == i, dice_score=True).cpu().detach().numpy())
    return metric_list

if __name__ == "__main__":
    data_raw_convert_Synapse13_To_8("./data/btcv")