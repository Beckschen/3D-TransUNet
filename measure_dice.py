import argparse
import yaml
import os
import numpy as np
import SimpleITK as sitk

from medpy import metric
from tqdm import tqdm
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')
parser.add_argument("--eval_mode", default='Val', type=str,)
parser.add_argument("--fold", default=0, help='0, 1, ..., 5 or \'all\'', )
parser.add_argument("--raw_data_dir", default='')
parser.add_argument("--pred_dir", default='')
parser.add_argument("--disable_split", default=False, action="store_true", help='just use raw_data_dir, do not use split!')
parser.add_argument("--which_ds", default=None, type=int, help="")
parser.add_argument("--num_classes", default=None, type=int, help="")
parser.add_argument("--evaluate_regions", default=False, action="store_true", help="only validate for 500Region")
args, remaining = parser.parse_known_args()

if args.config:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
else:
    raise NotImplementedError

if args.raw_data_dir:
    args.eval_mode=os.path.basename(args.raw_data_dir).replace('images', '')


def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    return mask_new


def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        # jc = metric.jc(pred, gt)
        # asd = metric.binary.asd(pred, gt, voxelspacing=voxel_spacing)
        # return np.array([dice, jc, asd, hd95])
        return np.array([dice])
    else:
        return np.array([0.0])
        # return np.array([0.0, 50])

def each_cases_metric(gt, pred, voxel_spacing):
    if args.num_classes is not None:
        classes_num = args.num_classes # gt.max() + 1
    else:
        classes_num =  gt.max().item() + 1
        assert classes_num - int(classes_num) == 0
        classes_num = int(classes_num)

    class_wise_metric = np.zeros((classes_num-1, 1))

    if args.config.find('500Region') != -1:
        regions = {"whole tumor": (1, 2, 3),
                "tumor core": (2, 3),
                "enhancing tumor": (3,)
                }
        for i, r_tuple in enumerate(regions.values()):
            pred_tmp = create_region_from_mask(pred, r_tuple)
            gt_tmp = create_region_from_mask(gt, r_tuple)
            print(i, r_tuple, pred_tmp.sum(), gt_tmp.sum())
            tmp = cal_metric(pred_tmp==1, gt_tmp==1, voxel_spacing)
            class_wise_metric[i, ...] = tmp
    else:
        for cls in range(1, classes_num):
            tmp = cal_metric(pred==cls, gt==cls, voxel_spacing)
            class_wise_metric[cls-1, ...] = tmp
    return class_wise_metric

network, task, network_trainer, hdfs_base = cfg['network'], cfg['task'], cfg['network_trainer'], cfg['hdfs_base']

fold_name = args.fold if str(args.fold).startswith('all') else 'fold_'+str(args.fold)
all_results = []

label_dir = os.getenv('nnUNet_preprocessed')+"/"+task+"/gt_segmentations/" # "/home/SENSETIME/luoxiangde.vendor/Projects/ABDSeg/data/ABDSeg/data/labelsTs/"
raw_dir = os.getenv('nnUNet_raw_data_base') +"/nnUNet_raw_data/"+task+"/"


if args.pred_dir is None:
    if args.raw_data_dir:
        pred_dir = os.getenv('nnUNet_raw_data_base')+"/nnUNet_inference/"+task+"/"+cfg['hdfs_base']+'/'+fold_name+'/'+os.path.basename(args.raw_data_dir)+"/"
    else:
        pred_dir = os.getenv('nnUNet_raw_data_base')+"/nnUNet_inference/"+task+"/"+cfg['hdfs_base']+"/" + fold_name+"/"
else:
    pred_dir = args.pred_dir

if args.which_ds is not None:
    pred_dir = pred_dir + 'ds_' + str(args.which_ds)+'/'

pred_dirs = None
if pred_dir.find(',') != -1:
    pred_dirs = pred_dir.split(",")
    print(f'Fusing pred from: {pred_dirs}')


if "brats" in task.lower():
    with open("split/brats_splits_final.pkl", "rb") as f:
        data = pickle.load(f)
elif "hepaticvessel" in task.lower():
    with open("split/hv_splits_final.pkl", "rb") as f:
        data = pickle.load(f)

eval_files = data[int(args.fold)]['val']
eval_files = [i+'.nii.gz' for i in eval_files]

r_ind = 0
for ind, case in enumerate(tqdm(eval_files)):
    if not case.endswith(".nii.gz"):
        continue
    gt_path = label_dir+case.replace("_pred", "")
    if not os.path.exists(gt_path):
        gt_path = raw_dir + 'labels'+args.eval_mode+'/' + case.replace("_pred", "").replace("_0000", "")
    if not os.path.exists(gt_path):
        print("not existed", gt_path)
        continue
    assert os.path.exists(gt_path), gt_path
    gt_itk = sitk.ReadImage(gt_path)
    voxel_spacing = (gt_itk.GetSpacing()[2], gt_itk.GetSpacing()[0], gt_itk.GetSpacing()[1])
    gt_array = sitk.GetArrayFromImage(gt_itk)
    if pred_dirs is not None:
        prob_maps = [
            np.load(os.path.join(pd, case.replace(".nii.gz", ".npz")))['softmax']
            for pd in pred_dirs
        ]
        pred_array = np.argmax(np.mean(prob_maps, axis=0), axis=0)
        pred_array = pred_array.astype(gt_array.dtype)
    else:
        pred_itk = sitk.ReadImage(pred_dir+case)
        pred_array = sitk.GetArrayFromImage(pred_itk) 

    per_result = each_cases_metric(gt_array, pred_array, voxel_spacing)
    all_results.append(per_result)
    r_ind += 1


print(f'num examples: {len(all_results)}')
print(
        "model {} ds {} fold {} dice per class {} phase {} dice average {}".format(
            args.config,
            args.which_ds,
            args.fold,
            np.mean(all_results, axis=0),
            os.path.basename(args.raw_data_dir),
            np.mean(all_results)
            ))
num_classes = max([val.shape[0] for val in all_results])
per_class_results = [
        np.mean([
            val[ic] for val in all_results if ic < val.shape[0]], axis=0).squeeze().item()
        for ic in range(num_classes)
]
print(( *per_class_results, np.mean(per_class_results)))
