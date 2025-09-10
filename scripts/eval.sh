export nnUNet_codebase="../"
export nnUNet_raw_data_base="/data1/data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/data1/data/nnUNet_raw_data_base/nnUNet_preprocessed"
export RESULTS_FOLDER="/your_dir"

CONFIG="configs/GeTU500Region_128128128_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250.yaml"

python3 measure_dice.py --config=$CONFIG --pred_dir pred_dir/brats/fold0/ --num_classes 4 --fold 0
python3 measure_dice.py --config=$CONFIG --pred_dir pred_dir/brats/fold1/ --num_classes 4 --fold 1
python3 measure_dice.py --config=$CONFIG --pred_dir pred_dir/brats/fold2/ --num_classes 4 --fold 2
python3 measure_dice.py --config=$CONFIG --pred_dir pred_dir/brats/fold3/ --num_classes 4 --fold 3
python3 measure_dice.py --config=$CONFIG --pred_dir pred_dir/brats/fold4/ --num_classes 4 --fold 4

