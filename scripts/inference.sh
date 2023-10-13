export nnUNet_N_proc_DA=36
export nnUNet_codebase="/home/lxh/code/nnUNet" # BUGGGG
export nnUNet_raw_data_base="/data1/data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/data1/data/nnUNet_raw_data_base/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/meijieru/workspace/transunet_max/results/ablations"

config='configs/Synapse/decoder_only.yaml'
export save_folder=./results/inference/test/task008/encoderonly

NUM_GPUS=8

inference() {
	fold=$1
	gpu=$2
	extra=${@:3}

	echo "inference: fold ${fold} on gpu ${gpu}, extra ${extra}"
	CUDA_VISIBLE_DEVICES=${gpu} \
		python3 inference.py --config=${config} \
		--fold=${fold} --raw_data_folder ${subset} \
		--save_folder=${save_folder}/fold_${fold} ${extra}
}

compute_metric() {
	fold=$1
	pred_dir=${2:-${save_folder}/fold_${fold}/}
	extra=${@:3}

	echo "compute_metric: fold ${fold}, extra ${extra}"
	python3 measure_dice.py \
		--config=${config} --fold=${fold} \
		--raw_data_dir=${raw_data_dir} \
		--pred_dir=${pred_dir} ${extra}
}

fold=$1
if [[ ${fold} == "all" ]]; then
	gpu=${2:-${gpu}}
else
	gpu=$((${fold} % ${NUM_GPUS}))
	gpu=${2:-${gpu}}
fi
extra=${@:3}

echo "extra: ${extra}"

# 5 fold eval
subset='imagesTr'

inference ${fold} ${gpu} ${extra}
compute_metric ${fold}

echo "finished: inference: fold ${fold} on ${config}"
exit

# # test set eval
# subset='imagesTs'
# inference ${fold} ${gpu} ${extra} --disable_split
# compute_metric ${fold} ${save_folder}/fold_${fold}/ --eval_mode Ts

# multi_save_folder=./results/inference/test/task201/encoderonly/fold_${fold},./results/inference/test/task201/decoderonly/fold_${fold}
# compute_metric ${fold} ${multi_save_folder} --eval_mode Ts
