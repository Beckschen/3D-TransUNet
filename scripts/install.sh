conda create --name nnunet python=3.8 -y
conda activate nnunet
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y

pip install numpy==1.23
pip install monai
pip install matplotlib batchgenerators pandas SimpleITK medpy tqdm

pip install segmentation_models_pytorch monai einops SimpleITK # installed
pip install pyyaml einops adamp gco-wrapper medpy nibabel tensorboardX tqdm ml_collections # in arnold, but now installed in venv
pip install fvcore

pip install nnunet
pip install 'git+https://github.com/facebookresearch/detectron2.git'

