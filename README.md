# GLEAM
Greedy Learning for Large-scale Accelerated MRI Reconstruction. Open source implementation of https://arxiv.org/abs/2207.08393.

### Setup

#### Environment
To avoid cuda-related issues, downloading `torch`, `torchvision`, and `cupy`
must be done prior to downloading other requirements.

```bash
# Create and activate the environment.
conda create -n gleam_env python=3.7
conda activate gleam_env

# Install cuda-dependant libraries. Change cuda version as needed.
# Below we show examples for cuda-10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install cupy-cuda101

# Install as package in virtual environment (recommended):
git clone https://github.com/batuozt/GLEAM
cd GLEAM && python -m pip install -e .

#### Registering New Machines/Clusters
To register new machines, you will have to find the regex pattern(s) that can be used to
identify the machine or set of machines you want to add functionality for. See
[ss_recon/utils/cluster.py](ss_recon/utils/cluster.py) for more details.

#### Registering New Users
To register users to existing machines, add your username and machines to support
with that username to the `_USER_PATHS` dictionary in
[ss_recon/utils/cluster.py](ss_recon/utils/cluster.py).

### Usage
To train a basic configuration from the repository folder in the command line, run
```bash
python tools/train_net.py --config-file configs/tests/basic.yaml

# Run in debug mode.
python tools/train_net.py --config-file configs/tests/basic.yaml --debug

# Run in reproducibility mode.
# This tries to make the run as reproducible as possible
# (e.g. setting seeds, deterministism, etc.).
python tools/train_net.py --config-file configs/tests/basic.yaml --reproducible
# or SSRECON_REPRO=True python tools/train_net.py --config-file configs/tests/basic.yaml

```

To evaluate the results, use `tools/eval_net.py`.
```bash
# Will automatically find best weights based on loss
python tools/eval_net.py --config-file configs/tests/basic.yaml

# Automatically find best weights based on psnr.
# options include psnr, l1, l2, ssim
python tools/eval_net.py --config-file configs/tests/basic.yaml --metric psnr

# Choose specific weights to run evaluation.
python tools/eval_net.py --config-file configs/tests/basic.yaml MODEL.WEIGHTS path/to/weights 
```

Example config files for different experiments from the paper are in [configs/GLEAM-configs](https://github.com/batuozt/GLEAM/tree/master/configs/GLEAM-configs)

## Weights and Biases
Our repository uses Weights and Biases (W&B) for experiment visualization. To use W&B with your entity and project name, you can modify the defaults for these at [ss_recon/config/defaults.py](https://github.com/batuozt/GLEAM/blob/master/ss_recon/config/defaults.py)

## Datasets

Files to format and use publicly available datasets mridata.org and fastMRI are available in the [datasets](https://github.com/batuozt/GLEAM/tree/master/datasets) folder.

## Acknowledgements
The code for GLEAM was developed based on [Meddlr](https://github.com/ad12/meddlr). Meddlr's and GLEAM's design are inspired by [detectron2](https://github.com/facebookresearch/detectron2).

Our implementation of decoupled greedy learning for unrolled neural networks was inspired by code for [https://arxiv.org/abs/1901.08164](https://arxiv.org/abs/1901.08164) at [https://github.com/eugenium/DGL)](https://github.com/eugenium/DGL).

For baseline comparisons with gradient checkpointing and memory-efficient learning, we used the open source implementation [https://github.com/mikgroup/MEL_MRI](https://github.com/mikgroup/MEL_MRI).

## About
If you use GLEAM for your work, please consider citing the following work:

```
@misc{ozturkler2022gleam,
      title={GLEAM: Greedy Learning for Large-Scale Accelerated MRI Reconstruction}, 
      author={Batu Ozturkler and Arda Sahiner and Tolga Ergen and Arjun D Desai and Christopher M Sandino and Shreyas Vasanawala and John M Pauly and Morteza Mardani and Mert Pilanci},
      year={2022},
      eprint={2207.08393},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Contact

For questions or comments, contact [ozt@stanford.edu](ozt@stanford.edu).

