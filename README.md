# Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes

Recently, cutting-plane methods such as [GCP-CROWN](https://arxiv.org/pdf/2208.05740.pdf) have been explored to enhance neural network verifiers and made significant advancements. However, GCP-CROWN currently relies on generic cutting planes (“cuts”) generated from external mixed integer programming (MIP) solvers. Due to the poor scalability of MIP solvers, large neural networks cannot benefit from these cutting planes.

In this work we exploit the structure of the neural network verification problem to generate efficient and scalable cutting planes specific to this problem setting. We propose a novel approach, Branch-and-bound Inferred Cuts with COnstraint
Strengthening __(BICCOS)__, that leverages the logical relationships of neurons within verified subproblems in the branch-and-bound search tree, and we introduce cuts that preclude these relationships in other subproblems. We develop a mechanism that assigns influence scores to neurons in each path to allow the strengthening of these cuts. Furthermore, we design a multi-tree search technique to identify more cuts, effectively narrowing the search space and accelerating the BaB algorithm.

Our results demonstrate that BICCOS can generate hundreds of useful cuts during the branch-and-bound process and consistently increase the number of verifiable instances compared to other state-of-the-art neural network verifiers on a wide range of benchmarks, including large networks that previous cutting plane methods could not scale to. BICCOS is part of the [α,β-CROWN](http://abcrown.org), the VNN-COMP 2024 winner.

More details can be found in our paper:

[Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes](https://openreview.net/pdf?id=FwhM1Zpyft)
**NeurIPS 2024**
Duo Zhou, Christopher Brix, Grani A Hanasusanto, Huan Zhang

## Reproducing results

BICCOS has been incorporated into our [α,β-CROWN](http://abcrown.org) verifier (alpha-beta-CROWN), which is the **winning tool of VNN-COMP 2021, 2022, 2023, and 2024 with the highest total score**. To reproduce the results in the paper, please use the codebase in [α,β-CROWN repo](http://abcrown.org) following the detailed instructions below.

<p align="center">
<a href="https://abcrown.org"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="28%"></a>
</p>

### Installation and setup

Our code is tested on Python 3.11+ and PyTorch 2.3.1. It can be installed easily into a conda environment. If you don't have conda, you can install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Clone the alpha-beta-CROWN verifier
git clone https://github.com/huanzhang12/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown  # install all dependents into the alpha-beta-crown environment
conda activate alpha-beta-crown  # activate the environment
```

We currently require the IBM CPLEX solver to use the MIP cuts with BICCOS for verification if needed. It is free for students and academics from [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students).
Please run the following commands to install CPLEX solver and compile its C++ interfacing code:

```bash
# Install IBM CPLEX >= 22.1.0
# Download from https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students
chmod +x cplex_studio2210.linux_x86_64.bin  # Any version >= 22.1.0 should work. Change executable name here.
# You can directly run the installer: ./cplex_studio2210.linux_x86_64.bin; the response.txt created below is for non-interactive installation.
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF
sudo ./cplex_studio2210.linux_x86_64.bin -f response.txt
# Build the C++ code for CPLEX interface. Assumming we are still inside the alpha-beta-CROWN folder.
sudo apt install build-essential  # A modern g++ (>=8.0) is required to compile the code.
# Change CPX_PATH in complete_verifier/CPLEX_cuts/Makefile if you installed CPlex to a non-default location, like inside your home folder.
make -C complete_verifier/CPLEX_cuts/
```

### Reproducing BICCOS results

Results and log files are stored in the repo, please check `BICCOS_results.ipynb` to get the detailed results. Our experiments are conducted on a server with an Intel Xeon 8468 Sapphire CPU, one NVIDIA H100 GPU (96 GB GPU memory), and 480 GB CPU memory.

All the configs to reproduce the running results are collected in `complete_verifier/exp_configs/BICCOS/` folder, using the December 2024 release of the α,β-CROWN (alpha-beta-CROWN) verifier. Below we show the detailed commands for running each experiments.

To reproduce the results for VNN-COMP24 benchmarks `cifar100` and `tinyimagnet`, please run

```python
python abcrown.py --config exp_configs/BICCOS/cifar100.yaml
python abcrown.py --config exp_configs/BICCOS/tinyimagenet.yaml
```

To reproduce the results for VNN-COMP22 benchmarks `oval22` and `cifar100-tinyimagnet-2022`, please run

```python
python abcrown.py --config exp_configs/BICCOS/oval22.yaml
python abcrown.py --config exp_configs/BICCOS/cifar100_small_2022.yaml
python abcrown.py --config exp_configs/BICCOS/cifar100_med_2022.yaml
python abcrown.py --config exp_configs/BICCOS/cifar100_large_2022.yaml
python abcrown.py --config exp_configs/BICCOS/cifar100_super_2022.yaml
python abcrown.py --config exp_configs/BICCOS/tinyimagenet.yaml
```

To reproduce the results for SDP models, please run

```bash
python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_mix.yaml
python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_mix4.yaml
python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_adv.yaml
python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_adv4.yaml
python abcrown.py --config exp_configs/BICCOS/cifar_cnn_b_adv.yaml
python abcrown.py --config exp_configs/BICCOS/cifar_cnn_b_adv4.yaml
python abcrown.py --config exp_configs/BICCOS/mnist_cnn_a_adv.yaml
```

### BibTex Entry

```
@inproceedings{zhou2024scalable,
  title={Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes},
  author={Zhou, Duo and Brix, Christopher and Hanasusanto, Grani A and Zhang, Huan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
