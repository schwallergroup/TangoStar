<img align="center" src="DESP.png" width="350px" />

**Tango*: Constrained synthesis planning using chemically informed value functions**\
_Daniel Armstrong. Zlatko Jončev, Jeff Guo, Philippe Schwaller_

This repo contains code for Tango*, a retrosynthesis framework for starting material constrained synthesis planning. We base our code off the DESP codebase [https://github.com/coleygroup/desp], adding a new node reward class, tango_value. 

## Quick Start

To reproduce our experimental results or to try Tango* with DESP's pretrained models, perform the following steps after cloning this repository.

#### 1. Environment Setup

DESP requires a GPU to run at a practical speed. Ensure that the `pytorch-cuda` dependency is compatible with the version of CUDA on your machine. To check, run the following command and look for the `CUDA Version`. For those who want to use the Tango* algorithm only, reasonable speed can be achieved on a machine wihtout GPU.
```bash
$ nvidia-smi
```

Now, create the `desp` conda environment from the project directory:
```bash
$ conda env create -f environment.yml
```

#### 2. Data and model installation

Download the pre-trained model weights [at this link](https://figshare.com/articles/preprint/25956076). Unzip the contents of `desp_data.zip` into `/desp/data/`. 

<details>
  <summary>All components and their descriptions (useful if you want to train your own models / use your own data)</summary>

  1. `building_blocks.npz` - Contains 256-bit Morgan fingerprints with radius 2 of each molecule in the building block catalog (eMolecules).
  2. `canon_building_block_mol2idx_no_isotope.json` - Corresponds to a dictionary indexed by the SMILES strings of each molecule in the building block catalog.
  3. `idx2template_fwd.json` - Maps one-hot encoded indices of each forward template to the SMARTS string of the template for the forward template model.
  4. `idx2template_retro.json` - Maps one-hot encoded indices of each retro template to the SMARTS string of the template for the one-step retrosynthesis model.
  5. `model_bb.pt` - Checkpoint of the building block model. Input dim: 6144. Output dim: 256.
  6. `model_fwd.pt` - Checkpoint of the forward template model. Input dim: 4096. Output dim: 196339.
  7. `model_retro.pt` - Checkpoint of the one-step retro model. Input dim: 2048. Output dim: 270794.
  8. `retro_value.pt` - Checkpoint of the Retro* value model. Input dim: 2048. Output dim: 1.
  9. `syn_dist.pt` - Checkpoint of the synthetic distance model. Input dim: 4096. Output dim: 1.
  10. `pistachio_hard_targets.txt` - Line-delimited text file of pairs of targets and their starting material for benchmarking on Pistachio Hard. (i.e. `('CCOc1cc(-c2ccc(F)cc2-c2nncn2C)cc(-c2nc3cc(CN[C@@H]4CCC[C@@H]4O)cc(OC)c3o2)n1', 'CCOC(=O)c1cc(F)ccc1Br')`)
  11. `pistachio_reachable_targets.txt` - Like above, but for the Pistachio Reachable test set.
  12. `uspto_190_targets.txt` - Like above, but for the USPTO-190 test set.
</details>

#### 3. Run experiments

To reproduce the experiments, navigate to the directory `/desp/experiments/` and run the evaluation script. The first argument refers the benchmark set to use, while the second argument refers to the method to use. The results of the experiments will be saved in `/desp/experiments/<benchmark_set>/<method>.txt`, along with a corresponding `.pkl` file containing the full search graphs of each search.
```bash
$ sh evaluate.sh [pistachio_reachable|pistachio_hard|uspto_190] [f2e|f2f|retro|retro_sd|random|bfs]
```
A GPU is required for DESP-F2E or DESP-F2F. Specify the device in the evaluation script and ensure that your GPU has enough memory to load the building block index (around 3 GB). Additional memory is required for DESP-F2F due to batch inference of the synthetic distance predictions. The forward prediction module takes a few minutes to initialize as it loads the index into memory.


## Processing and Training from Scratch

See the guide at `/processing/README.md`.

## Acknowledgements

We thank the developers of [DESP]{https://github.com/coleygroup/desp, https://arxiv.org/abs/2407.06334} for releasing open source versions of their code for easy adaptation. If you find this code useful, please remember to also cite the DESP paper or repository.

## Citation

```bibtex
@inproceedings{
armstrong2024tango,
title={Tango*: Constrained synthesis planning using chemically informed value functions},
author={Daniel P Armstrong, Zlatko Jončev, Jeff Guo, Philippe Schwaller},
booktitle={ELLIS ML for Molecules and Materials in the Era of LLMs Workshop},
year={2024},
url={https://openreview.net/forum?id=cRT95W6AZa}
}
```
