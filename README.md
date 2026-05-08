# PowerFlow Graph Transformer

Topology-aware Graph Transformer for learning AC power-flow mappings under variable grid topology.

This repository provides the public research-code release for the IEEE 39-bus dynamic-topology power-flow prototype associated with the manuscript:

> **A Power System Power Flow Foundation Model for Enhanced Variable-Topology and Cross-Task Generalization Capabilities**

The broader manuscript develops a power-flow foundation-model framework for variable-topology generalization and cross-task transfer in power-system analysis. This code release focuses on a reproducible IEEE 39-bus prototype that implements the core modeling pipeline: dynamic-topology data generation, topology-aware Graph Transformer encoding, physics-informed pretraining, supervised fine-tuning, and physically meaningful evaluation.

---

## Why this repository matters

Power grids are not static graphs. Outages, switching actions, maintenance schedules, and operating-mode changes continuously alter the network topology. Conventional task-specific neural models often fit a limited topology distribution and degrade when the grid structure changes.

`powerflow-graph-transformer` is designed around a more transferable modeling principle: represent the electrical network as graph-structured tokens, combine local physical connectivity with global attention, and constrain learning with power-flow physics. The result is a compact and reproducible prototype for studying how Graph Transformer architectures can support power-flow learning under unseen topologies.

---

## Key features

### 1. Variable-topology power-flow dataset generation

The IEEE 39-bus dataset generator is built on `pandapower.networks.case39()` and supports:

- random load and generation perturbations;
- N-k line-outage scenarios;
- train/test split generation;
- metadata export for bus, branch, generator, load, outage, and valid-state information;
- aligned storage of bus-state matrices, admittance matrices, and sample metadata.

The default generation script creates successful AC power-flow samples and stores each sample as:

```text
metadata_<idx>.json
H_<idx>.npy
Y_<idx>.npz
```

### 2. Topology-aware Graph Transformer backbone

The model combines two complementary mechanisms:

- **Dynamic local message passing**, which propagates information along the physical grid topology.
- **Global graph attention**, which captures long-range electrical coupling that may be difficult to represent with purely local GNN layers.

This hybrid design is intended to preserve the inductive bias of power networks while improving the model's ability to represent nonlocal dependencies.

### 3. Node-edge and admittance-aware encoding

The prototype represents the network through:

- bus-level state features;
- complex bus admittance matrices;
- topology-derived edge features;
- branch status/type information;
- valid-bus masks and graph masks.

This design makes topology changes explicit in the model input rather than treating them as incidental distributional noise.

### 4. Physics-informed and bus-type-aware learning

Training supports:

- masked self-supervised pretraining;
- supervised fine-tuning for power-flow targets;
- power-flow residual regularization;
- bus-type-aware target selection.

For the power-flow task, the effective prediction targets follow the physical role of each bus type:

| Bus type | Effective prediction targets |
| --- | --- |
| PQ bus | Voltage magnitude `Vm`, voltage angle `Va` |
| PV bus | Reactive generation `Qg`, voltage angle `Va` |
| Slack bus | Active generation `Pg`, reactive generation `Qg` |

### 5. Reproducible staged training and evaluation

The training pipeline supports:

- pretraining;
- fine-tuning;
- automatic mixed precision;
- checkpointing;
- TensorBoard logging;
- final metric export;
- physical residual evaluation.

The evaluator reports physically meaningful metrics instead of averaging over irrelevant raw feature columns.

---

## Relationship to the manuscript

The associated manuscript proposes a power-system power-flow foundation model with three central ideas:

1. **Graph Transformer architecture** combining dynamic message passing and graph attention.
2. **Physics-informed self-supervised pretraining** that unifies power-flow-related tasks as masked regression problems and regularizes predictions with power-flow equation residuals.
3. **Parameter-efficient downstream adaptation** for tasks such as power-flow calculation, contingency screening, outage prediction, and optimal power flow.

This repository is the public IEEE 39-bus prototype of that framework. It exposes the core engineering route for topology-varying power-flow learning and provides a concrete basis for reproducing, auditing, and extending the method. The broader multi-system and multi-task results reported in the manuscript are described in the paper and can be integrated into this codebase in future releases.

---

## Repository layout

```text
powerflow-graph-transformer/
└── 39-bus-sys/
    ├── gen_39bus_pf_samples_gpt0421fix.py
    ├── compute_train_h_stats_modular.py
    ├── train_powerflow_modular.py
    ├── evaluate_pf_model_physical_bus_type.py
    ├── pf_data_loader.py
    ├── pf_powerflow_model.py
    ├── pf_topology_encoder.py
    ├── pf_topology_utils.py
    ├── pf_physics_losses.py
    ├── pf_trainer.py
    └── logs/20260423_170301_pf_modular_gpt0421/
        ├── ckpt_pretrain_best.pt
        ├── ckpt_finetune_best.pt
        ├── ckpt_finetune_last.pt
        ├── final_test_metrics.json
        └── train_history.json
```

Main modules:

| File | Role |
| --- | --- |
| `gen_39bus_pf_samples_gpt0421fix.py` | Generates IEEE 39-bus dynamic-topology power-flow samples. |
| `compute_train_h_stats_modular.py` | Computes global feature statistics for normalization. |
| `train_powerflow_modular.py` | Main pretraining and fine-tuning entry point. |
| `evaluate_pf_model_physical_bus_type.py` | Evaluates checkpoints with bus-type-aware and physics-aware metrics. |
| `pf_data_loader.py` | Dataset, collator, and dataloader utilities. |
| `pf_powerflow_model.py` | Power-flow model wrapper. |
| `pf_topology_encoder.py` | Hybrid topology-aware Graph Transformer backbone. |
| `pf_topology_utils.py` | Bus/branch metadata, masks, and topology utilities. |
| `pf_physics_losses.py` | Masked MSE, bus-type-aware loss, and power-flow residual loss. |
| `pf_trainer.py` | Training loop, checkpointing, AMP, logging, and evaluation. |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/zyh1996saa/powerflow-graph-transformer.git
cd powerflow-graph-transformer/39-bus-sys
```

Create a Python environment:

```bash
conda create -n pfgt python=3.10 -y
conda activate pfgt
```

Install dependencies:

```bash
# Install PyTorch first according to your CUDA or CPU environment:
# https://pytorch.org/get-started/locally/

pip install numpy scipy pandas tqdm pandapower tensorboard
```

The current release does not include a `requirements.txt` or `environment.yml`. Adding one is recommended when preparing an archival or camera-ready release.

---

## Configuration style

The scripts use explicit configuration blocks near the top of each main file instead of command-line argument parsing. Before running the pipeline, edit the relevant paths and settings directly in the script, for example:

```python
DATA_DIR = "/data2/zyh/case39_samples"
DEVICE = "cuda"
BATCH_SIZE = 256
```

Common path variables to check include:

- `DATAPATH`
- `DATASET_ROOT`
- `DATA_DIR`
- `STANDARDIZATION_STATS_PATH`
- `CHECKPOINT_PATH`
- `CHECKPOINT_SEARCH_ROOT`
- `OUTPUT_ROOT`

---

## Data format

Each sample contains three aligned files:

```text
metadata_<idx>.json
H_<idx>.npy
Y_<idx>.npz
```

where:

- `H_<idx>.npy` stores bus-level features;
- `Y_<idx>.npz` stores the complex bus admittance matrix;
- `metadata_<idx>.json` stores network metadata, outage records, valid-state masks, and sample-level information.

The current six-dimensional bus feature convention is:

```text
[Pd, Qd, Pg, Qg, Vm, Va]
```

A typical dataset layout is:

```text
case39_samples/
├── train/
│   ├── metadata_0.json
│   ├── H_0.npy
│   └── Y_0.npz
└── test/
    ├── metadata_0.json
    ├── H_0.npy
    └── Y_0.npz
```

---

## Generate IEEE 39-bus samples

Edit the configuration block in:

```text
gen_39bus_pf_samples_gpt0421fix.py
```

Representative default settings include:

```python
DATAPATH = "/data2/zyh"
DATASET_ROOT = os.path.join(DATAPATH, "case39_samples")
TRAIN_NUM_SUCCESS_SAMPLES = 2048 * 32
TEST_NUM_SUCCESS_SAMPLES = 2048
MAX_OUTAGES = 3
MIN_FACTOR = 0.5
MAX_FACTOR = 1.5
```

Run:

```bash
python gen_39bus_pf_samples_gpt0421fix.py
```

The script will generate train/test folders containing converged AC power-flow samples.

---

## Compute normalization statistics

Before training, compute global statistics for the bus feature matrix:

```bash
python compute_train_h_stats_modular.py
```

Make sure the generated statistics path matches the `STANDARDIZATION_STATS_PATH` used by the training script.

---

## Train the model

Run:

```bash
python train_powerflow_modular.py
```

The default training route performs:

1. masked self-supervised pretraining;
2. supervised fine-tuning with bus-type-aware power-flow targets;
3. optional final testing;
4. checkpoint, metric, and training-history export.

Representative default model and training settings include:

```python
BATCH_SIZE = 256
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

node_feat_dim = 6
edge_feat_dim = 4
output_dim = 6
d_model = 192
num_layers = 4
num_heads = 8
dropout = 0.10

dynamic_depth_sampling = True
pretrain_phy_loss_weight = 0.02
finetune_phy_loss_weight = 0.10
mask_rate_feature = 0.08

num_pretrain_epochs = 120
num_finetune_epochs = 220
```

Training artifacts are saved under the configured `OUTPUT_ROOT`, including:

```text
ckpt_pretrain_best.pt
ckpt_pretrain_last.pt
ckpt_finetune_best.pt
ckpt_finetune_last.pt
train_history.json
final_test_metrics.json
```

TensorBoard logs are written under the corresponding `tb/` subdirectory:

```bash
tensorboard --logdir logs
```

---

## Evaluate a trained checkpoint

Run:

```bash
python evaluate_pf_model_physical_bus_type.py
```

The evaluator can automatically discover candidate checkpoints under `CHECKPOINT_SEARCH_ROOT` or use a manually specified `CHECKPOINT_PATH`.

It reports target-aware metrics for the physically meaningful prediction fields:

- PQ bus: `Vm`, `Va`
- PV bus: `Qg`, `Va`
- Slack bus: `Pg`, `Qg`

Typical output files include:

```text
metrics_summary.csv
metrics_summary.json
selected_node_comparison.csv
selected_sample_summary.csv
run_info.json
target_features_by_bus_type.json
```

---

## Included reference checkpoint

The repository includes a reference training run under:

```text
39-bus-sys/logs/20260423_170301_pf_modular_gpt0421/
```

The included `final_test_metrics.json` provides a reproducibility reference for the current IEEE 39-bus prototype. These values are intended to document the released code path and should not be interpreted as the complete multi-system results reported in the manuscript.

Selected metrics from the included run:

| Metric | Value |
| --- | ---: |
| Supervised MSE | 0.039425 |
| Overall RMSE | 0.197638 |
| Physics MSE | 0.016908 |
| Physics P RMSE (p.u.) | 0.151849 |
| Physics Q RMSE (p.u.) | 0.102657 |
| PQ RMSE | 0.310851 |
| PV RMSE | 0.105591 |
| Slack RMSE | 0.094352 |

---

## Practical notes

- Edit local absolute paths before running the scripts.
- The current training route expects dense `Y` matrices in the collated batch.
- Reproducibility depends on random seeds, generated topology perturbations, hardware, and software versions.
- For a formal open-source release, consider adding:
  - `requirements.txt` or `environment.yml`;
  - `LICENSE`;
  - a stable DOI or release tag;
  - final citation metadata after publication.

---

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{zhu_powerflow_foundation_model,
  title   = {A Power System Power Flow Foundation Model for Enhanced Variable-Topology and Cross-Task Generalization Capabilities},
  author  = {Zhu, Yuhong and Li, Peng and Xia, Liangming and Yan, Lei and Zhou, Yongzhi},
  journal = {Proceedings of the CSEE},
  year    = {2026},
  note    = {Manuscript metadata to be updated after formal publication}
}
```

Please replace the placeholder BibTeX entry with the official citation once the paper is published.

---

## Acknowledgements

This research code uses PyTorch and pandapower for learning-based power-system modeling and AC power-flow data generation.

The project is supported by research on foundation models for power-system analysis and operation. The released prototype is intended to make the core methodology easier to reproduce, inspect, and extend.
