# PowerFlow Graph Transformer

Topology-aware Graph Transformer prototype for power-flow learning under variable grid topology.

> **Current release scope.** This repository currently provides an IEEE 39-bus dynamic-topology power-flow prototype. The associated manuscript describes a broader power-flow foundation-model framework with variable-topology generalization and cross-task transfer capabilities.

## Overview

`powerflow-graph-transformer` implements a topology-aware Graph Transformer for power-flow calculation. The model represents an electrical network through bus-level state features and topology/admittance information, then combines local graph message passing with global self-attention to learn power-flow mappings under topology variations.

The codebase is designed around three ideas:

1. **Topology-aware node-edge encoding**  
   Bus states are encoded as node tokens, while network topology and admittance information are encoded through dense complex admittance matrices and edge-derived features. The model uses branch type/status embeddings, topology-derived attention bias, and graph masks to distinguish valid buses and active branches.

2. **Hybrid local-global Graph Transformer**  
   Each Transformer block combines dynamic local message passing with global graph attention. Local message passing propagates information along physical network connections, while global attention captures long-range electrical coupling.

3. **Physics-informed and bus-type-aware learning**  
   Training supports supervised and masked pretraining losses, power-flow residual regularization, and bus-type-aware target selection. For the power-flow task, the effective prediction targets are:

   | Bus type | Predicted physical targets |
   |---|---|
   | PQ bus | Voltage magnitude `Vm`, voltage angle `Va` |
   | PV bus | Reactive generation `Qg`, voltage angle `Va` |
   | Slack bus | Active generation `Pg`, reactive generation `Qg` |

## Repository layout

```text
powerflow-graph-transformer/
└── 39-bus-sys/
    ├── gen_39bus_pf_samples_gpt0421fix.py      # IEEE 39-bus dataset generation with N-k outages
    ├── compute_train_h_stats_modular.py        # Global feature statistics for normalization
    ├── train_powerflow_modular.py              # Main pretraining + finetuning entry point
    ├── evaluate_pf_model_physical_bus_type.py  # Physical and bus-type-aware evaluation
    ├── pf_data_loader.py                       # Dataset, collator, dataloader utilities
    ├── pf_powerflow_model.py                   # Power-flow prediction model wrapper
    ├── pf_topology_encoder.py                  # Hybrid node-edge Graph Transformer backbone
    ├── pf_topology_utils.py                    # Bus/branch metadata and masking utilities
    ├── pf_physics_losses.py                    # Masked MSE, bus-type loss, physics residual loss
    ├── pf_trainer.py                           # Training loop, checkpointing, AMP, metrics
    └── logs/20260423_170301_pf_modular_gpt0421/
        ├── ckpt_pretrain_best.pt
        ├── ckpt_finetune_best.pt
        ├── ckpt_finetune_last.pt
        ├── final_test_metrics.json
        └── train_history.json
```

## Installation

Clone the repository and enter the IEEE 39-bus prototype directory:

```bash
git clone https://github.com/zyh1996saa/powerflow-graph-transformer.git
cd powerflow-graph-transformer/39-bus-sys
```

Create an environment. The exact PyTorch command should match your CUDA or CPU setup.

```bash
conda create -n pfgt python=3.10 -y
conda activate pfgt

# Install PyTorch according to your hardware environment first.
# See: https://pytorch.org/get-started/locally/

pip install numpy scipy pandas tqdm pandapower tensorboard
```

The scripts use manual configuration blocks near the top of each main file. Before running them, edit paths such as `DATAPATH`, `DATASET_ROOT`, `DATA_DIR`, `OUTPUT_STATS_PATH`, `CHECKPOINT_PATH`, and `CHECKPOINT_SEARCH_ROOT` according to your local environment.

## Data format

Each sample is stored using three aligned files:

```text
metadata_<idx>.json
H_<idx>.npy
Y_<idx>.npz
```

where:

- `H_<idx>.npy` stores bus-level features.
- `Y_<idx>.npz` stores the complex bus admittance matrix.
- `metadata_<idx>.json` stores network metadata, valid-state masks, outage information, and other sample-level records.

The current six-dimensional bus feature convention is:

```text
[Pd, Qd, Pg, Qg, Vm, Va]
```

The dataloader supports a standard split layout:

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

## Generate IEEE 39-bus samples

The sample generation script is based on `pandapower.networks.case39()` and supports random load/generation perturbations and N-k outage scenarios.

Edit the configuration block in:

```text
gen_39bus_pf_samples_gpt0421fix.py
```

Important default settings include:

```python
DATAPATH = "/data2/zyh"
DATASET_ROOT = os.path.join(DATAPATH, "case39_samples")
TRAIN_NUM_SUCCESS_SAMPLES = 2048 * 32
TEST_NUM_SUCCESS_SAMPLES = 2048
MAX_OUTAGES = 3
LOAD_FACTOR_MIN = 0.5
LOAD_FACTOR_MAX = 1.5
```

Then run:

```bash
python gen_39bus_pf_samples_gpt0421fix.py
```

## Compute feature normalization statistics

Before training, compute global statistics for the bus feature matrix `H`:

```bash
python compute_train_h_stats_modular.py
```

By default, the script writes an `.npz` statistics file and a JSON summary. Make sure the generated statistics path matches the `STANDARDIZE_H_STATS_PATH` used by the training script.

## Train the model

The main training entry point is:

```bash
python train_powerflow_modular.py
```

The script performs:

1. masked self-supervised pretraining;
2. supervised finetuning with bus-type-aware power-flow targets;
3. optional final testing;
4. checkpoint and metric export.

Representative default configuration in the current code release:

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

Training artifacts are saved under the configured `LOG_DIR`, including:

```text
ckpt_pretrain_best.pt
ckpt_pretrain_last.pt
ckpt_finetune_best.pt
ckpt_finetune_last.pt
train_history.json
final_test_metrics.json
```

TensorBoard logs are written under the corresponding `tb/` subdirectory.

```bash
tensorboard --logdir logs
```

## Evaluate a trained checkpoint

Run:

```bash
python evaluate_pf_model_physical_bus_type.py
```

The evaluator can automatically discover candidate checkpoints under `CHECKPOINT_SEARCH_ROOT`, or use a manually specified `CHECKPOINT_PATH`.

It reports physically effective metrics instead of averaging over all raw feature columns. In particular, it evaluates only the target fields that are meaningful for each bus type:

- PQ bus: `Vm`, `Va`
- PV bus: `Qg`, `Va`
- Slack bus: `Pg`, `Qg`

The evaluator writes outputs such as:

```text
metrics_summary.csv
metrics_summary.json
selected_node_comparison.csv
selected_sample_summary.csv
run_info.json
target_features_by_bus_type.json
```

## Included checkpoint metrics

The repository currently includes one training run under:

```text
logs/20260423_170301_pf_modular_gpt0421/
```


These numbers are provided as a reproducibility reference for the current IEEE 39-bus prototype. They should not be interpreted as the full set of results reported in the manuscript.

## Relationship to the manuscript

The manuscript proposes a power-system power-flow foundation model with variable-topology generalization and cross-task transfer capabilities. At the methodological level, it introduces:

- a Graph Transformer architecture combining dynamic message passing and graph attention;
- node/branch tokenization for representing bus states, branch parameters, and grid topology;
- physics-informed self-supervised pretraining based on power-flow equation residuals;
- parameter-efficient downstream adaptation for tasks such as power-flow calculation, contingency screening, outage prediction, and optimal power flow.

This repository is the public prototype for the IEEE 39-bus power-flow calculation part of that framework. The broader multi-system, multi-task, LoRA transfer, and reinforcement-learning adaptation components described in the manuscript are not fully exposed in this code release.

## Main modules

### `pf_topology_encoder.py`

Implements the hybrid Graph Transformer backbone:

- node input encoding;
- edge/topology input encoding;
- dynamic local message passing;
- global graph attention with topology-derived attention bias;
- feed-forward Transformer blocks;
- valid-bus masking and dense-admittance-based topology construction.

### `pf_powerflow_model.py`

Wraps the backbone with a lightweight prediction head for bus-level power-flow targets.

### `pf_physics_losses.py`

Provides:

- masked MSE loss;
- bus-type-aware power-flow loss;
- complex-power computation from voltage and admittance;
- power-flow residual loss.

### `pf_trainer.py`

Provides the staged training loop, including:

- pretraining and finetuning stages;
- automatic mixed precision;
- AdamW optimizer and cosine learning-rate scheduler;
- checkpointing;
- metric logging;
- final evaluation.

## Practical notes

- The current scripts contain local absolute paths. Edit them before running.
- The training route currently expects dense `Y` matrices in the collated batch.
- Reproducibility depends on the generated dataset, topology perturbation settings, random seeds, and hardware/software environment.
- The current repository does not include a `requirements.txt` or `environment.yml`. Adding one is recommended for archival release.
- No open-source license file is currently included. Add a `LICENSE` file before encouraging external reuse.

## Citation

If you use this repository, please cite the associated manuscript. Replace the placeholder metadata after formal publication.


## Acknowledgements

This research code uses PyTorch and pandapower for learning-based modeling and power-flow data generation.
