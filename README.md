# PowerFlow Graph Transformer

Topology-aware Graph Transformer prototypes for learning power-flow mappings under variable grid topology.

This repository provides research code for learning bus-level power-flow quantities from graph-structured power-system data. The implementation targets the IEEE 39-bus test system and contains two related experimental pipelines:

1. an AC power-flow learning prototype in `39-bus-sys/`; and
2. a DC-baseline residual-learning variant in `39-bus-sys-dc-delta/`, where the model learns the correction from a DC power-flow baseline to the AC power-flow solution.

The code is intended as a transparent and reproducible prototype for topology-aware neural power-flow modeling. It is not a production-grade power-system solver.

---

## Table of Contents

- [Overview](#overview)
- [Repository Architecture](#repository-architecture)
- [Methodological Summary](#methodological-summary)
- [Data Representation](#data-representation)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs and Checkpoints](#outputs-and-checkpoints)
- [Evaluation](#evaluation)
- [Reproducibility Notes](#reproducibility-notes)
- [Relationship to the Manuscript](#relationship-to-the-manuscript)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

`powerflow-graph-transformer` implements topology-aware neural models for power-flow prediction. Each electrical network instance is represented by bus-level state features, a complex bus-admittance matrix, and metadata describing bus types, branches, generator locations, loads, and topology perturbations. The model combines local graph message passing with global Transformer-style attention, allowing the representation to incorporate both physical network connectivity and long-range electrical coupling.

The public release currently focuses on IEEE 39-bus experiments with dynamic topology variations. The data-generation scripts create perturbed operating points by applying load variations, generation adjustments, and line-outage scenarios. The training scripts then learn bus-level quantities under bus-type-specific target definitions.

The repository contains two complementary workflows:

- **AC target prediction (`39-bus-sys/`)**: the model directly predicts the physically relevant AC power-flow targets for PQ, PV, and slack buses.
- **DC-delta prediction (`39-bus-sys-dc-delta/`)**: the model receives a DC power-flow baseline and learns the residual correction to the AC solution. This variant is useful for studying whether a topology-aware neural model can learn the nonlinear error structure left by a DC approximation.

---

## Repository Architecture

```text
powerflow-graph-transformer/
├── README.md
├── 39-bus-sys/
│   ├── gen_39bus_pf_samples_gpt0421fix.py
│   ├── compute_train_h_stats_modular.py
│   ├── train_powerflow_modular.py
│   ├── evaluate_pf_model_physical_bus_type.py
│   ├── pf_data_loader.py
│   ├── pf_powerflow_model.py
│   ├── pf_topology_encoder.py
│   ├── pf_topology_utils.py
│   ├── pf_physics_losses.py
│   ├── pf_trainer.py
│   ├── audit_pf_dataset_gpt0421.py
│   ├── audit_pf_encoding_strategies_gpt0423.py
│   ├── audit_pf_training_flow_gpt0421.py
│   ├── replay_pf_samples_pandapower_gpt0421.py
│   └── logs/
└── 39-bus-sys-dc-delta/
    ├── gen_39bus_pf_samples_dc_delta.py
    ├── compute_train_hdc_delta_stats.py
    ├── train_powerflow_dc_delta.py
    ├── evaluate_pf_model_dc_delta_full_physics.py
    ├── check_dc_ac_error_print.py
    ├── pf_data_loader.py
    ├── pf_powerflow_model.py
    ├── pf_topology_encoder.py
    ├── pf_topology_utils.py
    ├── pf_physics_losses.py
    ├── pf_trainer.py
    ├── logs/
    └── pf_eval_outputs_dc_delta_full_physics/
```

### Main modules

| Module | Role |
| --- | --- |
| `gen_39bus_pf_samples_*.py` | Generates IEEE 39-bus samples under load, generation, and topology perturbations. |
| `compute_train_h_stats_modular.py` | Computes normalization statistics for the AC-target workflow. |
| `compute_train_hdc_delta_stats.py` | Computes normalization statistics for the DC-baseline and AC-minus-DC residual workflow. |
| `train_powerflow_modular.py` | Trains the direct AC power-flow model. |
| `train_powerflow_dc_delta.py` | Trains the DC-delta residual model. |
| `evaluate_pf_model_physical_bus_type.py` | Evaluates the direct AC model using bus-type-aware physical targets. |
| `evaluate_pf_model_dc_delta_full_physics.py` | Evaluates the DC-delta model using reconstructed physical quantities and power-flow residuals. |
| `check_dc_ac_error_print.py` | Reports diagnostic DC-vs-AC discrepancies before neural correction. |
| `pf_data_loader.py` | Implements sample indexing, data loading, batching, masks, and optional dense admittance conversion. |
| `pf_topology_encoder.py` | Implements the hybrid node-edge Graph Transformer backbone. |
| `pf_powerflow_model.py` | Wraps the topology encoder with a bus-level prediction head. |
| `pf_physics_losses.py` | Provides supervised, bus-type-aware, and physics-residual loss components. |
| `pf_trainer.py` | Implements optimization, checkpointing, logging, metric aggregation, and final testing. |
| `pf_topology_utils.py` | Provides metadata parsing, bus-type construction, standardization utilities, and topology-related helpers. |

---

## Methodological Summary

### Graph-structured power-system representation

Each sample is represented as a graph whose nodes correspond to buses. Node features are six-dimensional bus-level physical quantities:

```text
[Pd, Qd, Pg, Qg, Vm, Va]
```

where `Pd` and `Qd` denote active and reactive demand, `Pg` and `Qg` denote active and reactive generation, and `Vm` and `Va` denote voltage magnitude and voltage angle. The network topology and electrical coupling are represented through the complex bus-admittance matrix `Y` and branch metadata.

### Hybrid local-global Graph Transformer

The model uses a hybrid Graph Transformer architecture. Local message passing propagates information over physically connected bus pairs, while global attention allows each bus representation to attend to nonlocal electrical dependencies. Edge encodings include admittance-derived features, branch status information, branch type embeddings, and topology-derived attention bias. Invalid or isolated buses are masked during training and evaluation.

### Bus-type-aware prediction

The physically meaningful prediction targets depend on the bus type:

| Bus type | Effective target fields |
| --- | --- |
| PQ bus | `Vm`, `Va` |
| PV bus | `Qg`, `Va` |
| Slack bus | `Pg`, `Qg` |

This target definition avoids evaluating quantities that are specified as inputs rather than solved outputs for a given bus type.

### Physics-informed regularization

The training pipeline supports physics-informed regularization based on power-flow residuals. During training, the model prediction can be reconstructed into a physical bus-state matrix and evaluated against the admittance matrix through complex-power consistency terms. This regularization is used together with supervised bus-type-aware losses.

### DC-delta residual learning

The `39-bus-sys-dc-delta/` workflow stores both an AC solution and a DC baseline for each sample. The learning target is the residual

```text
Delta H = H_ac - H_dc
```

on bus-type-relevant target columns. The trained model predicts this residual, which is then added back to the DC baseline to reconstruct the corrected physical state. This formulation separates a low-cost linearized approximation from the learned nonlinear correction.

---

## Data Representation

### AC-target workflow

The direct AC workflow uses aligned files of the form:

```text
metadata_<idx>.json
H_<idx>.npy
Y_<idx>.npz
```

where:

- `H_<idx>.npy` stores the bus-level feature matrix;
- `Y_<idx>.npz` stores the complex bus-admittance matrix, typically in sparse format before batching;
- `metadata_<idx>.json` stores sample-level metadata, topology perturbations, masks, and auxiliary information.

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

### DC-delta workflow

The DC-delta workflow stores the AC solution, the DC baseline, and the admittance matrix:

```text
metadata_<idx>.json
H_<idx>.npy
H_dc_<idx>.npy
Y_<idx>.npz
```

where:

- `H_<idx>.npy` stores the converged AC power-flow solution;
- `H_dc_<idx>.npy` stores the DC baseline represented in the same six-column convention;
- `Y_<idx>.npz` stores the AC admittance matrix associated with the perturbed topology;
- `metadata_<idx>.json` stores bus validity masks, outage information, DC-baseline policy, and sample diagnostics.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/zyh1996saa/powerflow-graph-transformer.git
cd powerflow-graph-transformer
```

Create a Python environment:

```bash
conda create -n pfgt python=3.10 -y
conda activate pfgt
```

Install PyTorch according to the target CPU or CUDA environment. For example, consult the official PyTorch installation selector before choosing the exact command.

Install the remaining dependencies:

```bash
pip install numpy scipy pandas tqdm pandapower tensorboard
```

Optional but recommended development tools:

```bash
pip install matplotlib pytest black ruff
```

At the time of this README draft, the repository does not define a pinned `requirements.txt` or `environment.yml`. For archival reproducibility, it is recommended to add one after fixing the intended Python, PyTorch, CUDA, NumPy, SciPy, pandas, pandapower, and TensorBoard versions.

---

## Usage

The scripts are configured through explicit constants near the beginning of each main file. Before running an experiment, edit dataset paths, output paths, hardware settings, and checkpoint paths directly inside the corresponding script.

### Workflow A: direct AC power-flow prediction

Enter the AC-target prototype directory:

```bash
cd 39-bus-sys
```

Generate IEEE 39-bus samples:

```bash
python gen_39bus_pf_samples_gpt0421fix.py
```

Compute feature-normalization statistics:

```bash
python compute_train_h_stats_modular.py
```

Train the model:

```bash
python train_powerflow_modular.py
```

Evaluate a trained checkpoint:

```bash
python evaluate_pf_model_physical_bus_type.py
```

Useful diagnostic scripts include:

```bash
python audit_pf_dataset_gpt0421.py
python audit_pf_encoding_strategies_gpt0423.py
python audit_pf_training_flow_gpt0421.py
python replay_pf_samples_pandapower_gpt0421.py
```

### Workflow B: DC-baseline residual prediction

Enter the DC-delta prototype directory:

```bash
cd 39-bus-sys-dc-delta
```

Generate samples containing both AC solutions and DC baselines:

```bash
python gen_39bus_pf_samples_dc_delta.py
```

Compute normalization statistics for `H_dc` and `H_ac - H_dc`:

```bash
python compute_train_hdc_delta_stats.py
```

Train the residual model:

```bash
python train_powerflow_dc_delta.py
```

Evaluate a trained checkpoint using reconstructed physical quantities:

```bash
python evaluate_pf_model_dc_delta_full_physics.py
```

Inspect the baseline DC-vs-AC discrepancy:

```bash
python check_dc_ac_error_print.py
```

---

## Configuration

The current implementation uses manual configuration blocks. Important configuration fields include:

| Field | Meaning |
| --- | --- |
| `DATA_DIR` or `DATASET_ROOT` | Root directory containing generated samples. |
| `STANDARDIZATION_STATS_PATH` | Path to normalization statistics used by the training script. |
| `BATCH_SIZE` | Number of samples per training batch. |
| `TRAIN_SPLIT`, `VAL_SPLIT` | Data split fractions for training and validation. |
| `DEVICE` | Training device, usually `cuda` or `cpu`. |
| `Y_AS_DENSE` | Whether to collate admittance matrices as dense tensors. |
| `d_model` | Hidden dimension of the Graph Transformer. |
| `num_layers` | Number of hybrid Graph Transformer layers. |
| `num_heads` | Number of global-attention heads. |
| `phy_loss_weight` | Weight of the physics-residual loss term. |
| `dynamic_depth_sampling` | Whether to randomly vary the number of active Transformer layers during training. |
| `OUTPUT_ROOT` or `LOG_DIR` | Directory for checkpoints, TensorBoard logs, and metric files. |

The default DC-delta training configuration uses a 6-dimensional node input, 4-dimensional edge input, 6-dimensional output, hidden dimension 192, four Transformer layers, eight attention heads, and 220 training epochs. These values should be treated as experiment defaults rather than universal hyperparameters.

---

## Outputs and Checkpoints

Training scripts write experiment artifacts under the configured log directory, for example:

```text
logs/<timestamp>_pf_modular_gpt0421/
logs/<timestamp>_pf_dc_delta/
```

Common outputs include:

```text
ckpt_pretrain_best.pt
ckpt_pretrain_last.pt
ckpt_finetune_best.pt
ckpt_finetune_last.pt
ckpt_best.pt
ckpt_last.pt
train_history.json
final_test_metrics.json
tb/
```

The exact checkpoint names differ between the AC-target workflow and the DC-delta workflow. Evaluation scripts may either use a manually specified checkpoint path or search a configured checkpoint directory.

Evaluation outputs may include:

```text
metrics_summary.csv
metrics_summary.json
selected_node_comparison.csv
selected_sample_summary.csv
run_info.json
target_features_by_bus_type.json
```

---

## Evaluation

The evaluation protocol is bus-type-aware. Metrics are computed only on physically meaningful target fields for each bus type. This design avoids penalizing the model for not predicting quantities that are specified by the power-flow formulation rather than solved as outputs.

For the DC-delta workflow, evaluation reconstructs physical predictions as:

```text
H_pred = H_dc + Delta H_pred
```

and then computes target-space errors and physics-residual diagnostics. The script `check_dc_ac_error_print.py` can be used to quantify the uncorrected DC-baseline error before applying the learned residual model.

---

## Reproducibility Notes

Reproducibility depends on several factors:

- random seeds used for sample generation and training;
- load-perturbation ranges and generation-scaling policy;
- outage-generation policy, including the maximum number of line outages;
- pandapower version and solver behavior;
- PyTorch, CUDA, and hardware configuration;
- normalization statistics computed from the training split;
- checkpoint-selection policy during evaluation.

For a fully reproducible release, it is recommended to archive:

1. the generated dataset or the exact data-generation seed schedule;
2. the normalization-statistics files;
3. the full training configuration;
4. package versions and hardware information;
5. the selected checkpoint and evaluation configuration.

---

## Relationship to the Manuscript

The broader research direction concerns topology-aware and physics-informed neural power-flow modeling under variable operating conditions and topology perturbations. This repository exposes IEEE 39-bus prototype pipelines that instantiate core components of that direction, including topology encoding, local-global graph representation learning, bus-type-aware supervision, physics-informed loss terms, and DC-baseline residual correction.

If a manuscript associated with this repository describes additional systems, tasks, transfer-learning mechanisms, LoRA adaptation, reinforcement-learning adaptation, or large-scale foundation-model experiments, those components should be regarded as outside the current public code scope unless they are explicitly added to the repository.

---

## Limitations

This repository should be interpreted as research code. The following limitations are relevant for users and reviewers:

- The current public implementation focuses on IEEE 39-bus experiments.
- Some paths are local absolute paths and must be edited before execution.
- The repository does not yet provide a pinned environment file.
- The repository does not yet provide a license file.
- Large generated datasets are not necessarily included in the repository.
- Reported checkpoint metrics depend on the generated dataset, random seed, and evaluation configuration.
- The code is not validated as a replacement for certified power-system analysis tools.

---



---

## License

No open-source license file is currently included in this draft. Before distributing the project for external reuse, add a `LICENSE` file and state the intended terms of use in this section.

---

## Acknowledgements

This project uses PyTorch for neural-network modeling and pandapower for power-system simulation and data generation. The IEEE 39-bus case is used as the main public prototype system in the current release.
