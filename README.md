[![Dataset DOI](https://img.shields.io/badge/DOI-10.57760%2Fsciencedb.01be0-blue)](https://doi.org/10.57760/sciencedb.01be0)
# ChemOdorCL

ChemOdorCL adapts MoLFormer molecular representations for multi-label odor prediction. The workflow consists of constructing positive-sample pools, continued contrastive pretraining, and downstream odor-prediction fine-tuning.

## Workflow

1. Install the environment following [environment.md](./environment.md).
2. Download the pretrained MoLFormer checkpoint and place it in `data/checkpoints/`.
3. Prepare separate training, validation, and test CSV files.
4. Build positive-sample pools using [feature_extraction](./feature_extraction/).
5. Run contrastive pretraining and downstream fine-tuning using [finetune](./finetune/).

The commands below use Bash and a CUDA GPU. Start in the project root unless a different working directory is specified.

## 1. Install the environment

Follow [environment.md](./environment.md) to create the Conda environment, install the pinned Python packages, and compile NVIDIA Apex with the required CUDA extensions.

The documented environment uses Python 3.8.10, PyTorch 1.7.1, CUDA 11.0, and PyTorch Lightning 1.1.5. It also includes RDKit, pandas, NumPy, scikit-learn, Transformers, and pytorch-fast-transformers. Both training scripts import Apex, so complete that installation step before running them.

After installation, activate the environment:

```bash
conda activate MolTran_CUDA11
```

## 2. Download the pretrained MoLFormer checkpoint

Follow the pretrained-model instructions in the [official MoLFormer repository](https://github.com/IBM/molformer#pretrained-models-and-training-logs). The upstream checkpoint archive is available from the [MoLFormer data and model download page](https://ibm.box.com/v/MoLFormer-data).

Create the checkpoint directory from the project root:

```bash
mkdir -p data/checkpoints
```

Extract the archive and copy the required `.ckpt` file into this directory. The examples use:

```text
data/checkpoints/N-Step-Checkpoint_3_30000.ckpt
```

The upstream archive contains a `Pretrained MoLFormer/checkpoints/` directory. For this project, copy the checkpoint file from that directory into `data/checkpoints/`. If you choose another checkpoint filename, update `--seed_path` in the pretraining command accordingly.

## 3. Prepare the odor dataset

For the example dataset key `mop_cleanxclr`, place these files under `data/mop_cleanxclr/`:

| Split      | Filename                    | Purpose                                           |
| ---------- | --------------------------- | ------------------------------------------------- |
| Training   | `mop_cleanxclr_train.csv` | Optimize model parameters.                        |
| Validation | `mop_cleanxclr_valid.csv` | Select the best checkpoint.                       |
| Test       | `mop_cleanxclr_test.csv`  | Compute final metrics after checkpoint selection. |

Each CSV should contain these columns:

| Column               | Content                                                                              |
| -------------------- | ------------------------------------------------------------------------------------ |
| `smiles`           | Valid molecular SMILES string.                                                       |
| `canonical_smiles` | Canonical SMILES identifying the underlying molecule; used during pool construction. |
| `odor`             | Semicolon-separated odor labels, such as `fruity;sweet`.                           |

Use consistent label spelling and avoid missing values in required columns. Canonical molecules should be disjoint across the three splits. If molecular augmentation is used, keep all variants of the same molecule in its original split.

The pretraining script currently encodes 27 odor classes. The downstream script predicts the 21 classes defined in its `TARGET_LABELS` configuration. Its `CLASS_WEIGHTS_CONFIG` and `CLASS_THRESHOLDS_CONFIG` specify the positive-class loss weights and prediction thresholds. See [the downstream script](./finetune/finetune_odor_lessclass_weighted.py) when adapting the label set.

Dataset names and filenames must agree across all stages. For a different dataset key, check the loaders in [the pool-building script](./feature_extraction/find_similar_co.py) and the `get_split_dataset_filename` methods in the training scripts. Renaming a directory alone does not add a new supported task to the pool builder.
