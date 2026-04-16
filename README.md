# mi-hulladekfelismero

Waste classification project built with PyTorch, Gradio, and Modal.

The current model predicts four classes:

- `glass`
- `metal`
- `paper`
- `plastic`

## Project layout

```text
mi-hulladekfelismero/
|-- app/                         # legacy GUI entrypoint
|-- dataset/                     # versioned training, validation, and test images
|-- docs/
|   `-- fejlesztoi-naplo.md      # Hungarian first-person project write-up
|-- models/                      # saved checkpoints and run artifacts
|-- notebooks/                   # experiments
|-- scripts/                     # recommended CLI entrypoints
|-- src/
|   |-- waste_classifier/        # shared application package
|   `-- *.py                     # compatibility wrappers for old commands
|-- modal_train.py               # legacy Modal entrypoint
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Branch setup

The repository is intended to be used with two branches:

- `main`: application code, documentation, and trained model artifacts, but no full dataset images.
- `main-with-dataset`: everything from `main`, plus the versioned dataset images for training and evaluation.

If someone only wants to try the app with the trained model, `main` is enough.
If someone wants to retrain or inspect the full dataset locally, they should use `main-with-dataset`.

## Quick start: only try the model

Clone the lightweight branch, install dependencies, and launch the GUI:

```powershell
git clone -b main <REPO_URL>
cd mi-hulladekfelismero
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\app.py --model_path models\howa_modal_run1\best_model.pth
```

Then open the local Gradio URL printed in the terminal.

If you only want a quick CLI prediction instead of the GUI:

```powershell
python scripts\predict.py --model_path models\howa_modal_run1\best_model.pth --image_path path\to\image.jpg
```

## Full setup: code + dataset for training

If you want the full training dataset as well:

```powershell
git clone -b main-with-dataset <REPO_URL>
cd mi-hulladekfelismero
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\check_dataset.py --data_dir dataset
```

If you already cloned `main`, switch like this after fetching:

```powershell
git fetch origin
git checkout main-with-dataset
```

## Recommended commands

Create the virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Check the dataset:

```powershell
python scripts\check_dataset.py --data_dir dataset
```

Train locally:

```powershell
python scripts\train.py --data_dir dataset --epochs 10 --batch_size 32 --num_workers 0 --checkpoint_dir models\run1
```

Evaluate a checkpoint:

```powershell
python scripts\evaluate.py --model_path models\run1\best_model.pth --data_dir dataset --batch_size 32 --num_workers 0
```

Predict a single image:

```powershell
python scripts\predict.py --model_path models\run1\best_model.pth --image_path dataset\test\glass\glass101.jpg
```

Launch the GUI:

```powershell
python scripts\app.py --model_path models\run1\best_model.pth
```

Run Modal training:

```powershell
python -m modal run modal_train.py --accelerator l4 --epochs 6 --batch-size 64 --run-name howa_modal_run1
```

Legacy commands such as `python src\train.py` and `python app\app.py` still work.

## Dataset

The repository uses an `ImageFolder` layout:

```text
dataset/
  train/
    glass/
    metal/
    paper/
    plastic/
  val/
    glass/
    metal/
    paper/
    plastic/
  test/
    glass/
    metal/
    paper/
    plastic/
```

See [dataset/README.md](dataset/README.md) for placement guidance.

The current combined dataset in `main-with-dataset` contains `14528` images in total.
This includes the earlier dataset sources, the HOWA import, and an additional Kaggle Garbage Classification V2 import split into the existing `train / val / test` structure.

## Current best run

The strongest tracked result in the project came from a Modal L4 run with the HOWA-augmented dataset:

- best validation accuracy: `0.8059`
- test accuracy: `0.8161`

Artifacts are stored in [models/howa_modal_run1](models/howa_modal_run1).

## Notes

- `scripts/` contains the recommended user-facing entrypoints.
- `src/waste_classifier/` contains the shared implementation.
- `src/*.py`, `app/app.py`, and `modal_train.py` remain as compatibility wrappers.
- The project was refactored into a cleaner package structure after the initial implementation phase.
