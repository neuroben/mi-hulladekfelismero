# mi-hulladekfelismero - Waste Classifier

University assignment project for multi-class waste recognition with transfer
learning in PyTorch.

The model classifies four categories:

| Index | Class |
|-------|-------|
| 0 | glass |
| 1 | metal |
| 2 | paper |
| 3 | plastic |

The effective class order is determined by the alphabetical order of the class
folder names used by `torchvision.datasets.ImageFolder`.

## Repository map

```text
mi-hulladekfelismero/
|-- app/
|   `-- app.py                # Gradio web UI for image upload and prediction
|-- dataset/
|   |-- README.md             # Image placement guide and recommended counts
|   |-- train/
|   |   |-- glass/
|   |   |-- metal/
|   |   |-- paper/
|   |   `-- plastic/
|   |-- val/
|   |   |-- glass/
|   |   |-- metal/
|   |   |-- paper/
|   |   `-- plastic/
|   `-- test/
|       |-- glass/
|       |-- metal/
|       |-- paper/
|       `-- plastic/
|-- models/                   # Saved checkpoints, e.g. best_model.pth
|-- notebooks/                # Experiments / exploration
|-- src/
|   |-- check_dataset.py      # Validates dataset folders and counts images
|   |-- data.py               # ImageFolder + transforms + DataLoaders
|   |-- evaluate.py           # Test-set evaluation and metrics
|   |-- model.py              # ResNet18 model builder / loader
|   |-- predict.py            # Single-image inference
|   |-- train.py              # Training loop
|   `-- utils.py              # Device, seed, and accuracy helpers
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## How the application works

1. `src/data.py` loads images from `dataset/train`, `dataset/val`, and
   `dataset/test` using `ImageFolder`.
2. `src/train.py` trains a ResNet18-based classifier and saves the best model to
   `models/best_model.pth`.
3. `src/evaluate.py` loads that checkpoint and reports test accuracy,
   confusion matrix, and per-class metrics.
4. `src/predict.py` runs inference on a single image.
5. `app/app.py` exposes the trained model through a simple Gradio web app.

## Dataset layout

Training, validation, and test images must live under `dataset/` with one
sub-folder per class:

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

See [dataset/README.md](dataset/README.md) for practical guidance on what images
to place in each folder and how many you should aim for.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Validate the dataset before training:

```bash
python src/check_dataset.py --data_dir dataset
```

Train the model:

```bash
python src/train.py --data_dir dataset --epochs 15 --batch_size 32
```

Evaluate the best checkpoint:

```bash
python src/evaluate.py --model_path models/best_model.pth --data_dir dataset
```

Predict a single image:

```bash
python src/predict.py --model_path models/best_model.pth --image_path path/to/image.jpg
```

Launch the Gradio app:

```bash
python app/app.py --model_path models/best_model.pth
```

Then open [http://localhost:7860](http://localhost:7860).

## Transfer learning strategy

1. Stage 1 (default): train only the new classification head with the backbone
   frozen.
2. Stage 2: pass `--unfreeze_all` to fine-tune the full network once the head
   starts converging.

## License

MIT
