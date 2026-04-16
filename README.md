# mi-hulladekfelismero — Hulladékfelismerő (Waste Classifier)

**2026 Beadandó** — Gépi tanulás / Mesterséges intelligencia

A multiclass image classifier that identifies waste categories (**paper, plastic, metal, glass**) using transfer learning with a pretrained ResNet18 backbone in PyTorch.

---

## Project Overview

This project was developed as a university assignment on waste classification using deep learning and transfer learning. The model is fine-tuned from a ResNet18 network pretrained on ImageNet, with the final classification layer replaced to output four waste categories. Training leverages frozen backbone weights initially, making it efficient even with a modest dataset.

### Waste categories

| Index | Class   |
|-------|---------|
| 0     | glass   |
| 1     | metal   |
| 2     | paper   |
| 3     | plastic |

> The actual class order is determined by the alphabetical order of the dataset folder names.

---

## Repository structure

```
mi-hulladekfelismero/
├── app/
│   └── app.py            # Gradio web demo
├── dataset/
│   ├── train/            # Training images (one sub-folder per class)
│   ├── val/              # Validation images
│   └── test/             # Test images
├── models/               # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data.py           # Dataset & DataLoader helpers
│   ├── evaluate.py       # Evaluation script (accuracy, confusion matrix)
│   ├── model.py          # Model definition / loading
│   ├── predict.py        # Single-image inference
│   ├── train.py          # Training script
│   └── utils.py          # Shared utility functions
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dataset format

The dataset must follow the `ImageFolder` layout:

```
dataset/
  train/
    glass/   image1.jpg ...
    metal/   image2.jpg ...
    paper/   ...
    plastic/ ...
  val/
    glass/   ...
    ...
  test/
    glass/   ...
    ...
```

A good free dataset to start with is the [Garbage Classification dataset on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/train.py --data_dir dataset --epochs 15 --batch_size 32
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `dataset` | Path to dataset root |
| `--epochs` | `15` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Learning rate |
| `--num_classes` | `4` | Number of output classes |
| `--unfreeze_all` | off | Unfreeze all backbone layers for full fine-tuning |
| `--checkpoint_dir` | `models` | Directory to save best model |

### Evaluation

```bash
python src/evaluate.py --model_path models/best_model.pth --data_dir dataset
```

Prints accuracy, a confusion matrix, and per-class precision / recall / F1.

### Single-image prediction

```bash
python src/predict.py --model_path models/best_model.pth --image_path path/to/image.jpg
```

### Web demo (Gradio)

```bash
python app/app.py --model_path models/best_model.pth
```

Then open http://localhost:7860 in your browser.

---

## Transfer learning strategy

1. **Stage 1 – frozen backbone** (default): Only the new classification head is trained. Fast convergence, recommended when data is limited.
2. **Stage 2 – full fine-tuning**: Pass `--unfreeze_all` to `train.py` to unfreeze all layers and fine-tune end-to-end with a lower learning rate.

---

## License

MIT
