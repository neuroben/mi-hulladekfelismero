# Dataset image guide

This project expects a `torchvision.datasets.ImageFolder` layout. That means
every image must be placed inside a split folder and inside the correct class
folder.

## Required folders

The repository already contains the expected class folders:

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

Put images here:

- `dataset/train/<class>/`: images used for learning.
- `dataset/val/<class>/`: images used during training to decide whether the
  model is improving.
- `dataset/test/<class>/`: images kept aside for final evaluation only.

## What kind of images to use

- Use real photos of waste items.
- Keep the main object visible and reasonably large in the frame.
- Prefer variety: different backgrounds, lighting, angles, distances, and item
  shapes.
- Avoid near-duplicate images taken in the same burst unless you really need
  more data.
- Keep labels strict: do not place mixed or ambiguous items into the wrong
  class folder.

Common image types that work well:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`
- `.tif` / `.tiff`

## How many images to collect

Balanced classes matter more than exact totals. Try to keep the four classes
close to each other in size.

Recommended targets per class:

| Goal | Train | Val | Test | Total per class |
|------|------:|----:|-----:|----------------:|
| Minimum usable prototype | 100 | 20 | 20 | 140 |
| Good baseline | 300 | 60 | 60 | 420 |
| Strong small-project dataset | 500 | 100 | 100 | 700 |

For 4 classes, that means roughly:

- Minimum usable prototype: about 560 images total.
- Good baseline: about 1,680 images total.
- Strong small-project dataset: about 2,800 images total.

If you are starting from one large pool of images, a good split is:

- `train`: about 70%
- `val`: about 15%
- `test`: about 15%

## Quick examples

Examples of correct placement:

```text
dataset/train/glass/bottle_001.jpg
dataset/train/paper/newspaper_014.jpg
dataset/val/metal/can_022.png
dataset/test/plastic/container_005.jpg
```

## Check before training

Run this command:

```bash
python scripts/check_dataset.py --data_dir dataset
```

It will tell you:

- whether any required folder is missing,
- how many images are present in each split and class,
- whether some class folders are empty,
- and whether the dataset looks badly imbalanced.

The legacy command `python src/check_dataset.py --data_dir dataset` still works.
