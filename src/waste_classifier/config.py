DEFAULT_CLASSES = ["glass", "metal", "paper", "plastic"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HOWA_LABEL_MAP = {
    "glass": "glass",
    "metal": "metal",
    "plastic": "plastic",
    "carton": "paper",
}
HOWA_IGNORED_LABELS = {"__ignore__", "_background_"}
