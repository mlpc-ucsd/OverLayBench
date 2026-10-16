"""Dataset, annotation and image loading for OverLayBench."""
import os
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATASET = "cywang143/OverLayBench_Eval"
SPLITS = ("simple", "medium", "hard")


def default_cache_dir():
    """Where GT images and FID features are cached: $OVERLAYBENCH_CACHE, else ~/.cache/overlaybench."""
    return Path(os.environ.get("OVERLAYBENCH_CACHE", Path.home() / ".cache" / "overlaybench"))


def load_split(split):
    """The eval split with only `image_id` and `annotation` (decoding the GT image column is slow; only FID needs it)."""
    from datasets import load_dataset
    return load_dataset(DATASET, split=split).select_columns(["image_id", "annotation"])


def annotations_dict(dataset):
    """{image_id: annotation dict}, in dataset order."""
    return {data['image_id']: literal_eval(data['annotation']) for data in dataset}


def image_path(root, split, seed, img_id, extension):
    return os.path.join(root, split, str(seed), f"{img_id}.{extension}")


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_images(paths, num_workers=16):
    """{key: path} -> {key: RGB PIL image}, decoded in a thread pool (PNG/JPEG decoding releases the GIL)."""
    with ThreadPoolExecutor(max(1, num_workers)) as pool:
        return dict(zip(paths, pool.map(load_rgb, paths.values())))


def export_gt_images(split, out_dir):
    """Write the GT images of `split` to `out_dir/split/{image_id}.{ext}` exactly as stored in the dataset (no
    re-encoding), once. Returns the folder."""
    out = Path(out_dir) / split
    if (out / ".complete").exists():
        return out
    from datasets import Image as ImageFeature
    from datasets import load_dataset
    dataset = load_dataset(DATASET, split=split).select_columns(["image_id", "image"])
    dataset = dataset.cast_column("image", ImageFeature(decode=False))
    out.mkdir(parents=True, exist_ok=True)
    for row in dataset:
        image = row["image"]
        if image.get("bytes") is not None:
            ext = os.path.splitext(image.get("path") or "")[1].lower() or ".png"
            (out / f"{row['image_id']}{ext}").write_bytes(image["bytes"])
        else:
            load_rgb(image["path"]).save(out / f"{row['image_id']}.png")
    (out / ".complete").touch()
    return out
