"""FID between generated images and the OverLayBench GT images of the same split, computed with pyiqa.

`FIDScore(...)(files, split)` returns exactly what `pyiqa.create_metric("fid")(gen_dir, gt_dir, mode=mode,
num_workers=..., batch_size=...)` returns for a folder holding those files, but the GT features of each split are
extracted once and cached under the cache dir.
"""
import os
import tempfile
from pathlib import Path

import numpy as np

from .data import default_cache_dir, export_gt_images


class FIDScore:
    def __init__(self, device=None, mode="clean", num_workers=8, batch_size=32, cache_dir=None, logger=None):
        import pyiqa
        import torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.metric = pyiqa.create_metric("fid", device=self.device)
        self.version = pyiqa.__version__
        self.mode = mode
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.logger = logger

    def _folder_features(self, folder):
        # Same call as pyiqa's FID.forward for a pair of folders.
        from pyiqa.archs.fid_arch import get_folder_features
        net = self.metric.net
        return get_folder_features(str(folder), net.model, num_workers=self.num_workers, batch_size=self.batch_size,
                                   test_img_size=net.test_img_size, device=self.device, mode=self.mode,
                                   description=f"FID {os.path.basename(str(folder))}: ", verbose=False)

    def features(self, files):
        """Inception features of `files`, in the order pyiqa would read them from a folder (sorted paths)."""
        from pyiqa.utils.img_util import scandir_images
        files = [os.path.abspath(f) for f in files]
        folders = {os.path.dirname(f) for f in files}
        if len(folders) == 1:
            folder = folders.pop()
            if sorted(scandir_images(folder)) == sorted(files):
                return self._folder_features(folder)
        # The folder holds other images too: read only the requested ones through a folder of symlinks.
        with tempfile.TemporaryDirectory(prefix="overlaybench_fid_") as tmp:
            names = set()
            for f in files:
                name = os.path.basename(f)
                if name in names:
                    raise ValueError(f"duplicate file name {name} in FID inputs")
                names.add(name)
                os.symlink(f, os.path.join(tmp, name))
            return self._folder_features(tmp)

    def gt_features(self, split):
        cache = self.cache_dir / "fid_features" / f"{split}_{self.mode}_pyiqa-{self.version}.npy"
        if cache.exists():
            return np.load(cache)
        if self.logger:
            self.logger.info(f"Extracting FID features of the {split} GT images (cached at {cache})")
        feats = self._folder_features(export_gt_images(split, self.cache_dir / "gt_images"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, feats)
        return feats

    def __call__(self, files, split):
        from pyiqa.archs.fid_arch import frechet_distance
        feats1, feats2 = self.features(files), self.gt_features(split)
        mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
        mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
        return float(frechet_distance(mu1, sig1, mu2, sig2))
