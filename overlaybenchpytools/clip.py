import json
import torch
import os.path
import pathlib
import open_clip

from PIL import Image
from tqdm import tqdm
from .utils import _setup_logger
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPScore:
    def __init__(self, root, gt_json, save_dir, extension, resolution, batch_size,
                 model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", images=None, logger=None):
        """
        Initialize the CLIPScore class.
        :param root: The root directory containing images.
        :param gt_json: Dict: the ground truth JSON file containing annotations.
        :param save_dir: The directory where results will be saved.
        :param extension: The file extension of the images (e.g., 'jpg', 'png').
        :param resolution: The resolution to which the bounding boxes should be scaled (e.g., 512).
        :param batch_size: The batch size for processing images.
        :param model_name: The name of the CLIP model to use (default is "ViT-B-32").
        :param pretrained: The pretrained weights to use for the CLIP model (default is "laion2b_s34b_b79k").
        :param images: Optional list of PIL Image objects. If provided, these will be used instead of loading from disk.
        :param logger: Logger instance for logging (optional). If None, a new logger will be created.
        """
        if logger is None:
            self.logger = _setup_logger(os.path.join(save_dir, 'overlay_bench.log'))
        else:
            self.logger = logger
        self.root = pathlib.Path(root)
        self.gt_json = gt_json
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.extension = extension
        self.resolution = resolution
        self.batch_size = batch_size
        self.images = images if images is not None else []
        self.logger.info(f"CLIPScore initialized with model {model_name} and pretrained {pretrained}")
        self.model, self.tokenizer, self.preprocess = self._load_model(model_name, pretrained)


    @staticmethod
    def _load_model(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        """
        Load the CLIP model and tokenizer.
        :param model_name: The name of the CLIP model to use (default is "ViT-B-32").
        :param pretrained: The pretrained weights to use for the CLIP model (default is "laion2b_s34b_b79k").
        :return: The loaded model, tokenizer, and preprocessing function.
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return model.to(device), tokenizer, preprocess

    def _load_image(self, source):
        """
        Load an image from a file path or PIL Image object and preprocess it.
        :param source: The source of the image, either a file path (str) or a PIL Image object.
        :return: Preprocessed image tensor.
        """
        if isinstance(source, str):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, Image.Image):
            img = source
        else:
            raise ValueError("Source must be a file path (str) or a PIL Image object.")
        return self.preprocess(img)
        # return img

    @torch.no_grad()
    def clip_similarity(self, images, texts):
        """
        Calculate the CLIP similarity between images and texts.
        :param images: A single image file path, a PIL Image object, or a list of such items.
        :param texts: A single text string or a list of text strings.
        :return: A tensor of cosine similarities between the images and texts.
        """
        if not isinstance(images, (list, tuple)): images = [images]
        if not isinstance(texts, (list, tuple)): texts = [texts]

        # Load & stack
        img_batch = torch.stack([self._load_image(i) for i in images]).to(device)

        text_tokens = self.tokenizer(texts).to(device)

        img_feats = self.model.encode_image(img_batch)
        text_feats = self.model.encode_text(text_tokens)

        # Unit‑normalise then cosine‑similarity = dot‑product
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats * text_feats).sum(dim=-1)
        return sims.squeeze()  # drop dimensions of size 1 for convenience

    def evaluation(self):
        """
        Evaluate the CLIP similarity for global and local annotations.
        :return: The average global and local CLIP similarity scores.
        """
        annotations = list(self.gt_json.items())

        res = {
            'global_clip_sim_avg': 0,
            'local_clip_sim_avg': 0,
            'global_clip_sim_per_img': {},
            'local_clip_sim_per_img': defaultdict(dict),
        }
        global_sim_avg = 0

        bs = self.batch_size
        root = self.root

        self.logger.info(f"Calculating Global CLIP similarity with batch size={bs}")
        for i in tqdm(range(0, len(annotations), bs), desc=f"Calculate CLIP similarity with batch size={bs}"):
            batch = annotations[i: i + bs]
            if len(self.images) != 0:
                images = [self.images[data[0]] for data in batch]
            else:
                images = [os.path.join(root, data[0] + f".{self.extension}") for data in batch]
            texts = [data[1]['caption'] for data in batch]
            global_sims = self.clip_similarity(images, texts)
            global_sim_avg += global_sims.sum().item()
            for j, data in enumerate(batch):
                res['global_clip_sim_per_img'][data[0]] = {
                    "clip_score": global_sims[j].item(),
                    "caption": data[1]['caption'],
                    "image_path": os.path.join(root, data[0] + f".{self.extension}"),
                }
        res['global_clip_sim_avg'] = global_sim_avg / len(annotations)

        self.logger.info("Convert annotations to flat format")
        annotations_flat = []
        for annotation in annotations:
            img_id = annotation[0]
            for cat, cat_ann in annotation[1]['annotations'].items():
                if 'bbox' not in cat_ann or 'local_prompts' not in cat_ann:
                    continue
                bbox = cat_ann['bbox']
                if self.resolution == 512:
                    bbox = [int(b / 1024 * 512) for b in bbox]
                annotations_flat.append([img_id, bbox, cat_ann['local_prompts'], cat])

        self.logger.info(f"Calculating Local CLIP similarity with batch size={bs}")
        local_sim_avg = 0
        for i in tqdm(range(0, len(annotations_flat), bs), desc=f"Calculate CLIP similarity with batch size={bs}"):
            batch = annotations_flat[i: i + bs]
            if len(self.images) != 0:
                images = [self.images[img_id].crop(bbox) for img_id, bbox, _, _ in batch]
            else:
                images = [Image.open(os.path.join(root, image + f".{self.extension}")).convert("RGB").crop(bbox) for
                          image, bbox, _, _ in batch]
            texts = [data[2] for data in batch]
            local_sims = self.clip_similarity(images, texts)
            local_sim_avg += local_sims.sum().item()
            for j, data in enumerate(batch):
                res['local_clip_sim_per_img'][data[0]][data[-1]] = {
                    "clip_score": local_sims[j].item(),
                    "bbox": data[1],
                    "local_prompts": data[2],
                    "image_path": os.path.join(root, data[0] + f".{self.extension}"),
                }
        res['local_clip_sim_avg'] = local_sim_avg / len(annotations_flat)
        self.logger.info(f"Global CLIP similarity: {res['global_clip_sim_avg']}")
        self.logger.info(f"Local CLIP similarity: {res['local_clip_sim_avg']}")
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.save_dir, 'clip_score.json'), "w") as f:
            json.dump(res, f, indent=4)
        self.logger.info(f"CLIP scores saved to {self.save_dir / 'clip_score.json'}")
        return res['global_clip_sim_avg'], res['local_clip_sim_avg']
