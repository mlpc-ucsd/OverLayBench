import json
import os.path
import time
from ast import literal_eval
from pathlib import Path

from tqdm import tqdm

from .data import SPLITS, annotations_dict, image_path, load_images, load_split
from .utils import (
    bbox_prompt, entity_prompt, relationship_prompt,
    extract_and_parse_json, calculate_success_rate,
    calculate_overlap_iou,
    _setup_logger
)

METRICS = ("clip", "fid", "bbox", "entity", "relation")
JUDGE_METRICS = ("bbox", "entity", "relation")
JUDGE_TASKS = {  # metric -> (task name, prompt, output file)
    "bbox": ("BBox Prediction", bbox_prompt, "baseline_bbox_predictions.json"),
    "entity": ("Entity VQA", entity_prompt, "entity_VQA.json"),
    "relation": ("Relationship VQA", relationship_prompt, "relation_VQA.json"),
}
COLUMNS = ["mIoU", "O-mIoU", "SR_E", "SR_R", "CLIP_global", "CLIP_local", "FID"]

IMAGE_LAYOUT = """
EXP_NAME
├── simple
│   ├── seed_1
│   │   ├── img_id_1.png
│   │   ├── img_id_2.png
│   │   ├── img_id_3.png
│   │   └── ...
│   ├── seed_2
│   └── seed_3
├── medium
│   ├── seed_1
│   ├── seed_2
│   └── seed_3
└── hard
    ├── seed_1
    ├── seed_2
    └── seed_3
"""


def discover_seeds(root, split):
    folder = Path(root) / split
    return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


class OverLayBenchMeter:
    def __init__(self,
                 root,
                 extension='png',
                 save_dir='./metrics',
                 resolution=1024,
                 bs_qwen=8,
                 bs_clip=1024,
                 use_vllm=False,
                 vllm_args=None,
                 metrics=None,
                 data_parallel_size=1,
                 merge_judge_tasks=False,
                 num_workers=16,
                 cache_dir=None,
                 fid_args=None,
                 resume=False):
        """
        Initializes the OverLayBenchMeter for evaluating the OverLayBench dataset.

        :param root: The root directory where the generated images are stored, the image filename should be in f"{img_id}".{extension}.
        :param extension: The extension of the image files, default is 'png'.
        :param save_dir: The directory where the results will be saved.
        :param resolution: The resolution of the images, default is 1024, can be 1024 or 512.
        :param bs_qwen: The batch size for Qwen model inference, default is 8. We recommend using `all` when `use_vllm` is True, which will set the batch size to the length of the dataset.
        :param bs_clip: The batch size for CLIPScore model inference, default is 1024.
        :param use_vllm: Whether to use vLLM for inference, default is False.
        :param vllm_args: Additional arguments for vLLM, if `use_vllm` is True. Please refer to the vLLM documentation (https://docs.vllm.ai/en/latest/api/vllm/index.html#vllm.LLM) for reference.
        :param metrics: Metrics computed by `evaluate()`, any of "clip", "fid", "bbox" (mIoU, O-mIoU), "entity" (SR_E) and "relation" (SR_R). Default: all of them.
        :param data_parallel_size: Number of vLLM engines, each with `vllm_args["tensor_parallel_size"]` GPUs of its own (vLLM only).
        :param merge_judge_tasks: Send the bbox, entity and relationship prompts of a split to the judge together instead of one task after another.
        :param num_workers: Threads for image loading and preprocessing.
        :param cache_dir: Where GT images and their FID features are cached, default $OVERLAYBENCH_CACHE or ~/.cache/overlaybench.
        :param fid_args: Extra arguments of `FIDScore` (device, mode, num_workers, batch_size).
        :param resume: Reuse the outputs already saved in `save_dir` instead of recomputing them.

        The Qwen2.5-VL-32B judge is loaded the first time a judge metric is computed, so CLIP and FID do not need it.
        """
        self.root = root
        self.extension = extension
        self.save_root = save_dir
        Path(self.save_root).mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.bs_qwen = bs_qwen
        self.bs_clip = bs_clip
        self.use_vllm = use_vllm
        self.vllm_args = vllm_args or {}
        self.metrics = tuple(metrics) if metrics is not None else METRICS
        unknown = set(self.metrics) - set(METRICS)
        if unknown:
            raise ValueError(f"Unknown metrics {sorted(unknown)}, choose from {METRICS}")
        self.data_parallel_size = data_parallel_size
        if data_parallel_size > 1 and not use_vllm:
            raise ValueError("data_parallel_size > 1 needs use_vllm=True")
        self.merge_judge_tasks = merge_judge_tasks
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.fid_args = fid_args or {}
        self.resume = resume
        self.logger = _setup_logger(os.path.join(self.save_root, 'overlay_bench.log'))
        self.logger.info(f"Initializing OverLayBenchMeter with:\n"
                         f"root: {self.root}\n"
                         f"extension: {self.extension}\n"
                         f"save_dir: {self.save_root}\n"
                         f"resolution: {self.resolution}\n"
                         f"bs_qwen: {bs_qwen}\n"
                         f"bs_clip: {bs_clip}\n"
                         f"metrics: {', '.join(self.metrics)}\n"
                         f"judge: {self._judge_description()}")

        # Load the dataset
        self.split = None
        self.seed = None
        self.save_dir = None
        self.dataset = None
        self.annotations = None
        self._images = None

        self._judge = None
        self._fid = None
        self._written = set()  # results.json files written by this meter

    def _judge_description(self):
        if not self.use_vllm:
            return "transformers"
        tp = self.vllm_args.get("tensor_parallel_size", 1)
        return f"vLLM, {self.data_parallel_size} engine(s) x tensor_parallel_size={tp}" + \
            (", tasks merged" if self.merge_judge_tasks else "")

    def _read_images(self):
        paths = {img_id: image_path(self.root, self.split, self.seed, img_id, self.extension)
                 for img_id in self.annotations}
        try:
            return load_images(paths, self.num_workers)
        except Exception as e:
            self.logger.error(f"Error loading images: {e}")
            self.logger.error("OverLayBenchMeter expected the images to be stored in the following path format: ")
            self.logger.error(IMAGE_LAYOUT)
            raise e

    @property
    def images(self):
        """{image_id: PIL image} of the current split and seed, loaded on first use."""
        if self._images is None and self.split is not None:
            self._images = self._read_images()
        return self._images

    def set_split(self, split, seed):
        self.split = split
        self.seed = seed
        self.save_dir = os.path.join(self.save_root, self.split, str(self.seed))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Loading OverLayBench dataset {self.split} split...")
        self.dataset = load_split(self.split)
        self.annotations = annotations_dict(self.dataset)
        self._images = None
        self.logger.info(f"Dataset loaded with {len(self.dataset)} samples.")

    def check_set_split(self):
        if self.split is None:
            raise ValueError("Split is not set. Please set the split using set_split() before evaluating.")

    # ------------------------------------------------------------------ judge (Qwen2.5-VL-32B)

    @property
    def judge(self):
        if self._judge is None:
            from .judge import DataParallelVLLMJudge, HFJudge, VLLMJudge
            self.logger.info(f"Loading Qwen2.5-VL-32B-Instruct judge ({self._judge_description()})...")
            start = time.time()
            if not self.use_vllm:
                self._judge = HFJudge(logger=self.logger)
            elif self.data_parallel_size > 1:
                self._judge = DataParallelVLLMJudge(self.data_parallel_size, self.vllm_args, self.num_workers,
                                                    logger=self.logger)
            else:
                self._judge = VLLMJudge(self.vllm_args, self.num_workers, logger=self.logger)
            self.logger.info(f"Judge loaded in {time.time() - start:.0f}s.")
        return self._judge

    def close(self):
        """Shut down the judge (and its engine processes)."""
        if self._judge is not None:
            self._judge.close()
            self._judge = None

    def _task_annotations(self):
        """Fresh annotation dicts for one task, parsed from the dataset like the original meter did per task."""
        return [{'img_id': data['image_id'], **literal_eval(data['annotation'])} for data in self.dataset]

    def _judge_outputs(self, metrics):
        """Judge outputs of `metrics` (a subset of JUDGE_METRICS) -> {metric: {img_id: parsed json}}."""
        results, todo = {}, []
        for m in metrics:
            task, _, file_name = JUDGE_TASKS[m]
            output_file = os.path.join(self.save_dir, file_name)
            if self.resume and os.path.exists(output_file):
                with open(output_file) as f:
                    results[m] = json.load(f)
                self.logger.info(f"{task}: reusing {output_file}")
            else:
                todo.append(m)
        groups = [todo] if (self.merge_judge_tasks and todo) else [[m] for m in todo]
        for group in groups:
            names = " + ".join(JUDGE_TASKS[m][0] for m in group)
            self.logger.info(f"Starting {names}...")
            start = time.time()
            anns = {m: self._task_annotations() for m in group}
            n = len(self.dataset)
            bs = n if self.bs_qwen == "all" else int(self.bs_qwen)
            paths = {img_id: image_path(self.root, self.split, self.seed, img_id, self.extension)
                     for img_id in self.annotations}
            judge = self.judge
            images = None if self.data_parallel_size > 1 else self.images  # engine processes read the files
            texts = {m: [] for m in group}
            for i in tqdm(range(0, n, bs)):
                # image-major order: the prompts of one image are adjacent when tasks are merged
                order = [(m, j) for j in range(i, min(i + bs, n)) for m in group]
                jobs = [(JUDGE_TASKS[m][1], anns[m][j], anns[m][j]['img_id']) for m, j in order]
                for (m, _), text in zip(order, judge.run(jobs, self.resolution, images, paths)):
                    texts[m].append(text)
            for m in group:
                task, _, file_name = JUDGE_TASKS[m]
                res = {f"{ann['img_id']:08}": extract_and_parse_json(text) for text, ann in zip(texts[m], anns[m])}
                output_file = os.path.join(self.save_dir, file_name)
                with open(output_file, 'w') as f:
                    json.dump(res, f, indent=4)
                self.logger.info(f"Results saved to {output_file}")
                results[m] = res
            self.logger.info(f"{names} completed in {time.time() - start:.0f}s.")
        return results

    def _bbox_metrics(self, res):
        bbox_iou, overlap_iou = calculate_overlap_iou(self.annotations, res, self.resolution)
        self.logger.info(f"Bbox IoU: {bbox_iou:.4f}, Overlap IoU: {overlap_iou:.4f}")
        return bbox_iou, overlap_iou

    def _success_rate(self, res, task):
        success_rate = calculate_success_rate(res, self.annotations, task)
        self.logger.info(f"Success rate for {task}: {success_rate:.4f}")
        return success_rate

    def bbox_predication(self):
        """
        Performs bounding box prediction using the Qwen model and calculates IoU metrics.

        :return: A tuple containing the bounding box IoU and overlap IoU.
        """
        self.check_set_split()
        return self._bbox_metrics(self._judge_outputs(["bbox"])["bbox"])

    def entity_vqa(self):
        """
        Performs Entity VQA using the Qwen model and calculates the success rate.
        :return: The success rate of the Entity VQA task.
        """
        self.check_set_split()
        return self._success_rate(self._judge_outputs(["entity"])["entity"], 'Entity VQA')

    def relationship_vqa(self):
        """
        Performs Relationship VQA using the Qwen model and calculates the success rate.
        :return: The success rate of the Relationship VQA task.
        """
        self.check_set_split()
        return self._success_rate(self._judge_outputs(["relation"])["relation"], 'Relationship VQA')

    # ------------------------------------------------------------------ CLIP and FID

    def clip_score(self):
        """
        Calculates the CLIP score for the generated images against the ground truth annotations.
        :return: The CLIP score evaluation results, containing a global CLIPScore and a local CLIPScore.
        """
        self.check_set_split()
        output_file = os.path.join(self.save_dir, 'clip_score.json')
        if self.resume and os.path.exists(output_file):
            with open(output_file) as f:
                res = json.load(f)
            self.logger.info(f"CLIP: reusing {output_file}")
            return res['global_clip_sim_avg'], res['local_clip_sim_avg']
        import torch
        from .clip import CLIPScore
        # Load the CLIPScore model
        self.logger.info("Initializing CLIPScore model...")
        clip_score_model = CLIPScore(
            root=os.path.join(self.root, self.split),
            gt_json=self.annotations,
            save_dir=self.save_dir,
            extension=self.extension,
            resolution=self.resolution,
            batch_size=self.bs_clip,
            images=self.images,
            logger=self.logger,
            num_workers=self.num_workers,
        )
        self.logger.info("CLIPScore model initialized successfully.")
        ret = clip_score_model.evaluation()
        del clip_score_model  # Free memory
        torch.cuda.empty_cache()

        return ret

    def fid(self):
        """
        FID (pyiqa, clean-fid protocol) between the generated images of the split and its GT images.
        :return: The FID score.
        """
        self.check_set_split()
        output_file = os.path.join(self.save_dir, 'fid.json')
        if self.resume and os.path.exists(output_file):
            with open(output_file) as f:
                score = json.load(f)['fid']
            self.logger.info(f"FID: reusing {output_file}")
            return score
        if self._fid is None:
            from .fid import FIDScore
            self._fid = FIDScore(cache_dir=self.cache_dir, logger=self.logger, **self.fid_args)
        self.logger.info("Calculating FID...")
        files = [image_path(self.root, self.split, self.seed, img_id, self.extension) for img_id in self.annotations]
        score = self._fid(files, self.split)
        self.logger.info(f"FID: {score:.4f}")
        with open(output_file, 'w') as f:
            json.dump({"fid": score, "mode": self._fid.mode, "pyiqa": self._fid.version, "num_images": len(files)},
                      f, indent=4)
        return score

    # ------------------------------------------------------------------ evaluation

    def _save_results(self, results):
        output_file = os.path.join(self.save_dir, 'results.json')
        merged = {}
        # Keep the numbers of earlier calls (or of the resumed run), never stale ones left by an older run.
        if os.path.exists(output_file) and (self.resume or output_file in self._written):
            with open(output_file) as f:
                merged = json.load(f)
        merged.update(split=self.split, seed=str(self.seed))
        merged.update(results)
        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=4)
        self._written.add(output_file)
        return merged

    def _log_table(self, results):
        from prettytable import PrettyTable
        columns = COLUMNS[:6] + (["FID"] if "FID" in results else [])
        pt = PrettyTable()
        pt.field_names = ["Split", "Seed"] + columns
        pt.add_row([self.split, self.seed] + [
            "-" if c not in results else (f"{results[c]:.2f}" if c == "FID" else f"{results[c]:.4f}") for c in columns])
        self.logger.info(f"Evaluation results for {self.split} split:\n{pt}")

    def evaluate(self, metrics=None):
        """
        Computes `metrics` (default: the meter's metrics) for the current split and seed, logs the result table and
        merges the numbers into `save_dir/<split>/<seed>/results.json`.
        :return: dict with the results of this split and seed.
        """
        self.check_set_split()
        metrics = self.metrics if metrics is None else tuple(metrics)
        results = {}
        if "clip" in metrics:
            results["CLIP_global"], results["CLIP_local"] = self.clip_score()
        if "fid" in metrics:
            results["FID"] = self.fid()
        judge_metrics = [m for m in JUDGE_METRICS if m in metrics]
        if judge_metrics:
            outputs = self._judge_outputs(judge_metrics)
            if "bbox" in outputs:
                results["mIoU"], results["O-mIoU"] = self._bbox_metrics(outputs["bbox"])
            if "entity" in outputs:
                results["SR_E"] = self._success_rate(outputs["entity"], 'Entity VQA')
            if "relation" in outputs:
                results["SR_R"] = self._success_rate(outputs["relation"], 'Relationship VQA')
        merged = self._save_results(results)
        self._log_table(merged)
        return merged

    def run(self, splits=SPLITS, seeds=None):
        """
        Evaluates every split x seed. CLIP and FID run first for all of them, before the judge is loaded; then the
        judge metrics. Seeds default to the sub-directories of `root/<split>`.
        :return: {split: {seed: results}}; a mean/std summary over seeds is logged and saved to `save_dir/summary.json`.
        """
        seeds_of = {split: [str(s) for s in seeds] if seeds else discover_seeds(self.root, split) for split in splits}
        for split, split_seeds in seeds_of.items():
            if not split_seeds:
                raise FileNotFoundError(f"No seed folders under {os.path.join(self.root, split)}")
        phases = [[m for m in self.metrics if m not in JUDGE_METRICS], [m for m in self.metrics if m in JUDGE_METRICS]]
        results = {}
        for phase in phases:
            if not phase:
                continue
            for split, split_seeds in seeds_of.items():
                for seed in split_seeds:
                    self.set_split(split, seed)
                    results.setdefault(split, {})[seed] = self.evaluate(phase)
            self._release_light_models()
        self._log_summary(results)
        return results

    def _release_light_models(self):
        """Free the FID network and cached GPU memory before the judge takes the GPUs."""
        self._fid = None
        self._images = None
        import sys
        if "torch" in sys.modules:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_summary(self, results):
        import statistics
        from prettytable import PrettyTable
        summary = {}
        for split, by_seed in results.items():
            summary[split] = {}
            for c in COLUMNS:
                values = [r[c] for r in by_seed.values() if c in r]
                if values:
                    summary[split][c] = {"mean": statistics.mean(values), "std": statistics.pstdev(values),
                                         "seeds": {s: r[c] for s, r in by_seed.items() if c in r}}
        with open(os.path.join(self.save_root, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        columns = [c for c in COLUMNS if any(c in s for s in summary.values())]
        pt = PrettyTable()
        pt.field_names = ["Split", "#Seeds"] + columns
        for split, s in summary.items():
            scale = {c: (1 if c == "FID" else 100) for c in columns}
            pt.add_row([split, len(results[split])] + [
                "-" if c not in s else f"{s[c]['mean'] * scale[c]:.2f} ± {s[c]['std'] * scale[c]:.2f}" for c in columns])
        self.logger.info(f"Summary over seeds (mIoU, O-mIoU, SR and CLIP in %, std over seeds):\n{pt}")
