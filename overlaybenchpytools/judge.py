"""The Qwen2.5-VL-32B judge behind bbox prediction, entity VQA and relationship VQA.

Every backend turns the same chat messages into the same model inputs as the original meter did and returns the
generated texts in input order:

- VLLMJudge: one vLLM engine in this process (`use_vllm=True`).
- DataParallelVLLMJudge: several vLLM engines, each in its own process on its own GPUs; requests are split into
  contiguous shards and each engine builds its messages from the image files.
- HFJudge: transformers `generate` (`use_vllm=False`).
"""
import atexit
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_NEW_TOKENS = 1024


def load_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL, use_fast=True, padding_side='left')


def vision_inputs(messages, pool=None, chunks=64):
    """`process_vision_info(messages)[0]` (one image per message), computed on chunks in a thread pool."""
    from qwen_vl_utils import process_vision_info
    if pool is None or len(messages) <= chunks:
        return process_vision_info(messages)[0]
    size = -(-len(messages) // chunks)
    images = []
    for part in pool.map(lambda i: process_vision_info(messages[i:i + size])[0], range(0, len(messages), size)):
        images.extend(part)
    return images


def _halves(fn, messages):
    """Retry a batch that ran out of memory as two halves."""
    mid = len(messages) // 2
    return fn(messages[:mid]) + fn(messages[mid:])


class VLLMJudge:
    def __init__(self, vllm_args=None, num_workers=16, logger=None):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM
        self.model = LLM(model=MODEL, **(vllm_args or {}))
        self.sampling_params = self.model.get_default_sampling_params()
        self.sampling_params.max_tokens = MAX_NEW_TOKENS
        self.processor = load_processor()
        self.pool = ThreadPoolExecutor(max(1, num_workers))
        self.logger = logger

    def __call__(self, messages):
        import torch
        if not messages:
            return []
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = vision_inputs(messages, self.pool)
        inputs = [{"prompt": t, "multi_modal_data": {"image": [image]}} for t, image in zip(text, images)]
        try:
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        except torch.cuda.OutOfMemoryError:
            if len(messages) == 1:
                raise
            if self.logger:
                self.logger.warning(f"Out of memory error with batch size {len(messages)}")
            del text, images, inputs
            torch.cuda.empty_cache()
            return _halves(self, messages)
        return [output.outputs[0].text for output in outputs]

    def run(self, jobs, resolution, images=None, paths=None):
        """jobs: [(prompt_fn, annotation, image key)] -> generated texts. Uses the in-memory `images`."""
        return self([prompt_fn(ann, images[key], resolution) for prompt_fn, ann, key in jobs])

    def close(self):
        self.pool.shutdown(wait=False)


class HFJudge:
    def __init__(self, logger=None):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = load_processor()
        self.logger = logger

    def __call__(self, messages):
        import torch
        from qwen_vl_utils import process_vision_info
        if not messages:
            return []
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')
        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        except torch.cuda.OutOfMemoryError:
            if len(messages) == 1:
                raise
            if self.logger:
                self.logger.warning(f"Out of memory error with batch size {len(messages)}")
            del text, image_inputs, video_inputs, inputs
            torch.cuda.empty_cache()
            return _halves(self, messages)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def run(self, jobs, resolution, images=None, paths=None):
        return self([prompt_fn(ann, images[key], resolution) for prompt_fn, ann, key in jobs])

    def close(self):
        pass


def _dp_worker(conn, rank, gpus, vllm_args, num_workers, judge_cls):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if rank > 0 and "VLLM_CACHE_ROOT" not in os.environ:
        # Engines with the same config would share (and race on) one torch.compile cache directory.
        root = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "vllm")
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(root, f"dp_rank{rank}")
    try:
        from .data import load_images
        judge = judge_cls(vllm_args, num_workers)
        conn.send(("ready", None))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            jobs, resolution = msg
            images = load_images({path: path for path in dict.fromkeys(path for _, _, path in jobs)}, num_workers)
            conn.send(("ok", judge([prompt_fn(ann, images[path], resolution) for prompt_fn, ann, path in jobs])))
        except BaseException:
            conn.send(("error", traceback.format_exc()))
    judge.close()


class DataParallelVLLMJudge:
    """`data_parallel_size` vLLM engines with `tensor_parallel_size` GPUs each, on disjoint GPU groups."""

    def __init__(self, data_parallel_size, vllm_args=None, num_workers=16, gpus=None, logger=None,
                 judge_cls=VLLMJudge):
        import multiprocessing as mp
        vllm_args = dict(vllm_args or {})
        tp = vllm_args.setdefault("tensor_parallel_size", 1)
        gpus = list(gpus) if gpus is not None else visible_gpus()
        need = data_parallel_size * tp
        if len(gpus) < need:
            raise ValueError(f"data_parallel_size={data_parallel_size} x tensor_parallel_size={tp} needs {need} GPUs, "
                             f"but only {len(gpus)} are visible")
        self.logger = logger
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for rank in range(data_parallel_size):
            group = ",".join(str(g) for g in gpus[rank * tp:(rank + 1) * tp])
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_dp_worker, args=(child, rank, group, vllm_args, num_workers, judge_cls),
                               daemon=False)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
            if logger:
                logger.info(f"vLLM engine {rank} starting on GPUs {group}")
        atexit.register(self.close)
        for rank, conn in enumerate(self.conns):
            status, err = conn.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"vLLM engine {rank} failed to start:\n{err}")

    def run(self, jobs, resolution, images=None, paths=None):
        """jobs: [(prompt_fn, annotation, image key)] -> generated texts. Engines read the images from `paths`."""
        n = len(self.conns)
        size = -(-len(jobs) // n)
        shards = [jobs[i * size:(i + 1) * size] for i in range(n)]
        for conn, shard in zip(self.conns, shards):
            conn.send(([(fn, ann, paths[key]) for fn, ann, key in shard], resolution))
        texts = []
        for rank, conn in enumerate(self.conns):
            status, out = conn.recv()
            if status != "ok":
                raise RuntimeError(f"vLLM engine {rank} failed:\n{out}")
            texts.extend(out)
        return texts

    def close(self):
        for conn in getattr(self, "conns", []):
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in getattr(self, "procs", []):
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        self.conns, self.procs = [], []


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip()]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]
