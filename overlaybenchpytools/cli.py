"""overlaybench-eval: evaluate generated images on OverLayBench.

    overlaybench-eval --root runs/my_model                      # every split and seed folder, all metrics
    overlaybench-eval --root runs/my_model --metrics clip fid   # no judge needed
    overlaybench-eval --root runs/my_model --dp 2 --tp 4        # two vLLM engines on 8 GPUs

Images are read from ROOT/{simple,medium,hard}/{seed}/{image_id}.{ext}.
"""
import argparse
import json
import os

from .data import SPLITS
from .meter import METRICS


def _vllm_arg(text):
    key, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def build_parser():
    p = argparse.ArgumentParser(prog="overlaybench-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="folder with the generated images")
    p.add_argument("--save-dir", default="./metrics", help="where predictions, scores and logs are written")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--seeds", nargs="+", help="seed folders to evaluate (default: all folders under ROOT/<split>)")
    p.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS,
                   help="bbox = mIoU and O-mIoU, entity = SR_E, relation = SR_R (default: all)")
    p.add_argument("--extension", default="png")
    p.add_argument("--resolution", type=int, default=1024, choices=(512, 1024))
    p.add_argument("--resume", action="store_true", help="reuse outputs already in --save-dir")
    p.add_argument("--num-workers", type=int, default=16, help="threads for image loading and preprocessing")
    p.add_argument("--cache-dir", help="GT images and FID features (default: $OVERLAYBENCH_CACHE or ~/.cache/overlaybench)")

    judge = p.add_argument_group("judge (Qwen2.5-VL-32B-Instruct)")
    judge.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    judge.add_argument("--tp", type=int, help="tensor parallel size of each vLLM engine (default: visible GPUs / dp); "
                                              "--tp 2 on 48 GB GPUs also needs --vllm-arg max_model_len=32768")
    judge.add_argument("--dp", type=int, default=1, help="number of vLLM engines on disjoint GPUs")
    judge.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                       help="vLLM memory fraction; the rest leaves room for CLIP/FID on the first GPU")
    judge.add_argument("--merge-judge-tasks", action="store_true",
                       help="send the bbox, entity and relationship prompts of a split to the judge together")
    judge.add_argument("--bs-qwen", default=None, help="judge batch size, 'all' for the whole split "
                                                       "(default: all with vllm, 8 with hf)")
    judge.add_argument("--vllm-arg", type=_vllm_arg, action="append", default=[], metavar="KEY=VALUE",
                       help="extra vllm.LLM argument, e.g. --vllm-arg max_num_seqs=128 (repeatable)")

    other = p.add_argument_group("CLIP and FID")
    other.add_argument("--bs-clip", type=int, default=1024)
    other.add_argument("--fid-batch-size", type=int, default=32)
    other.add_argument("--fid-num-workers", type=int, default=8)
    other.add_argument("--fid-device", default=None, help="default: cuda if available")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from .meter import JUDGE_METRICS, OverLayBenchMeter

    use_vllm = args.backend == "vllm"
    vllm_args = {}
    if use_vllm and any(m in JUDGE_METRICS for m in args.metrics):
        tp = args.tp
        if tp is None:
            from .judge import visible_gpus
            tp = max(1, len(visible_gpus()) // args.dp)
        vllm_args = {"tensor_parallel_size": tp, "gpu_memory_utilization": args.gpu_memory_utilization,
                     **dict(args.vllm_arg)}
    bs_qwen = args.bs_qwen or ("all" if use_vllm else 8)
    meter = OverLayBenchMeter(
        root=args.root, extension=args.extension, save_dir=args.save_dir, resolution=args.resolution,
        bs_qwen=bs_qwen if bs_qwen == "all" else int(bs_qwen), bs_clip=args.bs_clip,
        use_vllm=use_vllm, vllm_args=vllm_args, metrics=args.metrics, data_parallel_size=args.dp,
        merge_judge_tasks=args.merge_judge_tasks, num_workers=args.num_workers, cache_dir=args.cache_dir,
        fid_args={"device": args.fid_device, "num_workers": args.fid_num_workers, "batch_size": args.fid_batch_size},
        resume=args.resume)
    try:
        meter.run(args.splits, args.seeds)
    finally:
        meter.close()


if __name__ == "__main__":
    main()
