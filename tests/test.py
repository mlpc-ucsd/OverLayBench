from overlaybenchpytools.meter import OverLayBenchMeter

import argparse

parser = argparse.ArgumentParser(description="Evaluate OverLayBench")
parser.add_argument("--root", type=str)
parser.add_argument("--save_dir", type=str, default="./results")
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--tensor_parallel_size", type=int, default=8)
parser.add_argument("--seeds", type=int, nargs='+')

args = parser.parse_args()

if __name__ == "__main__":
    meter = OverLayBenchMeter(
        root=args.root,
        extension='png', save_dir=args.save_dir,
        resolution=args.resolution, bs_qwen="all", use_vllm=True,
        vllm_args={"tensor_parallel_size": args.tensor_parallel_size})
    for split in ["simple", "medium", "hard"]:
        for seed in args.seeds:
            meter.set_split(split, seed)
            meter.evaluate()
