<h1 align="center">OverLayBench: A Benchmark
for Layout-to-Image Generation with Dense Overlaps</h1>


<p align="center">
    <a href="https://arxiv.org/abs/2509.19282" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-OverLayBench-red?logo=arxiv" height="20" /></a>
    <a href="https://mlpc-ucsd.github.io/OverLayBench/" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/🌎_Website-OverLayBench-blue.svg" height="20" /></a>
    <a href="https://huggingface.co/datasets/cywang143/OverLayBench_Eval" target="_blank">
        <img alt="HF Dataset: Science-T2I" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Hugging Face-OVerLayBench-ffc107?color=ffc107&logoColor=white" height="20" /></a>
</p>

<p align="center">
    <a href="https://www.bingnanli.com"><strong><ins>Bingnan Li <sup>1*</sup></ins></strong></a>
    ·
    <a href="https://www.linkedin.com/in/chenyu-wang-profile/"><strong><ins>Chen-Yu Wang <sup>1*</sup></ins></strong></a>
    ·
    <a href="https://xxuhaiyang.github.io"><strong><ins>Haiyang Xu <sup>1*</sup></ins></strong></a>
    ·
    <a href="https://xzhang.dev"><strong><ins>Xiang Zhang <sup>1</sup></ins></strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=LE6bioEAAAAJ&hl=en"><strong><ins>Ethan Armand <sup>1</sup></ins></strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=kw6DWjsAAAAJ&hl=en"><strong><ins>Divyansh Srivastava <sup>1</sup></ins></strong></a>
    ·
    <a href="https://shanxiaojun.github.io"><strong><ins>Xiaojun Shan <sup>1</sup></ins></strong></a>
    ·
    <a href="https://zeyuan-chen.com"><strong><ins>Zeyuan Chen <sup>1</sup></ins></strong></a>
    ·
    <a href="http://www.stat.ucla.edu/~jxie/"><strong><ins>Jianwen Xie <sup>2</sup></ins></strong></a>
    ·
    <a href="https://pages.ucsd.edu/~ztu/"><strong><ins>Zhuowen Tu <sup>1</sup></ins></strong></a>
</p>
<p align="center">
    <sup>1</sup>UC San Diego
    ·
    <sup>2</sup>Lambda, Inc.
</p>

![teaser.jpg](images/teaser.jpg)
<p align="center">
    Examples from <em>OverLayBench</em> with difficulty increasing from left to right.
</p>

## 📖 Abstract

>Despite steady progress in layout-to-image generation, current methods still struggle with layouts
containing significant overlap between bounding boxes. We identify two primary challenges: (1) large
overlapping regions and (2) overlapping instances with minimal semantic distinction. Through both
qualitative examples and quantitative analysis, we demonstrate how these factors degrade generation
quality. To systematically assess this issue, we introduce OverLayScore, a novel metric that
quantifies the complexity of overlapping bounding boxes. Our analysis reveals that existing
benchmarks are biased toward simpler cases with low OverLayScore values, limiting their
effectiveness in evaluating models under more challenging conditions. To reduce this gap, we present
OverLayBench, a new benchmark featuring balanced OverLayScore distributions and high-quality
annotations. As an initial step toward improved performance on complex overlaps, we also propose
CreatiLayout-AM, a model trained on a curated amodal mask dataset. Together, our contributions
establish a foundation for more robust layout-to-image generation under realistic and challenging
scenarios.

## 🔥 News

- **[2025-09-23]**: The preprint is available on [arXiv](https://arxiv.org/abs/2509.19282)!
- **[2025-09-19]**: OverLayBench is accepted by NeurIPS 2025 D&B Track! 🎉🎉🎉
- **[2024-06-17]**: The code and the evaluation toolkit are released!

## 🚀 Quick Start

### 🤖 Install Dependencies

If you are using Multi-GPUs, we recommend you to use [vllm](https://github.com/vllm-project/vllm) for accelerated inference.

```bash
git clone https://github.com/mlpc-ucsd/OverLayBench.git
cd OverLayBench
conda create -n overlaybench python=3.10.16 --yes
conda activate overlaybench
bash install_vllm.sh
```
Otherwise, you may also choose to use the default huggingface transformers, which is slower but more stable.
```bash
git clone https://github.com/mlpc-ucsd/OverLayBench.git
cd OverLayBench
conda create -n overlaybench python=3.10.16 --yes
conda activate overlaybench
bash install.sh
```

### 🧪 Evaluate Your Own Model!
OverLayBenchMeter assumes that the generated images are organized in the following structure:
```
EXP_NAME
├── simple
│   ├── seed_1
│   │   ├── img_id_1.png
│   │   ├── img_id_2.png
│   │   ├── img_id_3.png
│   │   └── ...
│   ├── seed_2
│   └── seed_3
├── medium
│   ├── seed_1
│   ├── seed_2
│   └── seed_3
└── hard
    ├── seed_1
    ├── seed_2
    └── seed_3
```

For `vllm` based inference:

```python
from overlaybenchpytools.meter import OverLayBenchMeter
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    meter = OverLayBenchMeter(
        root='{YOUR_GENERATED_IMAGES_DIR}',
        extension='png', save_dir='./metrics',
        resolution=1024, bs_qwen="all", use_vllm=True,
        vllm_args={"tensor_parallel_size": 8})
    for split in ["simple", "medium", "hard"]:
        meter.set_split(split, '{YOUR SEED}')
        meter.evaluate()
```

For `transformers` based inference, please remove the `use_vllm` and the `vllm_args` argument and set `bs_qwen` to a reasonable size.

```python
from overlaybenchpytools.meter import OverLayBenchMeter
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    meter = OverLayBenchMeter(
        root='{YOUR_GENERATED_IMAGES_DIR}',
        extension='png', save_dir='./metrics',
        resolution=1024, bs_qwen=8)
    for split in ["simple", "medium", "hard"]:
        meter.set_split(split, '{YOUR_SEED}')
        meter.evaluate()
```

`OverLayBenchMeter` covers the evaluation of `mIoU`, `Overlay mIoU(o-mIoU)`, `Entity Success Rate (SR_E)`,
`Relashionship Success Rate (SR_R)`, `Relationship Success Rate (SR_R)`, `Global CLIPScore` and `Local CLIPScore`.
For `FID`, please refer to the [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) package.

The expected log file structure is as follows:
```metrics
metrics
├── overlay_bench.log
├── simple
│   ├── seed_1
│   │   ├── baseline_bbox_predictions.json
│   │   ├── entity_VQA.json
│   │   ├── relation_VQA.json
│   │   └── ...
│   ├── seed_2
│   └── seed_3
├── medium
│   ├── seed_1
│   ├── seed_2
│   └── seed_3
└── hard
    ├── seed_1
    ├── seed_2
    └── seed_3
```

![qual-res.jpg](images/qual-res.jpg)
<p align="center">
Comparison of generated images from different models on <em>OverLayBench</em>.
</p>

## 🤝 Acknowledgements


We deeply appreciate the contributions of the following projects:

- [QWen2.5 VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)
- [Diffusers](https://github.com/huggingface/diffusers)
- [FLUX](https://github.com/black-forest-labs/flux)
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
- [VLLM](https://github.com/vllm-project/vllm)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
## ✏️ Citation


```bibtex
@article{li2025overlaybench,
  title={OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps},
  author={Li, Bingnan and Wang, Chen-Yu and Xu, Haiyang and Zhang, Xiang and Armand, Ethan and Srivastava, Divyansh and Shan, Xiaojun and Chen, Zeyuan and Xie, Jianwen and Tu, Zhuowen},
  journal={arXiv preprint arXiv:2509.19282},
  year={2025}
}
```
```bibtex
@inproceedings{NEURIPS2025_329ad516,
 author = {Li, Bingnan and Wang, Chen-Yu and Xu, Haiyang and Zhang, Xiang and Armand, Ethan and Srivastava, Divyansh and Xiaojun, Shan and Chen, Zeyuan and Xie, Jianwen and Tu, Zhuowen},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/329ad516cf7a6ac306f29882e9c77558-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {38},
 year = {2025}
}

```
