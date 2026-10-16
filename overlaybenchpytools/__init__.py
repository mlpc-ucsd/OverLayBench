"""OverLayBench evaluation toolkit: mIoU, O-mIoU, SR_E, SR_R (Qwen2.5-VL-32B judge), CLIPScore and FID."""
__version__ = "0.2.0"
__all__ = ["OverLayBenchMeter", "__version__"]


def __getattr__(name):
    # Lazy, so that `import overlaybenchpytools` does not pull in torch, transformers or vLLM.
    if name == "OverLayBenchMeter":
        from .meter import OverLayBenchMeter
        return OverLayBenchMeter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
