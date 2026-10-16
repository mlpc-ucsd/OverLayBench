"""CPU-only checks that need neither the dataset nor model weights: `pytest tests`."""
import subprocess
import sys

import pytest

from overlaybenchpytools.utils import calculate_overlap_iou, calculate_success_rate, extract_and_parse_json


def test_import_is_lazy():
    code = ("import sys, overlaybenchpytools; "
            "heavy = [m for m in ('torch', 'transformers', 'vllm', 'datasets') if m in sys.modules]; "
            "assert not heavy, heavy; print(overlaybenchpytools.__version__)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_help():
    out = subprocess.run([sys.executable, "-m", "overlaybenchpytools.cli", "-h"], check=True, capture_output=True,
                         text=True).stdout
    assert "--dp" in out and "--metrics" in out


GT = {"00000001": {
    "annotations": {"cat": {"category": "cat", "bbox": [0, 0, 100, 100]},
                    "dog": {"category": "dog", "bbox": [50, 50, 150, 150]}},
    "valid_overlap": [{"overlap_box": [0, 1]}],
    "relations": {"('cat', 'dog')": "next to"},
}}


def test_overlap_iou_perfect_and_missing():
    perfect = {"00000001": {"cat": [[0, 0, 100, 100]], "dog": [[50, 50, 150, 150]]}}
    assert calculate_overlap_iou(GT, perfect, 1024) == (1.0, 1.0)
    miou, omiou = calculate_overlap_iou(GT, {"00000001": {"cat": [[0, 0, 100, 100]]}}, 1024)
    assert miou == pytest.approx(0.5) and omiou == 0.0


def test_success_rate():
    pred = {"00000001": {"cat": "Yes", "dog": "no"}}
    assert calculate_success_rate(pred, GT, "Entity VQA") == 0.5
    assert calculate_success_rate({}, GT, "Relationship VQA") == 0.0


def test_parse_json():
    assert extract_and_parse_json('```json\n{"cat": "Yes"}\n```') == {"cat": "Yes"}
    assert extract_and_parse_json('{"cat": "No"}') == {"cat": "No"}
    assert extract_and_parse_json("not json") == {}


@pytest.mark.parametrize("resume", [False, True])
def test_results_json_keeps_old_numbers_only_on_resume(tmp_path, resume):
    import json
    from overlaybenchpytools.meter import OverLayBenchMeter
    out = tmp_path / "medium" / "0"
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps({"FID": 99.0}))  # left by an older run
    m = OverLayBenchMeter(root=str(tmp_path), save_dir=str(tmp_path), metrics=[], resume=resume)
    m.split, m.seed, m.save_dir = "medium", "0", str(out)
    m._save_results({"CLIP_global": 0.3})
    merged = m._save_results({"SR_E": 0.9})  # a later call of the same meter keeps the CLIP number
    assert ("FID" in merged) == resume and merged["CLIP_global"] == 0.3 and merged["SR_E"] == 0.9
