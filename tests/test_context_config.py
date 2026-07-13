import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_context_config.py"
SPEC = importlib.util.spec_from_file_location("check_context_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def values(per_slot="131072", parallel="5", total="655360"):
    return {
        "LLM_CTX_PER_SLOT": per_slot,
        "LLM_PARALLEL": parallel,
        "LLM_CTX_TOTAL": total,
    }


def test_valid_noob_profile_reserves_parent_and_four_children():
    assert MODULE.validate(values(), 4, 131072) == (131072, 5, 655360)


def test_rejects_bad_math_missing_parent_capacity_and_context_drift():
    with pytest.raises(ValueError, match="expected"):
        MODULE.validate(values(total="655359"), 4, 131072)
    with pytest.raises(ValueError, match="reserve one parent"):
        MODULE.validate(values(parallel="4", total="524288"), 4, 131072)
    with pytest.raises(ValueError, match="align them"):
        MODULE.validate(values(per_slot="65536", total="327680"), 4, 131072)


def test_live_slot_shape_must_match_count_and_per_slot_context(tmp_path):
    good = tmp_path / "slots.json"
    good.write_text(json.dumps([{"n_ctx": 131072} for _ in range(5)]))
    MODULE.validate_slots(str(good), 131072, 5)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"n_ctx": 131072}, {"n_ctx": 65536}]))
    with pytest.raises(ValueError, match="reports 2 slots"):
        MODULE.validate_slots(str(bad), 131072, 5)
