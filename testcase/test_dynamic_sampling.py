"""Test cases for dna_factory/dynamic_sampling.py (sampling_mask handling, refusals, regroup)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory import dynamic_sampling as ds


def _batch(rows, prompt_len, comp_len, k, dead):
    return {
        "prompt_ids": torch.ones(rows, prompt_len, dtype=torch.long),
        "prompt_mask": torch.ones(rows, prompt_len, dtype=torch.long),
        "completion_ids": torch.ones(rows, comp_len, dtype=torch.long),
        "completion_mask": torch.ones(rows, comp_len, dtype=torch.long),
        "sampling_mask": torch.arange(k).repeat(rows, comp_len, 1),
        "advantages": torch.zeros(rows),
        ds.DEAD_KEY: torch.full((rows,), dead, dtype=torch.bool),
    }


class TestSamplingMask:
    def test_truncate_keeps_trl_slice_assignment_valid(self):
        inputs = _batch(2, 5, 8, 3, dead=True)
        out, did = ds.truncate_if_all_dead(inputs)
        assert did
        keep = out["completion_ids"].shape[1]
        assert out["sampling_mask"].shape == (2, keep, 3)
        # TRL: support_ids[:, -logits_to_keep:] = sampling_mask[start:end]
        support_ids = torch.full((2, out["prompt_ids"].shape[1] + keep, 3), -1)
        support_ids[:, -keep:] = out["sampling_mask"]

    def test_concat_batches_pads_t_and_k_with_minus_one(self):
        a = _batch(2, 4, 6, 3, dead=False)
        b = _batch(1, 4, 8, 5, dead=False)
        out = ds.concat_batches(a, b)
        sm = out["sampling_mask"]
        assert sm.shape == (3, 8, 5)
        assert (sm[:2, 6:] == -1).all() and (sm[:2, :, 3:] == -1).all()
        assert (sm[:2, :6, :3] == a["sampling_mask"]).all()
        assert (sm[2] == b["sampling_mask"][0]).all()
        # shorter second batch, narrower K
        out2 = ds.concat_batches(b, a)
        assert out2["sampling_mask"].shape == (3, 8, 5)
        assert (out2["sampling_mask"][1:, 6:] == -1).all()


class _FakeBase:
    def __init__(self, args, loss_type="grpo", **attrs):
        self.args = args
        self.loss_type = loss_type
        self.num_generations = 2
        self.train_dataset = None
        for k, v in attrs.items():
            setattr(self, k, v)


class _Trainer(ds.DynamicSamplingMixin, _FakeBase):
    pass


def _args(**kw):
    base = {
        "beta": 0.0,
        "top_entropy_quantile": 1.0,
        "per_device_train_batch_size": 2,
        "steps_per_generation": 2,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class TestInitGuards:
    def test_mask_refuses_top_entropy_quantile(self):
        with pytest.raises(ValueError, match="top_entropy_quantile"):
            _Trainer(_args(top_entropy_quantile=0.2), dynamic_sampling="mask")

    def test_resample_disables_truncation_for_top_entropy_quantile(self):
        t = _Trainer(_args(top_entropy_quantile=0.2), dynamic_sampling="resample")
        assert t._truncation_is_lossless is False

    def test_default_quantile_is_accepted(self):
        t = _Trainer(_args(), dynamic_sampling="mask")
        assert t._truncation_is_lossless is True


class TestRegroupLossType:
    def test_bnpo_skips_regroup(self):
        t = _Trainer(_args(), loss_type="bnpo", dynamic_sampling="mask")
        assert t._regroup_is_neutral is False

    @pytest.mark.parametrize(
        "loss_type", ["grpo", "sapo", "dr_grpo", "luspo", "dapo", "cispo", "vespo"]
    )
    def test_other_loss_types_regroup(self, loss_type):
        t = _Trainer(_args(), loss_type=loss_type, dynamic_sampling="mask")
        assert t._regroup_is_neutral is True
