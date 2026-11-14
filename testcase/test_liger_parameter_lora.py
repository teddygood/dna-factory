import logging

import pytest
from trl import ModelConfig, SFTConfig

from dna_factory.training_runner import disable_liger_swiglu_for_parameter_lora

PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
log = logging.getLogger(__name__)


def make(liger=True, peft=True, params=PARAMS, config=None):
    training_args = SFTConfig(
        output_dir="out", use_liger_kernel=liger, liger_kernel_config=config
    )
    model_args = ModelConfig(use_peft=peft, lora_target_parameters=params)
    return training_args, model_args


def test_sets_swiglu_false_and_keeps_other_keys():
    args, margs = make(config={"rope": True})
    disable_liger_swiglu_for_parameter_lora(args, margs, log)
    assert args.liger_kernel_config == {"rope": True, "swiglu": False}


def test_sets_swiglu_false_without_existing_config():
    args, margs = make()
    disable_liger_swiglu_for_parameter_lora(args, margs, log)
    assert args.liger_kernel_config == {"swiglu": False}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"liger": False},
        {"peft": False},
        {"params": None},
        {"params": []},
    ],
)
def test_noop(kwargs):
    args, margs = make(**kwargs)
    before = args.liger_kernel_config
    disable_liger_swiglu_for_parameter_lora(args, margs, log)
    assert args.liger_kernel_config == before


def test_explicit_swiglu_true_raises():
    args, margs = make(config={"swiglu": True})
    with pytest.raises(ValueError, match="swiglu"):
        disable_liger_swiglu_for_parameter_lora(args, margs, log)
