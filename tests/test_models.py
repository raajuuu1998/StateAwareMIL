import torch

from stateaware_mil.ablations import StateAwareAblation
from stateaware_mil.baselines import DirectJoint, IndependentPair, NaiveMTL
from stateaware_mil.fourstate_abmil import FourStateABMIL
from stateaware_mil.state_aware import StateAwareInteractionMIL


def test_main_model_shapes():
    x = torch.randn(13, 32)
    direct = DirectJoint(32)
    pair = IndependentPair(32)
    mtl = NaiveMTL(32)
    state = StateAwareInteractionMIL(32)
    four = FourStateABMIL(32)

    assert direct(x)["joint_logit"].ndim == 0
    assert pair(x)["a_logit"].ndim == 0
    assert pair(x)["b_logit"].ndim == 0
    assert mtl(x)["joint_logit"].ndim == 0
    assert state(x)["state_logits"].shape == (4,)
    assert four(x).shape == (4,)


def test_state_reference_logit_is_zero():
    x = torch.randn(11, 24)
    model = StateAwareInteractionMIL(24)
    out = model(x)
    assert torch.equal(out["state_logits"][0], torch.zeros_like(out["state_logits"][0]))


def test_no_interaction_is_additive():
    x = torch.randn(9, 24)
    model = StateAwareAblation(24, "no_interaction")
    out = model(x)
    assert torch.allclose(out["state_logits"][3], out["a_logit"] + out["b_logit"])
    assert torch.equal(out["interaction_logit"], torch.zeros_like(out["interaction_logit"]))
