import torch

from stateaware_mil.data import deterministic_tile_indices


def test_tile_sampling_is_deterministic():
    a = deterministic_tile_indices("TCGA-AA-0001", 5000, fold=2, epoch=3, tile_cap=3000, seed=42)
    b = deterministic_tile_indices("TCGA-AA-0001", 5000, fold=2, epoch=3, tile_cap=3000, seed=42)
    assert torch.equal(a, b)
    assert len(a) == 3000


def test_no_sampling_when_bag_is_small():
    assert deterministic_tile_indices("TCGA-AA-0001", 1200, fold=0, epoch=1, tile_cap=3000, seed=42) is None
