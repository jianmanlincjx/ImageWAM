"""A2: Stage 1 null-image tokens.

The tokens are injected as the *reference* image because the mixed-attention
mask lets text attend txt+ref but not target; putting them in the target slot
would leave the text stream unchanged and the fix inert.
"""
import torch

from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.models.backbones.goal_pose_prior import STAGE1_ONLY_CHECKPOINT_PREFIXES


def test_grid_is_squarest_factorisation():
    assert ImageWAM._grid_for(32) == (4, 8)
    assert ImageWAM._grid_for(16) == (4, 4)
    assert ImageWAM._grid_for(64) == (8, 8)
    assert ImageWAM._grid_for(12) == (3, 4)
    h, w = ImageWAM._grid_for(37)  # prime -> degenerate but still valid
    assert h * w == 37


def test_null_tokens_are_stage1_only_for_checkpointing():
    assert any("stage1_null_image_tokens" in p for p in STAGE1_ONLY_CHECKPOINT_PREFIXES)


def test_null_stream_is_none_when_disabled():
    class Stub:
        stage1_null_image_tokens = None
        _grid_for = staticmethod(ImageWAM._grid_for)
        _stage1_null_image_stream = ImageWAM._stage1_null_image_stream

    t, i = Stub()._stage1_null_image_stream(4, torch.zeros(1, 1, 8))
    assert t is None and i is None


def test_null_stream_shapes_and_ids():
    class Stub:
        _grid_for = staticmethod(ImageWAM._grid_for)
        _stage1_null_image_stream = ImageWAM._stage1_null_image_stream

    s = Stub()
    s.stage1_null_image_tokens = torch.nn.Parameter(torch.randn(32, 128))
    like = torch.zeros(1, 1, 8)
    tokens, ids = s._stage1_null_image_stream(5, like)
    assert tokens.shape == (5, 32, 128)
    assert ids.shape == (5, 32, 4)
    # ref slot: time value 10.0, same as a real reference image
    assert torch.all(ids[..., 0] == 10.0)
    # a 4x8 grid of positions, not all-zero
    assert int(ids[0, :, 1].max()) == 3
    assert int(ids[0, :, 2].max()) == 7
    # every sample sees the same constant stand-in
    assert torch.equal(tokens[0], tokens[4])
