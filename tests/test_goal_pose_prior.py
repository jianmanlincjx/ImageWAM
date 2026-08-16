#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from imagewam.models.backbones.goal_pose_prior import (
    GOAL_PRIOR_LATENT_DIM,
    GOAL_PRIOR_NUM_GOAL_TOKENS,
    GOAL_PRIOR_NUM_GROUPS,
    GOAL_PRIOR_NUM_LATENTS,
    GOAL_PRIOR_NUM_POSE_TOKENS,
    GoalPoseDecoder,
    GoalPoseEncoder,
    SemanticVisualAggregator,
    build_stage2_action_attention_mask,
    compute_pose_reconstruction_loss,
    extract_goal_pose_from_proprio,
    layer_group_index,
    stage2_action_mask_excludes_raw_images,
    validate_goal_prior_checkpoint_keys,
)
from imagewam.models.backbones.imagewam import ImageWAM


class DummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4)
        self.flux2_lora_enabled = False


class DummyMoT(nn.Module):
    def __init__(self, video: nn.Module, action: nn.Module):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})


class GoalPosePriorTests(unittest.TestCase):
    def test_goal_encoder_production_shape(self):
        encoder = GoalPoseEncoder(pose_dim=8, num_tokens=GOAL_PRIOR_NUM_GOAL_TOKENS, hidden_size=3072)
        tokens = encoder(torch.randn(2, 8))
        self.assertEqual(tuple(tokens.shape), (2, 8, 3072))

    def test_aggregator_production_query_shape(self):
        aggregator = SemanticVisualAggregator(
            num_tokens=GOAL_PRIOR_NUM_LATENTS,
            latent_dim=GOAL_PRIOR_LATENT_DIM,
            context_dim=64,
            kv_dim=64,
            attn_head_dim=8,
            num_layer_groups=GOAL_PRIOR_NUM_GROUPS,
            num_heads=4,
        )
        queries = aggregator.initial_queries(3)
        self.assertEqual(tuple(queries.shape), (3, 100, 768))
        semantic = torch.randn(3, 5, 64)
        visual = torch.randn(3, 7, 64)
        semantic_mask = torch.ones(3, 5, dtype=torch.bool)
        image_mask = torch.ones(3, 7, dtype=torch.bool)
        image_mask[:, -2:] = False
        out = aggregator.forward_layer(
            queries,
            semantic,
            visual,
            semantic_mask=semantic_mask,
            image_mask=image_mask,
            layer_idx=7,
            num_layers=25,
        )
        self.assertEqual(tuple(out.shape), (3, 100, 768))
        key, value = aggregator.project_kv(out, layer_idx=7, num_layers=25)
        self.assertEqual(tuple(key.shape), (3, 100, 64))
        self.assertEqual(tuple(value.shape), (3, 100, 64))

    def test_pose_decoder_production_shape(self):
        decoder = GoalPoseDecoder(
            num_tokens=GOAL_PRIOR_NUM_POSE_TOKENS,
            hidden_size=GOAL_PRIOR_LATENT_DIM,
            pose_dim=8,
        )
        pose = decoder(torch.randn(2, 8, 768))
        self.assertEqual(tuple(pose.shape), (2, 8))

    def test_layer_group_index_covers_25_layers(self):
        groups = [layer_group_index(idx, 25, 5) for idx in range(25)]
        self.assertEqual(groups[:5], [0, 0, 0, 0, 0])
        self.assertEqual(groups[5:10], [1, 1, 1, 1, 1])
        self.assertEqual(groups[10:15], [2, 2, 2, 2, 2])
        self.assertEqual(groups[15:20], [3, 3, 3, 3, 3])
        self.assertEqual(groups[20:], [4, 4, 4, 4, 4])
        self.assertEqual(set(groups), {0, 1, 2, 3, 4})

    def test_pose_loss_respects_pad_masks(self):
        pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.zeros_like(pred)
        dim_is_pad = torch.tensor([[False, False, True], [False, False, True]])
        is_pad = torch.tensor([False, True])
        loss = compute_pose_reconstruction_loss(pred, target, is_pad=is_pad, dim_is_pad=dim_is_pad)
        expected = ((1.0**2) + (2.0**2)) / 2.0
        self.assertTrue(torch.allclose(loss, torch.tensor(expected)))

    def test_extract_goal_pose_from_proprio(self):
        proprio = torch.arange(17 * 8, dtype=torch.float32).view(17, 8)
        proprio_is_pad = torch.zeros(17, dtype=torch.bool)
        proprio_is_pad[-1] = True
        goal, goal_is_pad = extract_goal_pose_from_proprio(proprio, proprio_is_pad)
        self.assertTrue(torch.equal(goal, proprio[-1]))
        self.assertTrue(bool(goal_is_pad.item()))

    def test_stage2_action_mask_layout_has_no_raw_image_columns(self):
        txt_len, synthetic_len, action_len = 12, 100, 16
        mask = build_stage2_action_attention_mask(
            batch_size=2,
            txt_len=txt_len,
            synthetic_len=synthetic_len,
            action_len=action_len,
            device=torch.device("cpu"),
            text_attention_mask=torch.tensor(
                [[True] * 10 + [False, False], [True] * 12],
                dtype=torch.bool,
            ),
        )
        self.assertEqual(tuple(mask.shape), (2, action_len, txt_len + synthetic_len + action_len))
        self.assertTrue(
            stage2_action_mask_excludes_raw_images(
                mask,
                txt_len=txt_len,
                synthetic_len=synthetic_len,
                action_len=action_len,
            )
        )
        self.assertFalse(bool(mask[0, 0, 10].item()))
        self.assertFalse(bool(mask[0, 0, 11].item()))
        self.assertTrue(bool(mask[1, 0, 11].item()))
        self.assertTrue(torch.all(mask[:, :, txt_len : txt_len + synthetic_len]).item())
        self.assertTrue(torch.all(mask[:, :, txt_len + synthetic_len :]).item())

    def test_checkpoint_bridge_allows_stage1_only_and_stage2_only_keys(self):
        validate_goal_prior_checkpoint_keys(
            current_stage="stage2",
            payload_stage="stage1",
            missing_keys=[
                "semantic_visual_aggregator.queries",
                "semantic_visual_pose_norm.weight",
                "semantic_visual_pose_decoder.net.0.weight",
            ],
            unexpected_keys=["goal_pose_encoder.net.0.weight"],
            bridge_from_stage1=True,
        )
        with self.assertRaises(RuntimeError):
            validate_goal_prior_checkpoint_keys(
                current_stage="stage2",
                payload_stage="stage1",
                missing_keys=["mot.mixtures.action.double_blocks.0.qkv.weight"],
                unexpected_keys=[],
                bridge_from_stage1=True,
            )
        with self.assertRaises(RuntimeError):
            validate_goal_prior_checkpoint_keys(
                current_stage="stage2",
                payload_stage="stage2",
                missing_keys=["semantic_visual_aggregator.queries"],
                unexpected_keys=[],
                bridge_from_stage1=False,
            )

    def test_stage1_trainable_policy_freezes_video_and_keeps_goal_encoder(self):
        video = DummyExpert()
        action = DummyExpert()
        model = ImageWAM.__new__(ImageWAM)
        nn.Module.__init__(model)
        model.stack = "flux2"
        model.mot = DummyMoT(video, action)
        model.dit = model.mot
        model.goal_prior_stage = "stage1"
        model.goal_pose_encoder = GoalPoseEncoder(pose_dim=8, num_tokens=8, hidden_size=16, inner_dim=8)
        model.semantic_visual_aggregator = None
        model.semantic_visual_pose_norm = None
        model.semantic_visual_pose_decoder = None
        model.proprio_encoder = nn.Linear(8, 16)

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        ImageWAM.apply_trainable_policy(model)
        model.proprio_encoder.train()
        model.proprio_encoder.requires_grad_(True)

        self.assertFalse(any(param.requires_grad for param in video.parameters()))
        self.assertTrue(all(param.requires_grad for param in action.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.goal_pose_encoder.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.proprio_encoder.parameters()))
        trainable = ImageWAM.collect_trainable_parameters(model)
        trainable_ids = {id(param) for param in trainable}
        self.assertTrue(all(id(param) in trainable_ids for param in action.parameters()))
        self.assertTrue(all(id(param) in trainable_ids for param in model.goal_pose_encoder.parameters()))
        self.assertTrue(all(id(param) in trainable_ids for param in model.proprio_encoder.parameters()))
        self.assertFalse(any(id(param) in trainable_ids for param in video.parameters()))

    def test_flux2_layer_checkpoint_keeps_input_grads(self):
        from imagewam.models.backbones.mot import MoT

        class _Flags:
            mot_checkpoint_mixed_attn = True
            training = True

        frozen = torch.randn(4, 4)
        tokens = torch.randn(2, 4, requires_grad=True)

        def _layer(x):
            return x @ frozen

        out = MoT._flux2_maybe_checkpoint(_Flags(), _layer, tokens)
        out.sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNone(frozen.grad)

        flags_eval = _Flags()
        flags_eval.training = False
        tokens2 = torch.randn(2, 4, requires_grad=True)
        out2 = MoT._flux2_maybe_checkpoint(flags_eval, _layer, tokens2)
        out2.sum().backward()
        self.assertIsNotNone(tokens2.grad)

    def test_stage2_requires_stage1_checkpoint_unless_resuming(self):
        from imagewam.trainer import Wan22Trainer

        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.resume = None
        trainer.stage1_checkpoint = None
        trainer.model = type("M", (), {"goal_prior_stage": "stage2"})()
        with self.assertRaisesRegex(ValueError, "requires `stage1_checkpoint`"):
            Wan22Trainer._load_stage1_bridge_checkpoint_before_prepare(trainer)

        trainer.model.goal_prior_stage = "stage1"
        Wan22Trainer._load_stage1_bridge_checkpoint_before_prepare(trainer)

        trainer.model.goal_prior_stage = "stage2"
        trainer.resume = "/tmp/stage2_resume"
        Wan22Trainer._load_stage1_bridge_checkpoint_before_prepare(trainer)


if __name__ == "__main__":
    unittest.main()
