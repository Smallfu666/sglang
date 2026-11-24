import math
import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import mxfp4 as mxfp4_module
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MXFP4_BLOCK = 32
GPT_OSS_INTERMEDIATE = 2880
GPT_OSS_HIDDEN = 2880


def loader_rank_intermediate(intermediate_size, tp_size):
    # Mirrors GptOssForCausalLM._load_mxfp4_experts_weights.
    blocks = intermediate_size // MXFP4_BLOCK
    return math.ceil(blocks / tp_size) * MXFP4_BLOCK


class TestMxfp4CutlassTpShard(CustomTestCase):
    def _create_weights(self, fi_kernel, tp_size):
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = False
        method.use_aiter_moe = False
        method.use_flashinfer = True
        method._fi_kernel = fi_kernel
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        platform = types.SimpleNamespace(is_sm100=False)
        with patch.object(mxfp4_module, "get_platform", lambda: platform):
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=GPT_OSS_HIDDEN,
                intermediate_size_per_partition=GPT_OSS_INTERMEDIATE // tp_size,
                params_dtype=torch.bfloat16,
                with_bias=True,
            )
        return layer

    def test_buffers_hold_the_whole_block_shard(self):
        for fi_kernel in ("cutlass_sm90", "cutlass_sm120"):
            for tp_size in (1, 2, 4, 8):
                with self.subTest(fi_kernel=fi_kernel, tp_size=tp_size):
                    layer = self._create_weights(fi_kernel, tp_size)
                    rows = loader_rank_intermediate(GPT_OSS_INTERMEDIATE, tp_size)
                    self.assertGreaterEqual(layer.w2_weight.shape[2], rows // 2)
                    self.assertGreaterEqual(
                        layer.w2_weight_scale.shape[2], rows // MXFP4_BLOCK
                    )
                    self.assertGreaterEqual(layer.w13_weight.shape[1], 2 * rows)
                    self.assertGreaterEqual(layer.w13_weight_bias.shape[1], 2 * rows)

    def test_post_load_padding_covers_the_buffer(self):
        for tp_size in (1, 2, 4, 8):
            with self.subTest(tp_size=tp_size):
                layer = self._create_weights("cutlass_sm90", tp_size)
                buffer_rows = layer.w13_weight.shape[1] // 2
                padded = mxfp4_module.round_up(GPT_OSS_INTERMEDIATE // tp_size, 128)
                self.assertLessEqual(buffer_rows, padded)


if __name__ == "__main__":
    sys.exit(unittest.main())
