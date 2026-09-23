import unittest

import requests

from sglang.srt.utils import is_hip, is_sm90_supported
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.server_fixtures.standalone_fixture import StandaloneServerBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    CustomTestCase,
)

# Non-V2 standalone speculative decoding tests (FA3, Triton, FlashInfer
# backends). Sibling V2 classes stay per-commit in test_spec_standalone.py.
register_cuda_ci(est_time=345, stage="extra-a", runner_config="1-gpu-large")
# AMD: fa3 / flashinfer attention backends are not built in the ROCm
# sgl_kernel, so only the triton-backend class runs on ROCm (the fa3 and
# flashinfer classes are skipped on ROCm below).
register_amd_ci(est_time=103, suite="extra-a-test-1-gpu-large-amd")

_AMD_SKIP_BACKEND = "fa3 / flashinfer attention backends are CUDA-only (not in the ROCm sgl_kernel build)"


@unittest.skipIf(is_hip(), _AMD_SKIP_BACKEND)
class TestStandaloneSpeculativeDecodingBase(StandaloneServerBase, CustomTestCase):
    attention_backend = "fa3"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    disable_overlap = True


class TestStandaloneSpeculativeDecodingTriton(StandaloneServerBase, CustomTestCase):
    attention_backend = "triton"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    disable_overlap = True
    enable_deterministic_inference = True


@unittest.skipIf(is_hip(), _AMD_SKIP_BACKEND)
class TestStandaloneSpeculativeDecodingFlashinfer(StandaloneServerBase, CustomTestCase):
    attention_backend = "flashinfer"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    disable_overlap = True


@unittest.skipIf(
    is_hip() or not is_sm90_supported(),
    "the MXFP4 Marlin and FlashInfer MoE pair is SM90-only here",
)
class TestStandaloneDraftMoeRunnerBackend(StandaloneServerBase, CustomTestCase):
    """The draft is built with --speculative-moe-runner-backend while the
    target keeps --moe-runner-backend; building it with the target's backend
    fails the first draft forward on a TopK format assert."""

    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    draft_model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    attention_backend = "fa3"
    speculative_num_steps = 3
    speculative_num_draft_tokens = 4

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args[args.index(DEFAULT_DRAFT_MODEL_STANDALONE)] = cls.draft_model
        return args + [
            "--moe-runner-backend",
            "marlin",
            "--speculative-moe-runner-backend",
            "flashinfer_mxfp4",
        ]

    @unittest.skip("accuracy is not what this class guards")
    def test_gsm8k(self):
        pass

    def test_serves_with_split_moe_backends(self):
        res = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        res.raise_for_status()
        self.assertTrue(res.json()["text"])
        # Per-request meta_info: /server_info omits avg_spec_accept_length until
        # a decode-log interval has elapsed, which one short request never reaches.
        meta = res.json()["meta_info"]
        self.assertGreater(meta["spec_verify_ct"], 0)
        # 1.0 is the bonus token alone; a self-draft that accepts nothing is broken.
        self.assertGreater(meta["completion_tokens"] / meta["spec_verify_ct"], 1.0)


if __name__ == "__main__":
    unittest.main()
