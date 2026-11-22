import contextlib
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_OMEGACONF = importlib.util.find_spec("omegaconf") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class _DictAttr(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _to_dict_attr(obj):
    if isinstance(obj, dict):
        return _DictAttr({k: _to_dict_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dict_attr(v) for v in obj]
    return obj


class _NumpyList(list):
    def mean(self):
        return sum(self) / len(self) if self else 0.0

    def std(self, ddof=0):
        if len(self) <= ddof:
            return 0.0
        m = self.mean()
        return (sum((x - m) ** 2 for x in self) / (len(self) - ddof)) ** 0.5


def _install_missing_third_party_stubs():
    if not HAS_OMEGACONF and "omegaconf" not in sys.modules:
        oc = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def create(data):
                return _to_dict_attr(data)

            @staticmethod
            def load(path):
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                task = "test"
                for line in text.splitlines():
                    if line.startswith("task:"):
                        task = line.split(":", 1)[1].strip()
                        break
                return _to_dict_attr({
                    "task": task,
                    "data": {},
                    "model": {},
                    "tokenizer": {"pre_tokenizer": {}, "vqvae": {}},
                    "generation": {},
                    "eval": {},
                })

        oc.OmegaConf = _OmegaConf
        sys.modules["omegaconf"] = oc

    if not HAS_NUMPY and "numpy" not in sys.modules:
        np_mod = types.ModuleType("numpy")
        np_mod.mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        np_mod.array = lambda xs: _NumpyList(xs)
        np_mod.exp = lambda x: __import__("math").exp(x)
        sys.modules["numpy"] = np_mod

    if not HAS_TRANSFORMERS and "transformers" not in sys.modules:
        tf_mod = types.ModuleType("transformers")
        tf_mod.AutoTokenizer = MagicMock()
        tf_mod.AutoModelForCausalLM = MagicMock()
        sys.modules["transformers"] = tf_mod

    if not HAS_TORCH and "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _Device:
            def __init__(self, spec):
                s = str(spec)
                self.type = s.split(":", 1)[0]
                self.index = int(s.split(":", 1)[1]) if ":" in s else None
                self._spec = s

            def __str__(self):
                return self._spec

            def __repr__(self):
                return f"device(type={self.type!r})"

            def __eq__(self, other):
                return isinstance(other, _Device) and str(self) == str(other)

        class _Dtype:
            def __init__(self, name):
                self._name = name

            def __str__(self):
                return f"torch.{self._name}"

            def __repr__(self):
                return f"torch.{self._name}"

        class _NoGrad(contextlib.ContextDecorator):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Module:
            pass

        class _Conv1d(_Module):
            pass

        class _ModuleList(list):
            pass

        torch_mod.device = _Device
        torch_mod.bfloat16 = _Dtype("bfloat16")
        torch_mod.float16 = _Dtype("float16")
        torch_mod.float32 = _Dtype("float32")
        torch_mod.float64 = _Dtype("float64")
        torch_mod.long = _Dtype("int64")
        torch_mod.Tensor = object
        torch_mod.no_grad = _NoGrad
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        torch_mod.distributed = types.ModuleType("torch.distributed")
        torch_mod.nn = types.ModuleType("torch.nn")
        torch_mod.nn.Module = _Module
        torch_mod.nn.Conv1d = _Conv1d
        torch_mod.nn.ModuleList = _ModuleList
        torch_mod.nn.functional = types.ModuleType("torch.nn.functional")
        torch_mod.load = MagicMock()
        torch_mod.tensor = MagicMock()
        torch_mod.full = MagicMock()
        torch_mod.multinomial = MagicMock()
        torch_mod.arange = MagicMock()
        torch_mod.cat = MagicMock()
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = torch_mod.nn
        sys.modules["torch.nn.functional"] = torch_mod.nn.functional
        sys.modules["torch.distributed"] = torch_mod.distributed

    if "hydra" not in sys.modules and importlib.util.find_spec("hydra") is None:
        hydra_mod = types.ModuleType("hydra")
        hydra_mod.utils = types.SimpleNamespace(get_class=MagicMock())
        sys.modules["hydra"] = hydra_mod

    if "regex" not in sys.modules and importlib.util.find_spec("regex") is None:
        import re as _re
        sys.modules["regex"] = _re


def _make_xpu_compat_stub(
    backend_type="xpu",
    device_ids=None,
    resolved_device_str="xpu:0",
    fp64=False,
    ollama_notice=None,
):
    import torch

    stub = types.ModuleType("xpu_compat")
    ids = list(device_ids) if device_ids is not None else ["0"]
    resolved = torch.device(resolved_device_str)

    stub.resolve_device = MagicMock(side_effect=lambda preferred=None: torch.device(preferred) if preferred else resolved)
    stub.get_worker_device_ids = MagicMock(return_value=(backend_type, ids))

    def _cfg_env(base_env, btype, dev_id):
        env = dict(base_env)
        if btype == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = dev_id
            env.pop("NCM_DEVICE", None)
        elif btype == "xpu":
            env["ZE_AFFINITY_MASK"] = dev_id
            env.pop("NCM_DEVICE", None)
        else:
            env["NCM_DEVICE"] = "cpu"
        return env

    stub.configure_worker_env = MagicMock(side_effect=_cfg_env)
    stub.synchronize = MagicMock()
    stub.empty_cache = MagicMock()
    stub.supports_fp64 = MagicMock(return_value=fp64)
    stub.autocast_context = MagicMock(side_effect=lambda *args, **kwargs: contextlib.nullcontext())
    stub.configure_matmul_precision = MagicMock()
    stub.seed_everything = MagicMock()
    stub.get_vram_stats_mb = MagicMock(return_value={"allocated_mb": 128.0, "peak_allocated_mb": 256.5})
    stub.check_ollama_vram_notice = MagicMock(return_value=ollama_notice)
    return stub


def _import_with_xpu_stub(module_name: str, xpu_stub):
    had_prev = "xpu_compat" in sys.modules
    prev = sys.modules.get("xpu_compat")
    sys.modules["xpu_compat"] = xpu_stub
    try:
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)
    finally:
        if had_prev:
            sys.modules["xpu_compat"] = prev
        else:
            sys.modules.pop("xpu_compat", None)


_install_missing_third_party_stubs()


class TestEvaluateOrchestration(unittest.TestCase):
    def test_worker_planning_and_env_construction_xpu(self):
        stub = _make_xpu_compat_stub(
            backend_type="xpu",
            device_ids=["0", "1"],
            resolved_device_str="xpu:0",
            ollama_notice="[xpu] Ollama detected; 8.0 GB VRAM in use",
        )
        evaluate = _import_with_xpu_stub("evaluate", stub)
        with patch.dict(sys.modules, {"xpu_compat": stub}):
            backend, gpus = evaluate.resolve_worker_plan("", "")
            self.assertEqual(backend, "xpu")
            self.assertEqual(gpus, ["0", "1"])

            backend_cpu, gpus_cpu = evaluate.resolve_worker_plan("", "cpu")
            self.assertEqual(backend_cpu, "cpu")
            self.assertEqual(gpus_cpu, ["0"])

            env0 = evaluate.build_worker_env({"PATH": "/usr/bin"}, "xpu", "0")
            self.assertEqual(env0["ZE_AFFINITY_MASK"], "0")
            self.assertEqual(env0["TOKENIZERS_PARALLELISM"], "false")
            self.assertEqual(env0["NCM_ROOT"], evaluate.ROOT)

            env_dev = evaluate.build_worker_env({"PATH": "/usr/bin"}, "xpu", "1", device_arg="xpu:1")
            self.assertEqual(env_dev["ZE_AFFINITY_MASK"], "1")
            self.assertEqual(env_dev["NCM_DEVICE"], "xpu")

            merge_env = evaluate.build_merge_env({"PATH": "/usr/bin"}, "xpu", "0")
            self.assertEqual(merge_env["ZE_AFFINITY_MASK"], "0")
            self.assertEqual(merge_env["NCM_DEVICE"], "xpu")

    def test_evaluate_main_launches_shards_closes_logs_and_forwards_ncm_device(self):
        stub = _make_xpu_compat_stub(
            backend_type="xpu",
            device_ids=["0", "1"],
            resolved_device_str="xpu:0",
            ollama_notice="Ollama notice: 4.2 GB allocated",
        )
        evaluate = _import_with_xpu_stub("evaluate", stub)

        mock_proc_0 = MagicMock()
        mock_proc_0.wait.return_value = 0
        mock_proc_1 = MagicMock()
        mock_proc_1.wait.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "cfg.yaml")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write("task: dummy\n")
            out_dir = os.path.join(tmpdir, "out")

            fake_cfg = _DictAttr({"task": "dummy"})
            with (
                patch.dict(sys.modules, {"xpu_compat": stub}),
                patch.object(evaluate.OmegaConf, "load", return_value=fake_cfg),
                patch.object(evaluate.subprocess, "Popen", side_effect=[mock_proc_0, mock_proc_1]) as popen_mock,
                patch.object(evaluate.subprocess, "run") as run_mock,
                patch.object(sys, "argv", [
                    "evaluate.py",
                    "--config", cfg_path,
                    "--checkpoint-dir", tmpdir,
                    "--out", out_dir,
                    "--n", "16",
                    "--dtype", "bfloat16",
                    "--cfg", "1.0",
                ]),
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    evaluate.main()

            out_text = buf.getvalue()
            self.assertIn("Ollama notice: 4.2 GB allocated", out_text)
            self.assertIn("[evaluate] 2 shard(s) on XPU device(s) 0,1", out_text)

            self.assertEqual(popen_mock.call_count, 2)
            for idx, call in enumerate(popen_mock.call_args_list):
                cmd = call.args[0]
                kwargs = call.kwargs
                self.assertIn("--dtype", cmd)
                self.assertIn("bfloat16", cmd)
                self.assertIn("--cfg", cmd)
                self.assertIn("1.0", cmd)
                self.assertIn("--n", cmd)
                self.assertIn("16", cmd)
                self.assertEqual(kwargs["env"]["ZE_AFFINITY_MASK"], str(idx))
                self.assertEqual(kwargs["env"]["TOKENIZERS_PARALLELISM"], "false")
                self.assertEqual(kwargs["env"]["NCM_ROOT"], evaluate.ROOT)
                self.assertTrue(kwargs["stdout"].closed)

            run_mock.assert_called_once()
            merge_env = run_mock.call_args.kwargs["env"]
            self.assertEqual(merge_env["NCM_DEVICE"], "xpu")
            self.assertEqual(merge_env["ZE_AFFINITY_MASK"], "0")

    def test_evaluate_main_closes_logs_even_on_shard_failure(self):
        stub = _make_xpu_compat_stub(backend_type="cuda", device_ids=["0"], resolved_device_str="cuda:0")
        evaluate = _import_with_xpu_stub("evaluate", stub)

        bad_proc = MagicMock()
        bad_proc.wait.return_value = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "cfg.yaml")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write("task: dummy\n")
            out_dir = os.path.join(tmpdir, "out")
            fake_cfg = _DictAttr({"task": "dummy"})

            with (
                patch.dict(sys.modules, {"xpu_compat": stub}),
                patch.object(evaluate.OmegaConf, "load", return_value=fake_cfg),
                patch.object(evaluate.subprocess, "Popen", return_value=bad_proc) as popen_mock,
                patch.object(sys, "argv", [
                    "evaluate.py", "--config", cfg_path, "--checkpoint-dir", tmpdir, "--out", out_dir,
                ]),
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
                    evaluate.main()
            stub.check_ollama_vram_notice.assert_not_called()
            self.assertTrue(popen_mock.call_args.kwargs["stdout"].closed)


class TestEvalShardHelpers(unittest.TestCase):
    def test_resolve_dtype_fp64_fallback_and_autocast_dispatch(self):
        import torch

        stub_no_fp64 = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0", fp64=False)
        eval_shard = _import_with_xpu_stub("scripts.eval_shard", stub_no_fp64)

        with patch.dict(sys.modules, {"xpu_compat": stub_no_fp64}):
            self.assertEqual(eval_shard.resolve_dtype("auto", torch.device("xpu:0")), torch.bfloat16)
            self.assertEqual(eval_shard.resolve_dtype("auto", torch.device("cuda:0")), torch.bfloat16)
            self.assertEqual(eval_shard.resolve_dtype("auto", torch.device("cpu")), torch.float32)
            self.assertEqual(eval_shard.resolve_dtype("float32", torch.device("xpu:0")), torch.float32)
            self.assertEqual(eval_shard.resolve_dtype("float16", torch.device("xpu:0")), torch.float16)
            self.assertEqual(eval_shard.get_stat_dtype(torch.device("xpu:0")), torch.float32)

            fake_net = MagicMock()
            fake_tk = MagicMock()
            fake_tk.vqvae.vq.idx_to_zhats.return_value = (None, [MagicMock()])
            fake_tk.pre_tokenizer.decode.return_value = "decoded"
            with patch.object(eval_shard, "pick_tokens", return_value=MagicMock(tolist=lambda: [[1, 2]])):
                texts, ids = eval_shard.generate_batch(
                    fake_net,
                    fake_tk,
                    MagicMock(),
                    device=torch.device("xpu:0"),
                    dtype=torch.bfloat16,
                    eos=50256,
                    pad=50257,
                )
            self.assertEqual(texts, ["decoded"])
            self.assertEqual(ids, [[1, 2]])
            stub_no_fp64.autocast_context.assert_called_with(torch.device("xpu:0"), dtype=torch.bfloat16, enabled=True)

            with patch.object(eval_shard, "pick_tokens", return_value=MagicMock(tolist=lambda: [[1, 2]])):
                eval_shard.generate_batch(
                    fake_net,
                    fake_tk,
                    MagicMock(),
                    device=torch.device("xpu:0"),
                    dtype=torch.float32,
                    eos=50256,
                    pad=50257,
                )
            stub_no_fp64.autocast_context.assert_called_with(torch.device("xpu:0"), dtype=torch.float32, enabled=False)

        stub_fp64 = _make_xpu_compat_stub(backend_type="cuda", resolved_device_str="cuda:0", fp64=True)
        with patch.object(eval_shard, "xpu_compat", stub_fp64):
            self.assertEqual(eval_shard.get_stat_dtype(torch.device("cuda:0")), torch.float64)

    @unittest.skipUnless(HAS_TORCH, "requires real PyTorch")
    def test_load_allowance_and_pick_tokens_avoid_fp64_on_arc_a770(self):
        import torch

        stub_no_fp64 = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="cpu", fp64=False)
        eval_shard = _import_with_xpu_stub("scripts.eval_shard", stub_no_fp64)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "generator.pt")
            unigram = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
            torch.save({"inference": {"token_unigram": unigram}}, ckpt_path)

            with patch.dict(sys.modules, {"xpu_compat": stub_no_fp64}):
                allow = eval_shard.load_allowance(tmpdir, "generator.pt:inference.token_unigram", torch.device("cpu"))
                self.assertEqual(allow.dtype, torch.float32)

                logits = torch.tensor([[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]], dtype=torch.bfloat16)
                out = eval_shard.pick_tokens(
                    logits,
                    device=torch.device("cpu"),
                    freq=2.0,
                    mu=1.0,
                    allow=allow,
                    adj=[5.0, 4.0, 0.5],
                )
                self.assertEqual(out.shape, (1, 2))
                self.assertEqual(out.dtype, torch.long)


class TestEvalMergeAndGenPPL(unittest.TestCase):
    def test_exact_metric_scoring(self):
        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="cpu")
        eval_merge = _import_with_xpu_stub("scripts.eval_merge", stub)

        solved = "|".join([
            "534678912",
            "672195348",
            "198342567",
            "859761423",
            "426853791",
            "713924856",
            "961537284",
            "287419635",
            "345286179",
        ])
        bad = "1" * 81

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = eval_merge.exact([solved, bad], [solved, solved])

        self.assertEqual(stats["solved"], 1)
        self.assertEqual(stats["total"], 2)
        self.assertAlmostEqual(stats["solved_pct"], 50.0)
        self.assertAlmostEqual(stats["valid_pct"], 50.0)

    def test_eval_merge_load_ref_model_uses_bfloat16_on_xpu_and_fp32_on_cpu(self):
        import torch

        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        eval_merge = _import_with_xpu_stub("scripts.eval_merge", stub)

        fake_lm = MagicMock()
        fake_lm.to.return_value = fake_lm
        fake_lm.eval.return_value = fake_lm

        with (
            patch.dict(sys.modules, {"xpu_compat": stub}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_lm) as model_mock,
        ):
            eval_merge.load_ref_model("gpt2", torch.device("xpu:0"))
            self.assertEqual(model_mock.call_args.kwargs["torch_dtype"], torch.bfloat16)
            fake_lm.to.assert_called_with(torch.device("xpu:0"))

            eval_merge.load_ref_model("gpt2", torch.device("cpu"))
            self.assertEqual(model_mock.call_args.kwargs["torch_dtype"], torch.float32)
            fake_lm.to.assert_called_with(torch.device("cpu"))

    def test_eval_merge_passes_resolved_device_to_generation_perplexity(self):
        import torch

        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        eval_merge = _import_with_xpu_stub("scripts.eval_merge", stub)

        fake_lm = MagicMock()
        fake_lm.to.return_value = fake_lm
        fake_lm.eval.return_value = fake_lm
        fake_res = types.SimpleNamespace(mean_ppl=12.5)
        fake_scorer = MagicMock()
        fake_scorer.score.return_value = {
            k: types.SimpleNamespace(fmeasure=0.5) for k in ("rouge1", "rouge2", "rougeL")
        }
        fake_rouge_mod = types.ModuleType("rouge_score")
        fake_rouge_mod.rouge_scorer = types.SimpleNamespace(RougeScorer=MagicMock(return_value=fake_scorer))

        with (
            patch.dict(sys.modules, {"xpu_compat": stub, "rouge_score": fake_rouge_mod}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_lm),
            patch("evaluation.gen_ppl.generation_perplexity", return_value=fake_res) as gp_mock,
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                eval_merge.rouge(["hello world"], ["hello world"])
            self.assertEqual(gp_mock.call_args.kwargs["device"], "xpu:0")

            cfg = _DictAttr({"eval": _DictAttr({"ref_model": "gpt2-large"})})
            with contextlib.redirect_stdout(buf):
                eval_merge.genppl(cfg, [42], ["hello world"], [[1, 2]])
            self.assertEqual(gp_mock.call_args.kwargs["device"], "xpu:0")

    def test_eval_merge_genppl_flm_uses_resolved_device_without_cuda_hardcoding(self):
        import torch

        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        eval_merge = _import_with_xpu_stub("scripts.eval_merge", stub)

        fake_ids = MagicMock()
        fake_ids.shape = (1, 4)
        fake_ids.to.return_value = fake_ids
        fake_ids.__getitem__ = MagicMock(return_value=fake_ids)
        fake_ids.__eq__ = MagicMock(return_value=fake_ids)
        fake_ids.__ne__ = MagicMock(return_value=fake_ids)
        fake_ids.cumsum.return_value = fake_ids

        fake_tok = MagicMock()
        fake_tok.eos_token_id = 50256
        fake_tok.pad_token = None
        fake_tok.batch_decode.return_value = ["hello world"]
        fake_tok.return_value = {"input_ids": fake_ids, "attention_mask": fake_ids}

        fake_logits = MagicMock()
        fake_logits.float.return_value.transpose.return_value = fake_logits
        fake_lm = MagicMock()
        fake_lm.to.return_value = fake_lm
        fake_lm.eval.return_value = fake_lm
        fake_lm.return_value = [fake_logits]

        fake_nll = MagicMock()
        fake_nll.__mul__ = MagicMock(return_value=fake_nll)
        fake_nll.sum.return_value = 2.0
        fake_nll.dtype = torch.float32
        fake_ids.__add__ = MagicMock(return_value=fake_nll)
        fake_ids.to.return_value = fake_ids
        fake_ids.sum.return_value = 2.0

        cfg = _DictAttr({
            "tokenizer": _DictAttr({"pre_tokenizer": _DictAttr({"model_id": "bert-base-uncased"})}),
            "eval": _DictAttr({"ref_model": "gpt2-large"}),
        })

        buf = io.StringIO()
        with (
            patch.object(eval_merge, "xpu_compat", stub),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_lm) as model_mock,
            patch.object(torch.nn.functional, "cross_entropy", return_value=fake_nll, create=True),
            patch.object(torch, "special", types.SimpleNamespace(entr=MagicMock(return_value=MagicMock(sum=lambda: 1.0))), create=True),
            patch.object(torch, "unique", return_value=(None, MagicMock(float=lambda: MagicMock(__truediv__=lambda self, other: self))), create=True),
            patch.object(torch, "tensor", return_value=MagicMock(numel=lambda: 2), create=True),
            contextlib.redirect_stdout(buf),
        ):
            eval_merge.genppl_flm(cfg, [42], [[101, 102]])

        self.assertEqual(model_mock.call_args.kwargs["torch_dtype"], torch.bfloat16)
        fake_lm.to.assert_called_with(torch.device("xpu:0"))
        self.assertIn("genPPL", buf.getvalue())

    def test_gen_ppl_resolves_device_via_xpu_compat(self):
        import torch

        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="cpu")
        gen_ppl = _import_with_xpu_stub("evaluation.gen_ppl", stub)

        if HAS_TORCH:
            class _FakeBatch(dict):
                def to(self, dev):
                    return _FakeBatch({k: v.to(dev) for k, v in self.items()})

            class _FakeTok:
                pad_token_id = 0
                eos_token_id = 1
                is_fast = False

                def __call__(self, texts, **kwargs):
                    b = len(texts)
                    return _FakeBatch({
                        "input_ids": torch.tensor([[2, 3, 4]] * b, dtype=torch.long),
                        "attention_mask": torch.ones((b, 3), dtype=torch.long),
                    })

            class _FakeModel(torch.nn.Module):
                def forward(self, input_ids, attention_mask=None):
                    b, t = input_ids.shape
                    return types.SimpleNamespace(logits=torch.zeros((b, t, 8), dtype=torch.bfloat16))

            fake_model = _FakeModel()
            with (
                patch.object(gen_ppl, "xpu_compat", stub),
                patch.object(fake_model, "to", wraps=fake_model.to) as to_mock,
            ):
                res = gen_ppl.generation_perplexity(fake_model, _FakeTok(), ["hi"], ["there"])
            stub.resolve_device.assert_called_once_with()
            to_mock.assert_called_once_with(torch.device("cpu"))
            self.assertGreater(res.mean_ppl, 0.0)
        else:
            fake_model = MagicMock()
            fake_tok = MagicMock()
            fake_tok.pad_token_id = 0
            fake_tensor = MagicMock()
            fake_tensor.float.return_value = fake_tensor
            fake_tensor.sum.return_value.item.return_value = 0
            with (
                patch.object(gen_ppl, "xpu_compat", stub),
                patch.object(torch, "cat", return_value=fake_tensor, create=True),
                patch.object(torch, "clamp", return_value=fake_tensor, create=True),
                patch.object(torch, "exp", return_value=fake_tensor, create=True),
            ):
                res = gen_ppl.generation_perplexity(fake_model, fake_tok, [], [])
            stub.resolve_device.assert_called_once_with()
            fake_model.to.assert_called_once_with(torch.device("cpu"))
            self.assertEqual(res.mean_ppl, 1.0)


class TestGenerateCLI(unittest.TestCase):
    def test_arg_parsing_and_config_resolution(self):
        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        generate = _import_with_xpu_stub("scripts.generate", stub)

        args = generate.parse_args([
            "--task", "xsum",
            "--prompt", "News article text",
            "--num-samples", "2",
            "--seed", "7",
            "--dtype", "bfloat16",
            "--device", "xpu:0",
            "--warmup",
            "--download",
        ])
        self.assertEqual(args.task, "xsum")
        self.assertEqual(args.prompt, "News article text")
        self.assertEqual(args.num_samples, 2)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.device, "xpu:0")
        self.assertTrue(args.warmup)
        self.assertTrue(args.download)

        resolved_xsum = generate.resolve_config_path(None, "xsum", root=ROOT)
        self.assertEqual(resolved_xsum, os.path.join(ROOT, "config", "eval", "xsum.yaml"))

        resolved_lm1b = generate.resolve_config_path(None, "lm1b-small", root=ROOT)
        self.assertEqual(resolved_lm1b, os.path.join(ROOT, "config", "eval", "lm1b-small.yaml"))

        with self.assertRaises(ValueError):
            generate.resolve_config_path(None, None, root=ROOT)

    def test_hf_download_patterns_and_checkpoint_dir_resolution(self):
        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        generate = _import_with_xpu_stub("scripts.generate", stub)

        subpath, patterns = generate.get_hf_allow_patterns("lm1b-small")
        self.assertEqual(subpath, "lm1b/small")
        self.assertEqual(patterns, ["lm1b/tokenizer.pt", "lm1b/small/*"])

        subpath_s, patterns_s = generate.get_hf_allow_patterns("sudoku-hard")
        self.assertEqual(subpath_s, "sudoku/hard")
        self.assertEqual(patterns_s, ["sudoku/hard/*"])

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = MagicMock()
        cfg = _DictAttr({"task": "lm1b-small", "checkpoints": _DictAttr({"generator": "generator.pt"})})
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            ckpt_dir = generate.resolve_checkpoint_dir(cfg, "", download=True, root=tmpdir)
            self.assertEqual(ckpt_dir, os.path.join(tmpdir, "artifacts", "lm1b/small"))
            fake_hf.snapshot_download.assert_called_once_with(
                repo_id="hazyresearch/ncm-checkpoints",
                local_dir=os.path.join(tmpdir, "artifacts"),
                allow_patterns=["lm1b/tokenizer.pt", "lm1b/small/*"],
            )

    def test_sudoku_prompt_encoding_81_and_piped_and_vram_format(self):
        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        generate = _import_with_xpu_stub("scripts.generate", stub)
        from tokenization.sudoku_tokenizer import SudokuTokenizer

        pre_tok = SudokuTokenizer()
        tk = types.SimpleNamespace(pre_tokenizer=pre_tok)
        cfg = _DictAttr({"model": _DictAttr({"prefix_len": 91})})

        board_81 = "53..7...." + "6..195..." + ".98....6." + "8...6...3" + "4..8.3..1" + "7...2...6" + ".6....28." + "...419..5" + "....8..79"
        ids_81 = generate.encode_prompt(board_81, tk, cfg)
        self.assertEqual(len(ids_81), 91)
        self.assertEqual(ids_81[0], SudokuTokenizer.bos_token_id)
        self.assertEqual(ids_81[-1], SudokuTokenizer.row_separator_id)
        self.assertTrue(all(isinstance(i, int) for i in ids_81))

        board_piped = "|".join(board_81[i * 9:(i + 1) * 9] for i in range(9))
        ids_piped = generate.encode_prompt(board_piped, tk, cfg)
        self.assertEqual(ids_piped, ids_81)

        self.assertEqual(generate.format_vram_stats({}), "n/a")
        self.assertIn("256.5 MB", generate.format_vram_stats({"allocated_mb": 128.0, "peak_allocated_mb": 256.5}))

    def test_generate_main_unconditional_with_warmup_and_metrics(self):
        import torch

        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        generate = _import_with_xpu_stub("scripts.generate", stub)

        fake_cfg = _DictAttr({
            "task": "tinystories",
            "data": {},
            "checkpoints": _DictAttr({"generator": "generator.pt", "tokenizer": "tokenizer.pt"}),
            "model": _DictAttr({"prefix_len": 1}),
            "tokenizer": _DictAttr({"pre_tokenizer": _DictAttr({}), "vqvae": _DictAttr({})}),
            "generation": _DictAttr({"temperature": 1.0, "top_k": 0, "top_p": 1.0, "cfg_scale": 0.0, "level0_unigram": "generator.pt:inference.level0_unigram"}),
            "eval": _DictAttr({"split": "test"}),
        })
        fake_tk = MagicMock()
        fake_tk.pre_tokenizer.eos_token_id = 1
        fake_tk.pre_tokenizer.pad_token_id = 2
        fake_tk.vqvae.vq.levels = [1, 8, 16]
        fake_net = MagicMock()
        fake_l0_prior = MagicMock()
        fake_l0_prior.cpu.return_value = fake_l0_prior
        fake_sampled_l0 = MagicMock()
        fake_sampled_l0.view.return_value.to.return_value = MagicMock()

        with (
            patch.dict(sys.modules, {"xpu_compat": stub}),
            patch.object(generate.OmegaConf, "load", return_value=fake_cfg),
            patch.object(generate, "load_models", return_value=(fake_tk, fake_net)),
            patch.object(generate, "load_allowance", return_value=None),
            patch.object(generate, "build_l0_prior", return_value=fake_l0_prior),
            patch.object(generate, "generate_batch", return_value=(["Once upon a time"], [[10, 11]])) as gen_mock,
            patch.object(torch, "multinomial", return_value=fake_sampled_l0),
            patch.object(torch, "full", return_value=MagicMock()),
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                generate.main([
                    "--task", "tinystories", "--checkpoint-dir", "/tmp/ts", "--warmup", "--seed", "42",
                ])

        out_text = buf.getvalue()
        self.assertEqual(gen_mock.call_count, 2)
        stub.seed_everything.assert_called_with(42)
        self.assertGreaterEqual(stub.synchronize.call_count, 2)
        stub.get_vram_stats_mb.assert_called_once()
        self.assertIn("Once upon a time", out_text)
        self.assertIn("256.5 MB", out_text)
        self.assertIn("latency=", out_text)

    def test_generate_main_conditional_prompt_validation_and_execution(self):
        import torch

        stub = _make_xpu_compat_stub(backend_type="xpu", resolved_device_str="xpu:0")
        generate = _import_with_xpu_stub("scripts.generate", stub)

        fake_cfg = _DictAttr({
            "task": "xsum",
            "data": _DictAttr({"test": "passages.pickle"}),
            "checkpoints": _DictAttr({"generator": "generator.pt", "tokenizer": "tokenizer.pt"}),
            "model": _DictAttr({"prefix_len": 8}),
            "tokenizer": _DictAttr({"pre_tokenizer": _DictAttr({}), "vqvae": _DictAttr({})}),
            "generation": _DictAttr({"temperature": 1.0, "top_k": 1, "top_p": 0.0, "cfg_scale": 0.0}),
            "eval": _DictAttr({"split": "test"}),
        })
        fake_tk = MagicMock()
        fake_tk.pre_tokenizer.eos_token_id = 50256
        fake_tk.pre_tokenizer.pad_token_id = 50257
        fake_tk.pre_tokenizer.encode.return_value = [10, 20, 30]
        fake_net = MagicMock()

        with (
            patch.dict(sys.modules, {"xpu_compat": stub}),
            patch.object(generate.OmegaConf, "load", return_value=fake_cfg),
            patch.object(generate, "load_models", return_value=(fake_tk, fake_net)),
            patch.object(generate, "load_allowance", return_value=None),
            patch.object(generate, "build_l0_prior", return_value=None),
            patch.object(generate, "generate_batch", return_value=(["Short summary."], [[1, 2]])),
            patch.object(torch, "tensor", return_value=MagicMock()) as tensor_mock,
        ):
            with self.assertRaises(SystemExit):
                generate.main(["--task", "xsum", "--checkpoint-dir", "/tmp/xsum"])

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                generate.main(["--task", "xsum", "--checkpoint-dir", "/tmp/xsum", "--prompt", "Long article"])
            tensor_mock.assert_called_once()
            self.assertEqual(tensor_mock.call_args.args[0], [[10, 20, 30, 50257, 50257, 50257, 50257, 50257]])
            self.assertIn("Short summary.", buf.getvalue())


class TestXpuCompatShutdown(unittest.TestCase):
    def test_shutdown_device_cleanly_runs_gc_sync_empty_cache_and_flushes(self):
        import xpu_compat

        events: list[str] = []
        with (
            patch.object(xpu_compat.gc, "collect", side_effect=lambda: events.append("gc") or 0),
            patch.object(xpu_compat, "synchronize", side_effect=lambda dev=None: events.append(f"sync:{dev}")),
            patch.object(xpu_compat, "empty_cache", side_effect=lambda dev=None: events.append(f"empty:{dev}")),
            patch.object(sys.stdout, "flush", side_effect=lambda: events.append("stdout")),
            patch.object(sys.stderr, "flush", side_effect=lambda: events.append("stderr")),
        ):
            xpu_compat.shutdown_device_cleanly("cpu")

        self.assertEqual(events, ["gc", "sync:cpu", "empty:cpu", "stdout", "stderr"])

    def test_shutdown_device_cleanly_swallows_sync_and_empty_cache_errors(self):
        import xpu_compat

        flushed: list[str] = []
        with (
            patch.object(xpu_compat.gc, "collect", return_value=0) as gc_mock,
            patch.object(xpu_compat, "synchronize", side_effect=RuntimeError("sync failed")) as sync_mock,
            patch.object(xpu_compat, "empty_cache", side_effect=RuntimeError("cache failed")) as empty_mock,
            patch.object(sys.stdout, "flush", side_effect=lambda: flushed.append("out")),
            patch.object(sys.stderr, "flush", side_effect=lambda: flushed.append("err")),
        ):
            xpu_compat.shutdown_device_cleanly("xpu:0")

        gc_mock.assert_called_once()
        sync_mock.assert_called_once_with("xpu:0")
        empty_mock.assert_called_once_with("xpu:0")
        self.assertEqual(flushed, ["out", "err"])

    def test_shutdown_device_cleanly_cpu_real(self):
        import xpu_compat

        xpu_compat.shutdown_device_cleanly("cpu")
        xpu_compat.shutdown_device_cleanly(None)


if __name__ == "__main__":
    unittest.main()
