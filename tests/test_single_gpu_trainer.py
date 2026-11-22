from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xpu_compat  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from scripts import preencode_concepts, train_ncm, train_vqvae  # noqa: E402


def _make_mini_config(
    cfg_path: Path,
    *,
    prefix_len: int = 16,
    target_len: int = 8,
    levels: list[int] | None = None,
    grad_ckpt: bool = True,
    optimizer: str = "adamw",
) -> Path:
    levels = levels or [4, 8]
    vocab_n = 32
    in_vocab = 64
    cfg = OmegaConf.create(
        {
            "task": "ncm",
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "bias": True,
                "dropout": 0.0,
                "n_embd": 32,
                "context_length": prefix_len + sum(levels),
                "prefix_len": prefix_len,
                "raw_token_prefix": True,
                "prefix_pad_token_id": 63,
                "vocab_size": [vocab_n] * len(levels),
                "predict_level0": False,
                "attn_pattern": "block_diagonal",
                "use_positional_encoding": True,
                "use_level_encoding": True,
                "use_rope": True,
                "rope_base": 100000,
                "use_qk_norm": True,
                "ctx_encoder": {"enabled": False, "n_layer": 1, "vocab_size": in_vocab},
                "use_flex_attention": False,
                "query_level0": True,
                "tie_heads_to_codebook": True,
                "tie_heads_learn_scale": True,
            },
            "tokenizer": {
                "pre_tokenizer": {
                    "name": "hf",
                    "tokenizer": "hf",
                    "model_id": "gpt2",
                    "special_tokens": {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"},
                },
                "vqvae": {
                    "n_layers": 2,
                    "embed_dim": 16,
                    "in_vocab_size": in_vocab,
                    "context_length": target_len,
                    "compression_factor": 1,
                    "dropout": 0.0,
                    "pre_quant_groupnorm": 4,
                    "use_qk_norm": True,
                    "pre_quant_dropout": 0.0,
                    "enc_layers": 1,
                    "dec_layers": 1,
                    "vqvae_vocab_size": [vocab_n] * len(levels),
                    "checkpoint_path": "tokenizer.pt",
                    "vector_quantizer_config": {
                        "clss": "multiscale_residual_vector_quantizer",
                        "decay": 0.99,
                        "epsilon": 1.0e-5,
                        "commitment_cost": 0.25,
                        "restart_threshold": 1.0,
                        "kmeans_init": False,
                        "learned_l1_sampling": True,
                        "learned_l1_cascade": True,
                        "learned_l1_cascade_base": 2,
                        "learned_all_sampling": False,
                        "quant_resi": {
                            "enabled": True,
                            "ratio": 0.5,
                            "share_mode": 0,
                            "learnable_ratio": False,
                        },
                        "aux": {"fine_drop": {"prob": 0.5, "min_keep": 1}},
                        "levels": {"use_manual_levels": True, "manual_levels": levels},
                        "learned_l1_cascade_norm": True,
                        "learned_l1_cascade_norm_final": True,
                    },
                },
            },
            "train": {
                "stage1": {
                    "batch_size": 4,
                    "grad_accum": 1,
                    "lr": 1.0e-3,
                    "min_lr": 1.0e-4,
                    "warmup_steps": 1,
                    "max_steps": 4,
                    "save_every": 2,
                    "eval_every": 2,
                    "log_every": 1,
                    "kmeans_init_batches": 2,
                    "foreach_adamw": False,
                },
                "stage2": {
                    "optimizer": optimizer,
                    "batch_size": 2,
                    "grad_accum": 1,
                    "lr": 1.0e-3,
                    "min_lr": 1.0e-4,
                    "warmup_steps": 1,
                    "max_steps": 4,
                    "stable_frac": 0.5,
                    "gradient_checkpointing": grad_ckpt,
                    "foreach_adamw": False,
                    "save_every": 2,
                    "eval_every": 2,
                    "log_every": 1,
                    "export_every": 2,
                },
            },
        }
    )
    OmegaConf.save(cfg, cfg_path)
    return cfg_path


def _write_mini_memmap(
    memmap_root: Path,
    *,
    prefix_len: int = 16,
    target_len: int = 8,
    concept_len: int = 12,
    n_samples: int = 8,
    with_gt_idx: bool = False,
) -> None:
    rng = np.random.default_rng(7)
    s1 = memmap_root / "stage1_vqvae"
    s1.mkdir(parents=True, exist_ok=True)
    np.asarray(rng.integers(0, 60, size=(n_samples, target_len)), dtype=np.uint16).tofile(
        s1 / f"blocks_{target_len}.bin"
    )
    (s1 / "meta.json").write_text(
        json.dumps({"num_samples": n_samples, "block_len": target_len, "dtype": "uint16"}),
        encoding="utf-8",
    )

    for split_name in ("stage2_wide", "stage2_gold", "eval"):
        sd = memmap_root / split_name
        sd.mkdir(parents=True, exist_ok=True)
        np.asarray(rng.integers(0, 60, size=(n_samples, prefix_len)), dtype=np.uint16).tofile(
            sd / f"prefix_{prefix_len}.bin"
        )
        np.asarray(rng.integers(0, 60, size=(n_samples, target_len)), dtype=np.uint16).tofile(
            sd / f"target_{target_len}.bin"
        )
        np.asarray([i % 4 for i in range(n_samples)], dtype=np.uint8).tofile(sd / "modes.bin")
        meta = {
            "num_samples": n_samples,
            "prefix_len": prefix_len,
            "target_len": target_len,
            "concept_len": concept_len,
            "dtype": "uint16",
        }
        if with_gt_idx:
            np.asarray(rng.integers(0, 32, size=(n_samples, concept_len)), dtype=np.uint16).tofile(
                sd / f"gt_idx_{concept_len}.bin"
            )
        (sd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


class TestStage1VQVAETraining:
    def test_stage1_train_atomic_save_and_resume(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        _write_mini_memmap(memmap_root)

        rc1 = train_vqvae.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "2",
                "--no-init-gpt2-wte",
            ]
        )
        assert rc1 == 0
        latest = torch.load(run_dir / "vqvae_latest.pt", map_location="cpu", weights_only=False)
        assert latest["step"] == 2
        assert (run_dir / "export" / "tokenizer.pt").exists()
        tok_sd = torch.load(run_dir / "export" / "tokenizer.pt", map_location="cpu", weights_only=False)
        assert "model" in tok_sd
        assert not any("_km_done" in k for k in tok_sd["model"].keys())

        rc2 = train_vqvae.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "4",
                "--no-init-gpt2-wte",
                "--resume",
            ]
        )
        assert rc2 == 0
        latest2 = torch.load(run_dir / "vqvae_latest.pt", map_location="cpu", weights_only=False)
        assert latest2["step"] == 4
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["step"] == 4
        assert status["state"] == "completed"

    def test_stage1_stop_sentinel_exits_cleanly(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_mini_memmap(memmap_root)
        (run_dir / "STOP").write_text("stop", encoding="utf-8")

        rc = train_vqvae.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "4",
                "--no-init-gpt2-wte",
            ]
        )
        assert rc == 0
        latest = torch.load(run_dir / "vqvae_latest.pt", map_location="cpu", weights_only=False)
        assert latest["step"] == 0
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "stopped"

    def test_stage1_already_at_max_steps_writes_status_and_shuts_down_cleanly(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        _write_mini_memmap(memmap_root)

        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        (run_dir / "status.json").unlink()
        (run_dir / "export" / "tokenizer.pt").unlink()

        with patch.object(xpu_compat, "shutdown_device_cleanly", wraps=xpu_compat.shutdown_device_cleanly) as shutdown_mock:
            rc = train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                    "--no-init-gpt2-wte",
                    "--resume",
                ]
            )

        assert rc == 0
        assert (run_dir / "export" / "tokenizer.pt").is_file()
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["stage"] == "train_vqvae"
        assert status["state"] == "completed"
        assert status["step"] == 2
        shutdown_mock.assert_called_once_with(torch.device("cpu"))


class TestStage2NCMTraining:
    def test_wsd_schedule_wide_to_gold_and_continuous(self):
        phases = [
            train_ncm.wsd_schedule(s, max_steps=10, warmup_steps=2, stable_frac=0.8, lr=2e-4, min_lr=2e-5)
            for s in range(10)
        ]
        assert phases[0][0] == "warmup" and phases[0][2] == "wide"
        assert phases[1][0] == "warmup" and abs(phases[1][1] - 2e-4) < 1e-12
        assert phases[2][0] == "stable" and phases[2][2] == "wide" and abs(phases[2][1] - 2e-4) < 1e-12
        assert phases[7][0] == "stable" and phases[7][2] == "wide"
        assert phases[8][0] == "decay" and phases[8][2] == "gold"
        assert phases[9][0] == "decay" and phases[9][2] == "gold" and phases[9][1] < phases[8][1]

        c_phase, c_lr, c_split = train_ncm.wsd_schedule(
            10, max_steps=10, warmup_steps=2, stable_frac=0.8, lr=2e-4, min_lr=2e-5, continuous=True
        )
        assert "cycle_1" in c_phase and c_split == "wide" and c_lr >= 2e-5

    @pytest.mark.parametrize("with_gt_idx", [False, True])
    def test_stage2_train_resume_and_export_load(self, tmp_path: Path, with_gt_idx: bool):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml", grad_ckpt=True)
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        _write_mini_memmap(memmap_root, with_gt_idx=with_gt_idx)

        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        # Train steps 0 -> 2 (warmup + stable on wide)
        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                ]
            )
            == 0
        )
        latest_2 = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        assert latest_2["step"] == 2
        assert not any("checkpoint" in k or "_wrapped" in k for k in latest_2["model"].keys())

        # Resume steps 2 -> 4 (switches to decay on stage2_gold)
        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "4",
                    "--resume",
                ]
            )
            == 0
        )
        latest_4 = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        assert latest_4["step"] == 4
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["step"] == 4
        assert status["split"] == "gold"
        assert status["per_mode_loss"] is not None
        assert set(status["per_mode_loss"].keys()) == {"MODE_0", "MODE_1", "MODE_2", "MODE_3"}

        export_dir = run_dir / "export"
        assert (export_dir / "generator.pt").exists()
        assert (export_dir / "eval_config.yaml").exists()

        # Verify exported artifacts load cleanly (strict=True) with unigram priors
        eval_cfg = OmegaConf.load(export_dir / "eval_config.yaml")
        assert eval_cfg.task == "ncm"
        assert eval_cfg.checkpoints.generator == "generator.pt"
        assert eval_cfg.checkpoints.tokenizer == "tokenizer.pt"

        vq_cfg = OmegaConf.create(OmegaConf.to_container(eval_cfg.tokenizer.vqvae, resolve=True))
        vqvae = VQVAE(**vq_cfg)
        tok_ckpt = torch.load(export_dir / eval_cfg.checkpoints.tokenizer, map_location="cpu", weights_only=False)
        vqvae.load_state_dict(tok_ckpt["model"], strict=True)
        vqvae.eval()

        net = NextConceptTransformer(vqvae_local=vqvae, config=eval_cfg.model)
        gen_ckpt = torch.load(export_dir / eval_cfg.checkpoints.generator, map_location="cpu", weights_only=False)
        net.load_state_dict(gen_ckpt["model"], strict=True)
        net.eval()

        assert gen_ckpt["inference"]["token_unigram"].shape == (64,)
        assert gen_ckpt["inference"]["level0_unigram"].shape == (32,)

    def test_stage2_stop_sentinel_exits_cleanly(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        _write_mini_memmap(memmap_root)
        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "1",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )
        (run_dir / "STOP").write_text("stop", encoding="utf-8")
        rc = train_ncm.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "4",
            ]
        )
        assert rc == 0
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "stopped"
        assert (run_dir / "export" / "generator.pt").exists()

    def test_stage2_already_at_max_steps_writes_status_and_shuts_down_cleanly(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        _write_mini_memmap(memmap_root)
        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "1",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )
        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                ]
            )
            == 0
        )
        (run_dir / "status.json").unlink()
        (run_dir / "export" / "generator.pt").unlink()

        with patch.object(xpu_compat, "shutdown_device_cleanly", wraps=xpu_compat.shutdown_device_cleanly) as shutdown_mock:
            rc = train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                    "--resume",
                ]
            )

        assert rc == 0
        assert (run_dir / "export" / "generator.pt").is_file()
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["stage"] == "train_ncm"
        assert status["state"] == "completed"
        assert status["step"] == 2
        shutdown_mock.assert_called_once_with(torch.device("cpu"))

    def test_preencode_shuts_down_device_cleanly(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini.yaml")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run"
        _write_mini_memmap(memmap_root, with_gt_idx=False)
        for split_name in ("stage2_wide", "stage2_gold", "eval"):
            meta_p = memmap_root / split_name / "meta.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta["num_rows"] = meta["num_samples"]
            meta_p.write_text(json.dumps(meta), encoding="utf-8")
        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "1",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )
        with patch.object(xpu_compat, "shutdown_device_cleanly", wraps=xpu_compat.shutdown_device_cleanly) as shutdown_mock:
            summary = preencode_concepts.preencode_splits(
                config_path=str(cfg_path),
                tokenizer_ckpt=str(run_dir / "export" / "tokenizer.pt"),
                memmap_root=memmap_root,
                splits=["stage2_wide", "stage2_gold", "eval"],
                batch_size=4,
                device_arg="cpu",
                dtype_arg="float32",
            )
        assert (memmap_root / "unigrams.pt").is_file()
        assert summary["splits"]["eval"] == 8
        shutdown_mock.assert_called_once_with(torch.device("cpu"))

    def test_kahan_bf16_adamw_quadratic_convergence(self):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(8, 6, dtype=torch.bfloat16))
        b = torch.nn.Parameter(torch.randn(6, dtype=torch.bfloat16))
        target_w = torch.linspace(-1.0, 1.0, 48, dtype=torch.bfloat16).reshape(8, 6)
        target_b = torch.linspace(0.5, -0.5, 6, dtype=torch.bfloat16)

        opt = train_ncm.KahanBF16AdamW([w, b], lr=0.05, betas=(0.9, 0.95), weight_decay=0.0)
        init_loss = float(((w.float() - target_w.float()) ** 2).sum() + ((b.float() - target_b.float()) ** 2).sum())
        for _ in range(150):
            opt.zero_grad(set_to_none=True)
            loss = ((w.float() - target_w.float()) ** 2).sum() + ((b.float() - target_b.float()) ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = float(((w.float() - target_w.float()) ** 2).sum() + ((b.float() - target_b.float()) ** 2).sum())
        assert final_loss < init_loss * 0.01
        assert final_loss < 0.05

        st_w = opt.state[w]
        assert isinstance(st_w["step"], int) and st_w["step"] == 150
        assert st_w["kahan_comp"].dtype == torch.bfloat16
        assert st_w["exp_avg"].dtype == torch.bfloat16
        assert st_w["exp_avg_sq_row"].dtype == torch.float32 and st_w["exp_avg_sq_row"].shape == (8,)
        assert st_w["exp_avg_sq_col"].dtype == torch.float32 and st_w["exp_avg_sq_col"].shape == (6,)

        st_b = opt.state[b]
        assert st_b["kahan_comp"].dtype == torch.bfloat16
        assert st_b["exp_avg"].dtype == torch.bfloat16
        assert st_b["exp_avg_sq"].dtype == torch.float32 and st_b["exp_avg_sq"].shape == (6,)

        for p_state in opt.state.values():
            for val in p_state.values():
                if isinstance(val, torch.Tensor):
                    assert val.dtype != torch.float64

    def test_kahan_bf16_adamw_state_dict_resume_preserves_dtypes(self, tmp_path: Path):
        torch.manual_seed(11)
        w0 = torch.randn(4, 4, dtype=torch.bfloat16)
        b0 = torch.randn(4, dtype=torch.bfloat16)
        target_w = torch.ones(4, 4, dtype=torch.bfloat16)
        target_b = -torch.ones(4, dtype=torch.bfloat16)

        w_a = torch.nn.Parameter(w0.clone())
        b_a = torch.nn.Parameter(b0.clone())
        opt_a = train_ncm.KahanBF16AdamW([w_a, b_a], lr=1e-2, weight_decay=0.01)

        for _ in range(5):
            opt_a.zero_grad(set_to_none=True)
            loss = ((w_a.float() - target_w.float()) ** 2).sum() + ((b_a.float() - target_b.float()) ** 2).sum()
            loss.backward()
            opt_a.step()

        ckpt_path = tmp_path / "kahan_opt.pt"
        torch.save({"w": w_a.detach().cpu(), "b": b_a.detach().cpu(), "opt": opt_a.state_dict()}, ckpt_path)

        # Continue uninterrupted reference for 5 more steps
        for _ in range(5):
            opt_a.zero_grad(set_to_none=True)
            loss = ((w_a.float() - target_w.float()) ** 2).sum() + ((b_a.float() - target_b.float()) ** 2).sum()
            loss.backward()
            opt_a.step()

        # Resume from CPU checkpoint
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        w_b = torch.nn.Parameter(loaded["w"].to(torch.bfloat16))
        b_b = torch.nn.Parameter(loaded["b"].to(torch.bfloat16))
        opt_b = train_ncm.KahanBF16AdamW([w_b, b_b], lr=1e-2, weight_decay=0.01)
        opt_b.load_state_dict(loaded["opt"])

        assert opt_b.state[w_b]["exp_avg_sq_row"].dtype == torch.float32
        assert opt_b.state[w_b]["exp_avg_sq_col"].dtype == torch.float32
        assert opt_b.state[w_b]["kahan_comp"].dtype == torch.bfloat16
        assert opt_b.state[b_b]["exp_avg_sq"].dtype == torch.float32

        for _ in range(5):
            opt_b.zero_grad(set_to_none=True)
            loss = ((w_b.float() - target_w.float()) ** 2).sum() + ((b_b.float() - target_b.float()) ** 2).sum()
            loss.backward()
            opt_b.step()

        assert torch.allclose(w_a.float(), w_b.float(), atol=0.0, rtol=0.0)
        assert torch.allclose(b_a.float(), b_b.float(), atol=0.0, rtol=0.0)

    def test_stage2_train_kahan_bf16_resume(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini_kahan.yaml", grad_ckpt=True, optimizer="kahan_bf16_adamw")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run_kahan"
        _write_mini_memmap(memmap_root, with_gt_idx=False)
        for split_name in ("stage2_wide", "stage2_gold", "eval"):
            meta_p = memmap_root / split_name / "meta.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta["num_rows"] = meta["num_samples"]
            meta_p.write_text(json.dumps(meta), encoding="utf-8")

        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "bfloat16",
                    "--max-steps",
                    "2",
                ]
            )
            == 0
        )
        ckpt_2 = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        assert ckpt_2["step"] == 2
        assert all(
            t.dtype == torch.bfloat16
            for t in ckpt_2["model"].values()
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        )

        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "bfloat16",
                    "--max-steps",
                    "4",
                    "--resume",
                ]
            )
            == 0
        )
        ckpt_4 = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        assert ckpt_4["step"] == 4

    def test_kahan_bf16_adamw_2d_factored_large_shape_stays_nonnegative_and_finite(self):
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.randn(8192, 2048, dtype=torch.bfloat16))
        opt = train_ncm.KahanBF16AdamW([p], lr=1.5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        for step_idx in range(3):
            opt.zero_grad(set_to_none=True)
            p.grad = (torch.randn_like(p) * (10.0 ** (step_idx - 1))).to(torch.bfloat16)
            opt.step()

        state = opt.state[p]
        assert torch.all(state["exp_avg_sq_row"] >= 0)
        assert torch.all(state["exp_avg_sq_col"] >= 0)
        assert torch.isfinite(p).all()
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq_row"]).all()
        assert torch.isfinite(state["exp_avg_sq_col"]).all()
        assert torch.isfinite(state["kahan_comp"]).all()

    def test_memmap_stage2_dataset_loads_mode_id_bin_and_eval_splits_modes(self, tmp_path: Path):
        cfg_path = _make_mini_config(tmp_path / "mini_modes.yaml")
        cfg = OmegaConf.load(cfg_path)
        vqvae_cfg = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer.vqvae, resolve=True))
        vqvae = train_ncm.VQVAE(**vqvae_cfg)
        vqvae.eval()
        net = train_ncm.NextConceptTransformer(vqvae_local=vqvae, config=cfg.model)
        net.eval()

        split_dir = tmp_path / "eval_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(123)
        np.asarray(rng.integers(0, 60, size=(4, 16)), dtype=np.uint16).tofile(split_dir / "prefix_16.bin")
        np.asarray(rng.integers(0, 60, size=(4, 8)), dtype=np.uint16).tofile(split_dir / "target_8.bin")
        np.asarray(rng.integers(0, 32, size=(4, 12)), dtype=np.uint16).tofile(split_dir / "gt_idx_12.bin")
        np.asarray([0, 1, 2, 3], dtype=np.uint8).tofile(split_dir / "mode_id.bin")
        (split_dir / "meta.json").write_text(
            json.dumps(
                {
                    "num_rows": 4,
                    "prefix_len": 16,
                    "target_len": 8,
                    "concept_len": 12,
                    "dtype": "uint16",
                    "mode_id_file": "mode_id.bin",
                }
            ),
            encoding="utf-8",
        )

        ds = train_ncm._MemmapStage2Dataset(split_dir, prefix_len=16, target_len=8, concept_len=12)
        assert len(ds) == 4
        assert [ds[i][3] for i in range(4)] == [0, 1, 2, 3]

        eval_loader = torch.utils.data.DataLoader(
            ds, batch_size=2, shuffle=False, drop_last=False, collate_fn=train_ncm._collate_stage2
        )
        val_loss, l0_acc, per_mode_loss = train_ncm._evaluate_ncm(
            net,
            vqvae,
            eval_loader,
            pad_token_id=63,
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            autocast_enabled=False,
        )
        assert np.isfinite(val_loss)
        assert 0.0 <= l0_acc <= 1.0
        assert set(per_mode_loss.keys()) == {"MODE_0", "MODE_1", "MODE_2", "MODE_3"}
        assert all(np.isfinite(v) for v in per_mode_loss.values())

        _, _, custom_mode_loss = train_ncm._evaluate_ncm(
            net,
            vqvae,
            eval_loader,
            pad_token_id=63,
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            autocast_enabled=False,
            mode_names={0: "TASK_A", 2: "TASK_C"},
        )
        assert set(custom_mode_loss.keys()) == {"TASK_A", "MODE_1", "TASK_C", "MODE_3"}

    def test_clip_grad_norm_fp32_and_nonfinite_guard(self, tmp_path: Path, capsys):
        p1 = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        p1.grad = torch.full_like(p1, 3.0)
        p2.grad = torch.full_like(p2, 4.0)
        norm_val, ok = train_ncm._clip_grad_norm_fp32([p1, p2], max_norm=1.0)
        assert ok is True
        expected_pre_norm = float(torch.sqrt(torch.tensor(16 * 9.0 + 4 * 16.0)).item())
        assert norm_val == pytest.approx(expected_pre_norm, rel=1e-2)
        post_norm = float(torch.sqrt(p1.grad.float().square().sum() + p2.grad.float().square().sum()).item())
        assert post_norm <= 1.0 + 1e-2

        for bad_val in (float("inf"), float("nan")):
            p_bad = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
            orig_grad = torch.tensor([[1.0, bad_val], [2.0, 3.0]], dtype=torch.bfloat16)
            p_bad.grad = orig_grad.clone()
            bad_norm, bad_ok = train_ncm._clip_grad_norm_fp32([p_bad], max_norm=1.0)
            assert bad_ok is False
            assert not np.isfinite(bad_norm)
            assert torch.equal(p_bad.grad.nan_to_num(nan=-999.0), orig_grad.nan_to_num(nan=-999.0))

        cfg_path = _make_mini_config(tmp_path / "mini_nonfinite.yaml", optimizer="kahan_bf16_adamw")
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run_nonfinite"
        _write_mini_memmap(memmap_root)
        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "1",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        orig_forward = train_ncm._forward_ncm_step
        call_count = 0

        def flaky_forward(net, vqvae, prefix_ids, target_ids, gt_idx, pad_token_id, device, autocast_dtype, autocast_enabled):
            nonlocal call_count
            call_count += 1
            loss, logits, resolved_gt = orig_forward(
                net, vqvae, prefix_ids, target_ids, gt_idx, pad_token_id, device, autocast_dtype, autocast_enabled
            )
            if call_count == 1:
                first_param = next(p for p in net.parameters() if p.requires_grad)
                bad_loss = loss + (first_param.float() * float("nan")).sum()
                return bad_loss, logits, resolved_gt
            return loss, logits, resolved_gt

        with patch.object(train_ncm, "_forward_ncm_step", side_effect=flaky_forward):
            rc = train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "bfloat16",
                    "--max-steps",
                    "2",
                ]
            )

        assert rc == 0
        out = capsys.readouterr().out
        assert "skipping optimizer.step()" in out
        ckpt = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        for k, v in ckpt["model"].items():
            if k != "attn_mask_for_masking" and torch.is_floating_point(v):
                assert torch.isfinite(v).all(), f"non-finite tensor {k}"

    def test_trim_prefix_to_bucket(self):
        pad_id = 50257
        p_full = 1536

        # 501 real tokens with bucket_size=256 rounds up to 512
        prefix = torch.full((2, p_full), pad_id, dtype=torch.long)
        prefix[0, :501] = 42
        prefix[1, :120] = 7
        trimmed = train_ncm._trim_prefix_to_bucket(prefix, pad_id, bucket_size=256, min_tokens=4)
        assert trimmed.shape == (2, 512)
        assert trimmed.is_contiguous()
        assert torch.equal(trimmed, prefix[:, :512])

        # Verify all 6 buckets (256..1536) cover lengths [1..1536]
        for n_tok, expected_len in [
            (1, 256),
            (256, 256),
            (257, 512),
            (501, 512),
            (512, 512),
            (513, 768),
            (1024, 1024),
            (1025, 1280),
            (1535, 1536),
            (1536, 1536),
        ]:
            row = torch.full((1, p_full), pad_id, dtype=torch.long)
            row[0, :n_tok] = 11
            out = train_ncm._trim_prefix_to_bucket(row, pad_id, bucket_size=256, min_tokens=4)
            assert out.shape == (1, expected_len)

        # Non-pad token at index -1 preserves full 1536 length unchanged
        full_tail = torch.full((2, p_full), pad_id, dtype=torch.long)
        full_tail[1, -1] = 99
        out_tail = train_ncm._trim_prefix_to_bucket(full_tail, pad_id, bucket_size=256, min_tokens=4)
        assert out_tail.shape == (2, p_full)
        assert torch.equal(out_tail, full_tail)

        # All-pad row clamps to eff_bucket = max(bucket_size, min_tokens, 64)
        all_pad = torch.full((2, p_full), pad_id, dtype=torch.long)
        out_pad_256 = train_ncm._trim_prefix_to_bucket(all_pad, pad_id, bucket_size=256, min_tokens=4)
        assert out_pad_256.shape == (2, 256)
        out_pad_64 = train_ncm._trim_prefix_to_bucket(all_pad, pad_id, bucket_size=32, min_tokens=4)
        assert out_pad_64.shape == (2, 64)

    def test_snapshot_to_cpu_preserves_dtypes_and_structure(self):
        bf16_t = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
        fp32_t = torch.randn(5, dtype=torch.float32, requires_grad=True)
        int_t = torch.tensor([1, 2, 3], dtype=torch.int64)
        payload = {
            "model": {"w": bf16_t, "b": fp32_t},
            "meta": [int_t, ("label", 123, None)],
            "step": 7,
        }
        snap = train_ncm._snapshot_to_cpu(payload)

        assert isinstance(snap, dict)
        assert isinstance(snap["model"], dict)
        assert isinstance(snap["meta"], list)
        assert isinstance(snap["meta"][1], tuple)
        assert snap["step"] == 7
        assert snap["meta"][1] == ("label", 123, None)

        for snap_t, orig_t in (
            (snap["model"]["w"], bf16_t),
            (snap["model"]["b"], fp32_t),
            (snap["meta"][0], int_t),
        ):
            assert isinstance(snap_t, torch.Tensor)
            assert snap_t.device == torch.device("cpu")
            assert snap_t.dtype == orig_t.dtype
            assert snap_t.requires_grad is False
            assert torch.equal(snap_t, orig_t.detach().cpu())

    def test_stage2_coinciding_checkpoint_export_best_snapshots_once(self, tmp_path: Path, monkeypatch):
        cfg_path = _make_mini_config(tmp_path / "mini_coincide.yaml", grad_ckpt=True)
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run_coincide"
        _write_mini_memmap(memmap_root)

        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "1",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        orig_net_sd = train_ncm.NextConceptTransformer.state_dict
        orig_opt_sd = torch.optim.Optimizer.state_dict
        orig_atomic_save = train_ncm._atomic_save
        net_sd_calls = 0
        opt_sd_calls = 0
        saved_paths: list[Path] = []

        def spy_net_sd(self, *args, **kwargs):
            nonlocal net_sd_calls
            net_sd_calls += 1
            return orig_net_sd(self, *args, **kwargs)

        def spy_opt_sd(self, *args, **kwargs):
            nonlocal opt_sd_calls
            opt_sd_calls += 1
            return orig_opt_sd(self, *args, **kwargs)

        def spy_atomic_save(obj, path):
            saved_paths.append(Path(path))
            return orig_atomic_save(obj, path)

        monkeypatch.setattr(train_ncm.NextConceptTransformer, "state_dict", spy_net_sd)
        monkeypatch.setattr(torch.optim.Optimizer, "state_dict", spy_opt_sd)
        monkeypatch.setattr(train_ncm, "_atomic_save", spy_atomic_save)

        rc = train_ncm.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "2",
                "--eval-every",
                "2",
                "--export-every",
                "2",
                "--save-every",
                "2",
            ]
        )
        assert rc == 0

        export_gen_path = run_dir / "export" / "generator.pt"
        best_ckpt_path = run_dir / "ncm_best.pt"
        latest_ckpt_path = run_dir / "ncm_latest.pt"

        assert saved_paths.count(export_gen_path) == 1
        assert saved_paths.count(best_ckpt_path) == 1
        assert saved_paths.count(latest_ckpt_path) == 1
        assert net_sd_calls == 1
        assert opt_sd_calls == 1

        for p in (export_gen_path, best_ckpt_path, latest_ckpt_path):
            loaded = torch.load(p, map_location="cpu", weights_only=False)
            assert loaded["step"] == 2
            assert "model" in loaded

    def test_stage2_desktop_pacing_and_vram_cap_hooks(self, tmp_path: Path, monkeypatch, capsys):
        cfg_path = _make_mini_config(tmp_path / "mini_pacing.yaml")
        cfg = OmegaConf.load(cfg_path)
        cfg.train.stage2.below_normal_priority = True
        cfg.train.stage2.xpu_memory_fraction = 0.85
        cfg.train.stage2.max_reserved_vram_mb = 12288.0
        cfg.train.stage2.desktop_yield_ms = 15.0
        OmegaConf.save(cfg, cfg_path)

        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run_pacing"
        _write_mini_memmap(memmap_root)
        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "1",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        prio_calls: list[int] = []
        cap_calls: list[tuple[torch.device, float, float]] = []
        trim_calls: list[tuple[torch.device, float]] = []

        monkeypatch.setattr(xpu_compat, "lower_process_priority_for_desktop", lambda: prio_calls.append(1), raising=False)
        monkeypatch.setattr(
            xpu_compat,
            "configure_vram_cap",
            lambda dev, *, max_reserved_mb=0.0, memory_fraction=0.0: cap_calls.append(
                (dev, float(max_reserved_mb), float(memory_fraction))
            ),
            raising=False,
        )
        monkeypatch.setattr(
            xpu_compat,
            "trim_vram_cache_if_needed",
            lambda dev, mb: trim_calls.append((dev, float(mb))),
            raising=False,
        )

        rc = train_ncm.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "2",
                "--grad-accum",
                "2",
                "--eval-every",
                "2",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "bucket=256 vram_cap_mb=12288 yield_ms=15" in out
        assert len(prio_calls) == 1
        assert cap_calls == [(torch.device("cpu"), 12288.0, 0.85)]
        assert len(trim_calls) >= 3
        assert all(dev == torch.device("cpu") and mb == 12288.0 for dev, mb in trim_calls)

        # CLI overrides: --no-below-normal-priority, --prefix-bucket-size, --max-reserved-vram-mb, --xpu-memory-fraction, --desktop-yield-ms
        prio_calls.clear()
        cap_calls.clear()
        trim_calls.clear()
        run_dir_cli = tmp_path / "run_pacing_cli"
        (run_dir_cli / "export").mkdir(parents=True, exist_ok=True)
        (run_dir_cli / "export" / "tokenizer.pt").write_bytes((run_dir / "export" / "tokenizer.pt").read_bytes())

        rc2 = train_ncm.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir_cli),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "1",
                "--grad-accum",
                "1",
                "--eval-every",
                "1",
                "--prefix-bucket-size",
                "128",
                "--max-reserved-vram-mb",
                "4096",
                "--xpu-memory-fraction",
                "0.6",
                "--desktop-yield-ms",
                "0",
                "--no-below-normal-priority",
            ]
        )
        assert rc2 == 0
        out2 = capsys.readouterr().out
        assert "bucket=128 vram_cap_mb=4096 yield_ms=0" in out2
        assert len(prio_calls) == 0
        assert cap_calls == [(torch.device("cpu"), 4096.0, 0.6)]
        assert len(trim_calls) >= 2
        assert all(dev == torch.device("cpu") and mb == 4096.0 for dev, mb in trim_calls)

        # Verify unmocked real xpu_compat.configure_vram_cap and trim_vram_cache_if_needed accept kwargs with zero TypeError
        monkeypatch.undo()
        run_dir_real = tmp_path / "run_pacing_real"
        (run_dir_real / "export").mkdir(parents=True, exist_ok=True)
        (run_dir_real / "export" / "tokenizer.pt").write_bytes((run_dir / "export" / "tokenizer.pt").read_bytes())

        rc3 = train_ncm.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir_real),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "1",
                "--grad-accum",
                "1",
                "--eval-every",
                "1",
                "--max-reserved-vram-mb",
                "11800",
                "--xpu-memory-fraction",
                "0.86",
                "--no-below-normal-priority",
            ]
        )
        assert rc3 == 0
        out3 = capsys.readouterr().out
        assert "bucket=256 vram_cap_mb=11800 yield_ms=15" in out3

    def test_stage2_cpu_offload_and_adaptive_ckpt_cross_resume(self, tmp_path: Path):
        cfg_path = _make_mini_config(
            tmp_path / "mini_hybrid_offload.yaml",
            grad_ckpt=True,
            optimizer="kahan_bf16_adamw",
        )
        memmap_root = tmp_path / "memmap"
        run_dir = tmp_path / "run_hybrid"
        _write_mini_memmap(memmap_root, with_gt_idx=False)

        assert (
            train_vqvae.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--max-steps",
                    "2",
                    "--no-init-gpt2-wte",
                ]
            )
            == 0
        )

        # Stage A: 2 steps without optimizer CPU offload or adaptive activation budget
        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "bfloat16",
                    "--no-optimizer-cpu-offload",
                    "--activation-budget-mb",
                    "0",
                    "--max-steps",
                    "2",
                ]
            )
            == 0
        )

        # Stage B: cross-resume -> 4 steps with hybrid CPU offload and adaptive checkpointing
        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "bfloat16",
                    "--resume",
                    "--optimizer-cpu-offload",
                    "--activation-budget-mb",
                    "32",
                    "--max-steps",
                    "4",
                ]
            )
            == 0
        )

        ckpt_b = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        assert ckpt_b["step"] == 4
        opt_state = ckpt_b["optimizer"]["state"]
        assert len(opt_state) > 0
        canonical_keys = {
            "step",
            "kahan_comp",
            "exp_avg",
            "exp_avg_sq_row",
            "exp_avg_sq_col",
            "exp_avg_sq",
        }
        for param_id, p_state in opt_state.items():
            assert "param_bf16" not in p_state
            assert set(p_state.keys()).issubset(canonical_keys), (
                f"unexpected optimizer state keys for {param_id}: {set(p_state.keys())}"
            )
            for val in p_state.values():
                if isinstance(val, torch.Tensor):
                    assert val.device == torch.device("cpu")

        # Stage C: cross-resume back -> 6 steps without CPU offload or adaptive checkpointing
        assert (
            train_ncm.main(
                [
                    "--config",
                    str(cfg_path),
                    "--memmap-root",
                    str(memmap_root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cpu",
                    "--dtype",
                    "bfloat16",
                    "--resume",
                    "--no-optimizer-cpu-offload",
                    "--activation-budget-mb",
                    "0",
                    "--max-steps",
                    "6",
                ]
            )
            == 0
        )

        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "completed"
        assert status["step"] == 6
        assert (run_dir / "export" / "generator.pt").is_file()
        assert (run_dir / "ncm_latest.pt").is_file()

        gen_sd = torch.load(run_dir / "export" / "generator.pt", map_location="cpu", weights_only=False)
        for k, v in gen_sd["model"].items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                assert v.dtype == torch.bfloat16, f"expected bfloat16 for {k}, got {v.dtype}"
                if k != "attn_mask_for_masking":
                    assert torch.isfinite(v).all(), f"non-finite tensor {k}"
