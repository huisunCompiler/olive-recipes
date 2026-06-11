# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Export FLUX.2-klein-4B sub-models to ONNX via Olive.
#
# Usage:
#   python export_models.py [--models transformer vae_decoder text_encoder]
#                           [--model_id <hf_id_or_local_path>]
#                           [--resolutions 1024x1024]
#                           [--output_dir ./output_model]
#
# Output layout:
#   output_model/
#     transformer/dd/replaced.onnx   NPU (RyzenAI)
#     vae_decoder/dd/replaced.onnx   NPU (RyzenAI)
#     text_encoder/model.onnx        CPU ONNX
#     tokenizer/                     from pipeline
#     scheduler/                     from pipeline

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from olive.workflows import run as olive_run

SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_MODEL_ID   = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_RESOLUTIONS = ["1024x1024"]
ALL_MODELS = ["transformer", "vae_decoder", "text_encoder"]
TEXT_ENCODER_BACKENDS = {
    "olive": "config_text_encoder.json",
    # Olive ModelBuilder fp16 (default recipe) → MatMulNBits INT4; see export_text_encoder_matmulnbits.
    "genai": None,
    # Legacy Olive SMP + GPTQ + onnxruntime-genai ModelBuilder path.
    "genai_olive": "config_text_encoder_genai.json",
}
MATMULNBITS_TEXT_ENCODER_BACKENDS = frozenset({"genai"})
DEFAULT_TEXT_ENCODER_BACKEND = "olive"
# genai: Olive ModelBuilder fp16 (prompt_embeds) → MatMulNBits INT4; no CLI --run_config required.
DEFAULT_TEXT_ENCODER_GENAI_OLIVE_RECIPE = (
    SCRIPT_DIR / "recipes" / "qwen3-4b-fp16-prompt-embeds-modelbuilder.json"
)

NON_ONNX_COMPONENTS = ["tokenizer", "tokenizer_2", "scheduler", "feature_extractor"]

STAGED_DIR = SCRIPT_DIR / "staged"
STAGING_MARKER = ".staged_from"
TEXT_ENCODER_WEIGHT_GLOBS = ("*.safetensors", "*.json")


def set_dd_env() -> None:
    if os.environ.get("DD_PLUGINS_ROOT"):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("ryzenai_dynamic_dispatch")
        if spec and spec.origin:
            dd_root = os.environ.get("DD_ROOT")
            if not dd_root or not os.path.exists(dd_root):
                os.environ["DD_ROOT"] = os.path.dirname(spec.origin).replace("\\", "/")
            bin_dir = os.path.join(os.path.dirname(spec.origin), "bin")
            if os.path.isdir(bin_dir):
                os.environ["DD_PLUGINS_ROOT"] = bin_dir
    except Exception:
        pass


def _fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _text_encoder_config_path(backend: str) -> Path:
    if backend not in TEXT_ENCODER_BACKENDS:
        raise ValueError(
            f"Unknown text encoder backend '{backend}'. Choices: {', '.join(TEXT_ENCODER_BACKENDS)}"
        )
    config_name = TEXT_ENCODER_BACKENDS[backend]
    if config_name is None:
        raise ValueError(f"Backend '{backend}' does not use an Olive config file.")
    return SCRIPT_DIR / config_name


def _uses_matmulnbits_backend(backend: str) -> bool:
    return backend in MATMULNBITS_TEXT_ENCODER_BACKENDS


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink large weight files when possible; fall back to copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _staging_dir_for_pipeline(pipeline_root: Path) -> Path:
    digest = hashlib.sha256(str(pipeline_root).encode()).hexdigest()[:12]
    return STAGED_DIR / f"text_encoder_{digest}"


def resolve_pipeline_root(model_id: str | Path) -> Path:
    """Return the diffusers pipeline root for Flux2KleinPipeline loading."""
    path = Path(model_id).resolve()
    if (path / "model_index.json").exists():
        return path
    if (path / "text_encoder" / "config.json").exists():
        return path

    marker = path / STAGING_MARKER
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())

    raise ValueError(
        f"Cannot resolve diffusers pipeline root from '{path}'. "
        "Pass --model_id pointing to the FLUX.2-klein-4B pipeline directory."
    )


def stage_text_encoder_bundle(pipeline_root: str | Path) -> Path:
    """Assemble text_encoder weights + tokenizer into one HF-style directory.

    Diffusers pipelines keep ``text_encoder/`` and ``tokenizer/`` as siblings.
    Olive's ModelBuilder expects a single checkpoint directory with
    ``config.json``, weight shards, and tokenizer files together.
    """
    pipeline_root = Path(pipeline_root).resolve()
    text_encoder_src = pipeline_root / "text_encoder"
    tokenizer_src = pipeline_root / "tokenizer"

    if not text_encoder_src.is_dir():
        raise FileNotFoundError(f"Missing text_encoder directory: {text_encoder_src}")
    if not (text_encoder_src / "config.json").exists():
        raise FileNotFoundError(f"Missing text_encoder config: {text_encoder_src / 'config.json'}")
    if not tokenizer_src.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_src}")

    dest = _staging_dir_for_pipeline(pipeline_root)
    marker = dest / STAGING_MARKER
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == str(pipeline_root):
        if (dest / "config.json").exists() and (dest / "tokenizer.json").exists():
            print(f"  [STAGE] Reusing staged text_encoder bundle: {dest}")
            return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for pattern in TEXT_ENCODER_WEIGHT_GLOBS:
        for src_file in sorted(text_encoder_src.glob(pattern)):
            if src_file.is_file():
                _link_or_copy(src_file, dest / src_file.name)

    for src_file in sorted(tokenizer_src.iterdir()):
        if src_file.is_file():
            _link_or_copy(src_file, dest / src_file.name)

    marker.write_text(str(pipeline_root), encoding="utf-8")
    print(f"  [STAGE] Assembled text_encoder bundle: {dest}")
    return dest


def _apply_genai_text_encoder_path(cfg: dict, staged_path: Path) -> bool:
    input_model = cfg.setdefault("input_model", {})
    changed = False
    staged = str(staged_path)
    if input_model.get("model_path") != staged:
        input_model["model_path"] = staged
        changed = True

    load_kwargs = input_model.setdefault("load_kwargs", {})
    extra_args = load_kwargs.get("extra_args")
    if isinstance(extra_args, dict) and extra_args.pop("subfolder", None) is not None:
        if not extra_args:
            load_kwargs.pop("extra_args", None)
        changed = True
    return changed


def _write_genai_text_encoder_config(staged_path: Path) -> None:
    config_path = SCRIPT_DIR / TEXT_ENCODER_BACKENDS["genai_olive"]
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    if _apply_genai_text_encoder_path(cfg, staged_path):
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        print(f"  [CONFIG] Updated {config_path.name} -> {staged_path}")


def prepare_genai_text_encoder(pipeline_root: str | Path, *, update_olive_config: bool = False) -> Path:
    """Stage a flat HF checkpoint directory for text encoder export."""
    staged_path = stage_text_encoder_bundle(pipeline_root)
    if update_olive_config:
        _write_genai_text_encoder_config(staged_path)
    return staged_path


def _write_text_encoder_footprint(footprint_dir: Path) -> None:
    """Write a minimal Olive footprint so assemble_output_dir can copy model.onnx."""
    footprint_dir.mkdir(parents=True, exist_ok=True)
    node_id = "matmulnbits_export"
    footprint = {
        node_id: {
            "parent_model_id": None,
            "model_id": node_id,
            "model_config_data": {
                "type": "onnxmodel",
                "config": {
                    "model_path": str(footprint_dir),
                    "onnx_file_name": "model.onnx",
                },
            },
            "from_pass": "onnxconversion",
        }
    }
    with (footprint_dir / "footprint.json").open("w", encoding="utf-8") as f:
        json.dump(footprint, f, indent=4)


def _find_latest_model_onnx(search_root: Path) -> Path:
    """Pick the most recently modified ``model.onnx`` under an Olive output tree."""
    candidates = list(search_root.rglob("model.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No model.onnx found under {search_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_olive_fp16_prompt_embed_modelbuilder(staged_model_dir: Path, run_config: Path) -> Path:
    """Run Olive ModelBuilder (fp16, prompt_embeds) and return path to ``model.onnx``."""
    run_config = run_config.resolve()
    if not run_config.is_file():
        raise FileNotFoundError(f"Olive run config not found: {run_config}")

    with run_config.open(encoding="utf-8") as f:
        cfg = json.load(f)

    input_model = cfg.setdefault("input_model", {})
    input_model["model_path"] = str(staged_model_dir.resolve())
    load_kw = input_model.setdefault("load_kwargs", {})
    load_kw.setdefault("trust_remote_code", True)

    out_rel = cfg.get("output_dir", "footprints/text_encoder_olive_mb_fp16")
    olive_out = Path(out_rel)
    if not olive_out.is_absolute():
        olive_out = (SCRIPT_DIR / olive_out).resolve()
    cfg["output_dir"] = str(olive_out)
    olive_out.mkdir(parents=True, exist_ok=True)

    sidecar = olive_out / "_export_models_olive_run_config.json"
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  [OLIVE] ModelBuilder fp16 (prompt_embeds): effective config → {sidecar}")

    olive_run(cfg)
    fp16_onnx = _find_latest_model_onnx(olive_out)
    print(f"  [OLIVE] fp16 ONNX: {fp16_onnx}")
    return fp16_onnx


def export_text_encoder_matmulnbits(
    staged_model_dir: Path,
    *,
    fp16_onnx_path: Path | None = None,
    olive_run_config: Path | None = None,
) -> Path:
    """genai text encoder: Olive ModelBuilder fp16 (default recipe) → MatMulNBits INT4.

    ``fp16_onnx_path``: skip Olive and quantize this ONNX.
    ``olive_run_config``: override the built-in Olive JSON (default:
    ``DEFAULT_TEXT_ENCODER_GENAI_OLIVE_RECIPE``). Ignored when ``fp16_onnx_path`` is set.
    """
    from text_encoder_matmulnbits import export_prompt_embeds_matmulnbits

    if fp16_onnx_path is not None and olive_run_config is not None:
        raise ValueError(
            "Do not pass both fp16_onnx_path and olive_run_config. "
            "Use --text_encoder_fp16_onnx alone, or omit it to run Olive (default or --text_encoder_olive_run_config)."
        )

    if fp16_onnx_path is not None:
        resolved_fp16 = fp16_onnx_path
    else:
        recipe = olive_run_config or DEFAULT_TEXT_ENCODER_GENAI_OLIVE_RECIPE
        recipe = Path(recipe).resolve()
        if not recipe.is_file():
            raise FileNotFoundError(
                f"genai Olive recipe not found: {recipe}. "
                "Restore recipes/ in this package or pass --text_encoder_olive_run_config."
            )
        print(f"  [GENAI] Using Olive recipe: {recipe}")
        resolved_fp16 = _run_olive_fp16_prompt_embed_modelbuilder(staged_model_dir, recipe)

    footprint_dir = SCRIPT_DIR / "footprints" / "text_encoder"
    output_onnx = footprint_dir / "model.onnx"
    export_prompt_embeds_matmulnbits(staged_model_dir, output_onnx, fp16_onnx_path=resolved_fp16)
    _write_text_encoder_footprint(footprint_dir)
    return output_onnx


def _config_paths_for_update(models: list[str], text_encoder_backend: str) -> list[Path]:
    paths: list[Path] = []
    for name in models:
        if name == "text_encoder":
            if _uses_matmulnbits_backend(text_encoder_backend):
                continue
            paths.append(_text_encoder_config_path(text_encoder_backend))
        else:
            paths.append(SCRIPT_DIR / f"config_{name}.json")
    return paths


def update_config_files(
    model_id: str | None,
    resolutions: list[str] | None,
    models: list[str],
    text_encoder_backend: str,
) -> None:
    for config_path in _config_paths_for_update(models, text_encoder_backend):
        if not config_path.exists():
            continue
        with config_path.open() as f:
            cfg = json.load(f)

        changed = False
        if model_id is not None:
            if config_path.name == TEXT_ENCODER_BACKENDS["genai_olive"]:
                pipeline_root = resolve_pipeline_root(model_id)
                changed |= _apply_genai_text_encoder_path(cfg, prepare_genai_text_encoder(pipeline_root, update_olive_config=False))
            elif cfg.get("input_model", {}).get("model_path") != model_id:
                cfg["input_model"]["model_path"] = model_id
                changed = True
        if resolutions is not None:
            for pass_cfg in cfg.get("passes", {}).values():
                if "resolutions" in pass_cfg and pass_cfg["resolutions"] != resolutions:
                    pass_cfg["resolutions"] = resolutions
                    changed = True

        if changed:
            with config_path.open("w") as f:
                json.dump(cfg, f, indent=4)
            print(f"  [CONFIG] Updated {config_path.name}")


def load_olive_config(submodel_name: str, text_encoder_backend: str = DEFAULT_TEXT_ENCODER_BACKEND) -> dict:
    if submodel_name == "text_encoder":
        if _uses_matmulnbits_backend(text_encoder_backend):
            raise ValueError(f"Backend '{text_encoder_backend}' does not use Olive configs.")
        config_path = _text_encoder_config_path(text_encoder_backend)
    else:
        config_path = SCRIPT_DIR / f"config_{submodel_name}.json"
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)



def _read_footprint(footprints_dir: Path, submodel_name: str) -> tuple[Path, Path]:
    """Parse footprint.json and return (conversion_path, optimized_path)."""
    from olive.model import ONNXModelHandler

    fp_path = footprints_dir / submodel_name / "footprint.json"
    with fp_path.open() as f:
        footprints = json.load(f)

    conversion_node = None
    optimized_node  = None
    modelbuilder_node = None
    for node in footprints.values():
        from_pass = (node.get("from_pass") or "").lower()
        if from_pass == "onnxconversion":
            conversion_node = node
        elif from_pass == "modelbuilder":
            modelbuilder_node = node
        else:
            optimized_node = node

    if conversion_node is None:
        if modelbuilder_node is not None:
            conversion_node = modelbuilder_node
            optimized_node = modelbuilder_node
        elif optimized_node is not None:
            print(
                f"  [WARN] OnnxConversion footprint node not found for '{submodel_name}'; "
                "using last optimization pass output."
            )
            conversion_node = optimized_node
        else:
            raise RuntimeError(
                f"OnnxConversion footprint node not found for '{submodel_name}' in {fp_path}."
            )
    # CPU-only models (text_encoder, vae_encoder) have no optimization pass;
    # the conversion output is the final artifact.
    if optimized_node is None:
        print(f"  [WARN] No optimization pass found for '{submodel_name}'; using conversion output.")
        optimized_node = conversion_node

    def _model_path(node: dict) -> Path:
        cfg = node.get("model_config_data") or node.get("model_config")
        if not cfg:
            raise KeyError(f"Footprint node for '{submodel_name}' missing model_config_data/model_config")
        return Path(ONNXModelHandler(**cfg["config"]).model_path)

    return _model_path(conversion_node), _model_path(optimized_node)


_PIPELINE_COMPONENT_MAP = {
    "transformer": "transformer",
    "vae":         ["vae_encoder", "vae_decoder"],   # VAE covers both encoder & decoder
    "text_encoder": "text_encoder",
}


def _save_vae_decoder_bn_stats(pipeline, output_dir: Path) -> None:
    """Extract BN running_mean / running_var from the VAE and save as
    bn.running_x.safetensors next to the vae_decoder ONNX model.

    The RyzenAI runtime loads these stats separately at inference time because
    the ONNX graph does not carry them as initializers.

    Strategy: scan the full VAE state_dict for keys ending in
    'running_mean' / 'running_var', pick the pair with the smallest
    channel dimension (typically 128 for AutoencoderKLFlux2).
    """
    dst = output_dir / "vae_decoder" / "bn.running_x.safetensors"
    if not (output_dir / "vae_decoder").exists() or dst.exists():
        return

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("  [WARN] safetensors not installed; skipping bn.running_x.safetensors")
        return

    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return

    sd = vae.state_dict()

    bn_candidates: list[tuple[str, torch.Tensor]] = []
    for key, val in sd.items():
        if key.endswith(".running_mean") and val.ndim == 1:
            prefix = key[: -len(".running_mean")]
            var_key = prefix + ".running_var"
            if var_key in sd:
                bn_candidates.append((prefix, val))

    if not bn_candidates:
        print("  [WARN] No BN running_mean found in VAE state_dict; skipping bn.running_x.safetensors")
        return

    # Use the entry with the smallest channel count.
    prefix, running_mean = min(bn_candidates, key=lambda t: t[1].numel())
    running_var = sd[prefix + ".running_var"]

    tensors = {
        "bn.running_mean": running_mean.detach().to(torch.bfloat16),
        "bn.running_var":  running_var.detach().to(torch.bfloat16),
    }
    save_file(tensors, str(dst))
    print(f"  [SAVE]  vae_decoder/bn.running_x.safetensors  ({prefix}, shape {list(running_mean.shape)})")


def _save_component_configs(pipeline, output_dir: Path) -> None:
    """Save config.json (and generation_config.json) for each ONNX sub-model."""
    import json as _json

    def _write_config(component, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        cfg = getattr(component, "config", None)
        if cfg is None:
            return
        save_fn = getattr(cfg, "save_pretrained", None) or getattr(component, "save_config", None)
        if save_fn:
            try:
                save_fn(str(dst_dir))
                return
            except Exception:
                pass
        to_dict = getattr(cfg, "to_dict", None)
        if to_dict:
            with (dst_dir / "config.json").open("w") as f:
                _json.dump(to_dict(), f, indent=2)

    for attr, targets in _PIPELINE_COMPONENT_MAP.items():
        component = getattr(pipeline, attr, None)
        if component is None:
            continue
        for target in ([targets] if isinstance(targets, str) else targets):
            dst_dir = output_dir / target
            if dst_dir.exists():
                _write_config(component, dst_dir)
                if attr == "text_encoder":
                    gen_cfg = getattr(component, "generation_config", None)
                    save_gen = getattr(gen_cfg, "save_pretrained", None) if gen_cfg else None
                    if save_gen:
                        try:
                            save_gen(str(dst_dir))
                        except Exception:
                            pass
                print(f"  [CONFIG]  {target}/config.json")

    _save_vae_decoder_bn_stats(pipeline, output_dir)


def assemble_output_dir(
    pipeline,
    submodel_names: list[str],
    footprints_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in submodel_names:
        dst_dir = output_dir / name

        try:
            _, optimized_path = _read_footprint(footprints_dir, name)
        except Exception as exc:
            print(f"  [WARN] Could not read footprint for '{name}': {exc}; skipping.")
            continue

        candidates = [
            optimized_path / "dd",
            optimized_path / "dynamic" / "dd",
            optimized_path.parent / "dd",
            optimized_path.parent / "dynamic" / "dd",
        ]

        if optimized_path.is_dir():
            if optimized_path.name == "dynamic":
                candidates.append(optimized_path / "dd")
            if optimized_path.name == "dd":
                candidates.append(optimized_path)

        for dd_path in candidates:
            if (dd_path / "replaced.onnx").exists():
                dd_src = dd_path
                break
        else:
            dd_src = None
        
        if dd_src is not None:
            dst_path = dst_dir / ("dd" if dd_src.name == "dd" else "dynamic" / "dd")
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(dd_src, dst_path)
            print(f"  [COPY NPU] {name} → {dst_path}")
        else:
            # CPU / plain ONNX: copy model.onnx and external data file only.
            onnx_file = optimized_path if optimized_path.is_file() else optimized_path / "model.onnx"
            if not onnx_file.exists():
                print(f"  [WARN] No ONNX file found for '{name}' at {onnx_file}; skipping.")
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_file, dst_dir / "model.onnx")
            for companion in onnx_file.parent.iterdir():
                if companion == onnx_file or companion.is_dir():
                    continue
                if companion.suffix == ".data" or companion.name.startswith(onnx_file.stem + "."):
                    shutil.copy2(companion, dst_dir / companion.name)
            print(f"  [COPY CPU]  {name} → {dst_dir / 'model.onnx'}")

    _save_component_configs(pipeline, output_dir)

    for attr in NON_ONNX_COMPONENTS:
        component = getattr(pipeline, attr, None)
        if component is None:
            continue
        save_fn = getattr(component, "save_pretrained", None)
        if save_fn is None:
            continue
        dest = output_dir / attr
        dest.mkdir(parents=True, exist_ok=True)
        save_fn(str(dest))
        print(f"  [SAVE]  {attr} → {dest}")

    # Write the top-level model_index.json so the directory is recognised
    # as a Diffusers pipeline by downstream loaders.
    pipeline.save_config(str(output_dir))
    print("  [SAVE]  model_index.json")

    print(f"\n  Pipeline assembled at: {output_dir}")


def optimize(args) -> dict[str, bool]:
    model_id   = args.model_id
    output_dir = Path(args.output_dir).resolve()

    print(f"\n[PIPELINE] Loading Flux2KleinPipeline from '{model_id}' ...")
    from diffusers import Flux2KleinPipeline
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    t_cfg   = pipeline.transformer.config
    vae_cfg = pipeline.vae.config
    print(f"  Transformer : in_channels={t_cfg.in_channels}, "
          f"joint_attention_dim={t_cfg.joint_attention_dim}, "
          f"num_layers={t_cfg.num_layers}")
    print(f"  VAE         : latent_channels={vae_cfg.latent_channels}, "
          f"scaling_factor={getattr(vae_cfg, 'scaling_factor', 'N/A')}")

    results: dict[str, bool] = {}
    total_t0 = time.monotonic()

    for submodel_name in args.models:
        print(f"\n{'=' * 60}\n  Exporting: {submodel_name}\n{'=' * 60}")
        backend = args.text_encoder_backend if submodel_name == "text_encoder" else DEFAULT_TEXT_ENCODER_BACKEND
        if submodel_name == "text_encoder":
            print(f"  text_encoder backend: {backend} ({TEXT_ENCODER_BACKENDS.get(backend, 'matmulnbits')})")
        t0 = time.monotonic()
        try:
            if submodel_name == "text_encoder" and _uses_matmulnbits_backend(backend):
                staged_path = prepare_genai_text_encoder(resolve_pipeline_root(model_id))
                fp16_onnx = (
                    Path(args.text_encoder_fp16_onnx).resolve()
                    if getattr(args, "text_encoder_fp16_onnx", None)
                    else None
                )
                olive_rc = (
                    Path(args.text_encoder_olive_run_config).resolve()
                    if getattr(args, "text_encoder_olive_run_config", None)
                    else None
                )
                export_text_encoder_matmulnbits(
                    staged_path,
                    fp16_onnx_path=fp16_onnx,
                    olive_run_config=olive_rc,
                )
                success = True
            else:
                olive_config = load_olive_config(submodel_name, text_encoder_backend=backend)
                olive_run(olive_config)
                success = True
        except Exception as exc:
            print(f"\n[ERROR] {submodel_name} export failed: {exc}")
            success = False
        elapsed = time.monotonic() - t0
        results[submodel_name] = success
        print(f"\n  [{'OK' if success else 'FAILED'}]  {submodel_name}  ({_fmt_seconds(elapsed)})")

    total_elapsed = time.monotonic() - total_t0

    print(f"\n{'=' * 60}\n  Assembling output directory ...\n{'=' * 60}")
    assemble_output_dir(pipeline, args.models, SCRIPT_DIR / "footprints", output_dir)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n  Export Summary\n{'=' * 60}")
    for name in args.models:
        print(f"  {'OK    ' if results.get(name) else 'FAILED'}  {name}")
    print(f"{'─' * 60}")
    print(f"  Total time : {_fmt_seconds(total_elapsed)}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    return results


def parse_args(raw_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B sub-models to ONNX via Olive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_models.py\n"
            "  python export_models.py --models transformer\n"
            "  python export_models.py --model_id /local/path/to/model\n"
            "  python export_models.py --models text_encoder --text_encoder_backend genai\n"
            "  python export_models.py --output_dir /data/flux2_klein_onnx"
        ),
    )
    parser.add_argument(
        "--model_id", default=None, type=str,
        help=(
            "HuggingFace model ID or local path. "
            "When provided, writes back to all config_*.json. "
            f"Default: value in config_*.json (initially '{DEFAULT_MODEL_ID}')."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=None, metavar="MODEL",
        help=f"Sub-models to export (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--resolutions", nargs="+", default=None, metavar="WxH",
        help=(
            "Target resolutions for VitisGenerateModelSD. "
            "When provided, writes back to all config_*.json. "
            f"Default: value in config_*.json (initially '{' '.join(DEFAULT_RESOLUTIONS)}')."
        ),
    )
    parser.add_argument(
        "--output_dir", default=str(SCRIPT_DIR / "output_model"), type=str,
        help="Assembled pipeline output directory. Default: <script_dir>/output_model",
    )
    parser.add_argument(
        "--text_encoder_backend",
        choices=sorted(TEXT_ENCODER_BACKENDS),
        default=DEFAULT_TEXT_ENCODER_BACKEND,
        help=(
            "Text encoder export backend. "
            "'olive' = OnnxConversion + ORT optimization (default). "
            "'genai' = Olive ModelBuilder fp16 (built-in recipe) → MatMulNBits INT4 prompt_embeds ONNX. "
            "'genai_olive' = legacy Olive SMP + GPTQ + ModelBuilder path."
        ),
    )
    parser.add_argument(
        "--text_encoder_fp16_onnx",
        default=None,
        type=str,
        metavar="PATH",
        help=(
            "Only with --text_encoder_backend genai: use this fp16 model.onnx instead of running Olive. "
            "MatMul → MatMulNBits INT4 only. Mutually exclusive with --text_encoder_olive_run_config."
        ),
    )
    parser.add_argument(
        "--text_encoder_olive_run_config",
        default=None,
        type=str,
        metavar="PATH",
        help=(
            "Only with --text_encoder_backend genai: override the built-in Olive JSON "
            f"(default: {DEFAULT_TEXT_ENCODER_GENAI_OLIVE_RECIPE.name}). "
            "Mutually exclusive with --text_encoder_fp16_onnx."
        ),
    )
    return parser.parse_args(raw_args)


def main(raw_args=None) -> None:
    set_dd_env()
    args = parse_args(raw_args)

    if args.text_encoder_fp16_onnx and args.text_encoder_olive_run_config:
        raise SystemExit(
            "Use at most one of --text_encoder_fp16_onnx and --text_encoder_olive_run_config."
        )

    if args.models:
        args.models = [m for m in ALL_MODELS if m in args.models]
    else:
        args.models = list(ALL_MODELS)

    if args.model_id is not None or args.resolutions is not None:
        print("\n[CONFIG] Syncing config_*.json ...")
        update_config_files(args.model_id, args.resolutions, args.models, args.text_encoder_backend)

    if args.model_id is None:
        if "text_encoder" in args.models and _uses_matmulnbits_backend(args.text_encoder_backend):
            args.model_id = str(DEFAULT_MODEL_ID)
        elif "text_encoder" in args.models:
            cfg_path = _text_encoder_config_path(args.text_encoder_backend)
            with cfg_path.open(encoding="utf-8") as f:
                args.model_id = json.load(f)["input_model"]["model_path"]
        else:
            cfg_path = SCRIPT_DIR / f"config_{args.models[0]}.json"
            with cfg_path.open(encoding="utf-8") as f:
                args.model_id = json.load(f)["input_model"]["model_path"]

    pipeline_root = resolve_pipeline_root(args.model_id)

    if "text_encoder" in args.models and (
        _uses_matmulnbits_backend(args.text_encoder_backend)
        or args.text_encoder_backend == "genai_olive"
    ):
        print("\n[STAGE] Preparing flat text_encoder bundle ...")
        staged_path = prepare_genai_text_encoder(
            pipeline_root,
            update_olive_config=args.text_encoder_backend == "genai_olive",
        )
        print(f"  text_encoder bundle: {staged_path}")
        print(f"  pipeline model_id  : {pipeline_root}")
        args.model_id = str(pipeline_root)

    if args.resolutions is None:
        args.resolutions = DEFAULT_RESOLUTIONS
        for name in args.models:
            with (SCRIPT_DIR / f"config_{name}.json").open() as f:
                cfg = json.load(f)
            for pass_cfg in cfg.get("passes", {}).values():
                if "resolutions" in pass_cfg:
                    args.resolutions = pass_cfg["resolutions"]
                    break
            else:
                continue
            break

    print("=" * 60)
    print("  FLUX.2-klein-4B  —  Olive ONNX Export")
    print("=" * 60)
    print(f"  model_id    : {args.model_id}")
    print(f"  sub-models  : {', '.join(args.models)}")
    if "text_encoder" in args.models:
        backend_label = TEXT_ENCODER_BACKENDS.get(args.text_encoder_backend) or "matmulnbits"
        print(f"  text_encoder: {args.text_encoder_backend} ({backend_label})")
        if args.text_encoder_backend == "genai":
            if args.text_encoder_fp16_onnx:
                print(f"  text_encoder fp16: existing ONNX → {args.text_encoder_fp16_onnx}")
            elif args.text_encoder_olive_run_config:
                print(f"  text_encoder fp16: Olive (custom recipe) → {args.text_encoder_olive_run_config}")
            else:
                print(f"  text_encoder fp16: Olive (default) → {DEFAULT_TEXT_ENCODER_GENAI_OLIVE_RECIPE}")
    print(f"  resolutions : {', '.join(args.resolutions)}")
    print(f"  output_dir  : {args.output_dir}")
    print("=" * 60)

    results = optimize(args)
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
