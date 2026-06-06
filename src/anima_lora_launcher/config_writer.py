from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .recommender import RecommendedSettings


class RawToml(str):
    pass


def write_configs(
    *,
    run_dir: Path,
    sd_scripts_dir: Path,
    anima_model: Path,
    qwen3_model: Path,
    vae_model: Path,
    image_dir: Path,
    output_dir: Path,
    output_name: str,
    settings: RecommendedSettings,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = run_dir / "dataset_config.toml"
    train_config = run_dir / "train_config.toml"

    dataset_config.write_text(
        build_dataset_config(image_dir=image_dir, settings=settings),
        encoding="utf-8",
    )
    train_config.write_text(
        build_train_config(
            sd_scripts_dir=sd_scripts_dir,
            anima_model=anima_model,
            qwen3_model=qwen3_model,
            vae_model=vae_model,
            dataset_config=dataset_config,
            output_dir=output_dir,
            output_name=output_name,
            settings=settings,
        ),
        encoding="utf-8",
    )
    return dataset_config, train_config


def build_dataset_config(*, image_dir: Path, settings: RecommendedSettings) -> str:
    return "\n".join(
        [
            "[general]",
            f"caption_extension = {toml_value(settings.caption_extension)}",
            f"shuffle_caption = {toml_value(settings.shuffle_caption)}",
            f"keep_tokens = {toml_value(settings.keep_tokens)}",
            f"caption_dropout_rate = {toml_value(settings.caption_dropout_rate)}",
            f"caption_tag_dropout_rate = {toml_value(settings.caption_tag_dropout_rate)}",
            f"token_warmup_step = {toml_value(settings.token_warmup_step)}",
            "",
            "[[datasets]]",
            f"resolution = {toml_value(settings.resolution)}",
            f"batch_size = {toml_value(settings.train_batch_size)}",
            "enable_bucket = true",
            "bucket_reso_steps = 64",
            "min_bucket_reso = 256",
            f"max_bucket_reso = {toml_value(max(1024, settings.resolution))}",
            "",
            "  [[datasets.subsets]]",
            f"  image_dir = {toml_value(str(image_dir))}",
            f"  num_repeats = {toml_value(settings.num_repeats)}",
            "",
        ]
    )


def build_train_config(
    *,
    sd_scripts_dir: Path,
    anima_model: Path,
    qwen3_model: Path,
    vae_model: Path,
    dataset_config: Path,
    output_dir: Path,
    output_name: str,
    settings: RecommendedSettings,
) -> str:
    network_args: list[str] = []
    if settings.train_llm_adapter:
        network_args.extend(
            [
                "train_llm_adapter=True",
                "network_reg_lrs=.*llm_adapter.*=5e-5",
            ]
        )

    values: dict[str, Any] = {
        "pretrained_model_name_or_path": str(anima_model),
        "qwen3": str(qwen3_model),
        "vae": str(vae_model),
        "dataset_config": str(dataset_config),
        "output_dir": str(output_dir),
        "output_name": output_name,
        "save_model_as": "safetensors",
        "network_module": "networks.lora_anima",
        "network_dim": settings.network_dim,
        "network_alpha": settings.network_alpha,
        "learning_rate": RawToml(settings.learning_rate),
        "network_train_unet_only": settings.network_train_unet_only,
        "max_train_steps": settings.max_train_steps,
        "optimizer_type": settings.optimizer_type,
        "lr_scheduler": settings.lr_scheduler,
        "mixed_precision": settings.mixed_precision,
        "timestep_sampling": settings.timestep_sampling,
        "discrete_flow_shift": RawToml(settings.discrete_flow_shift),
        "gradient_checkpointing": settings.gradient_checkpointing,
        "cache_latents": settings.cache_latents,
        "cache_text_encoder_outputs": settings.cache_text_encoder_outputs,
        "vae_disable_cache": settings.vae_disable_cache,
        "vae_chunk_size": settings.vae_chunk_size,
        "seed": settings.seed,
        "save_every_n_epochs": settings.save_every_n_epochs,
    }
    if network_args:
        values["network_args"] = network_args

    lines = [f"# sd-scripts: {sd_scripts_dir}", ""]
    lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
    lines.append("")
    return "\n".join(lines)


def toml_value(value: Any) -> str:
    if isinstance(value, RawToml):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def settings_from_form(values: dict[str, Any]) -> RecommendedSettings:
    defaults = asdict(
        RecommendedSettings(
            resolution=1024,
            train_batch_size=1,
            num_repeats=8,
            max_train_steps=1600,
            network_dim=32,
            network_alpha=1,
            learning_rate="2e-5",
            network_train_unet_only=True,
            optimizer_type="AdamW8bit",
            lr_scheduler="constant",
            mixed_precision="bf16",
            timestep_sampling="sigmoid",
            discrete_flow_shift="1.0",
            vae_chunk_size=48,
            caption_extension=".txt",
            shuffle_caption=False,
            keep_tokens=1,
            caption_dropout_rate=0.0,
            caption_tag_dropout_rate=0.0,
            token_warmup_step=0.0,
            seed=42,
            save_every_n_epochs=1,
            gradient_checkpointing=True,
            cache_latents=True,
            cache_text_encoder_outputs=True,
            vae_disable_cache=True,
            train_llm_adapter=False,
            estimated_epochs=0,
            notes=(),
        )
    )
    defaults.update(values)
    return RecommendedSettings(**defaults)
