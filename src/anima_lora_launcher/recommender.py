from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import struct


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


GOAL_LABELS = {
    "character": "キャラ",
    "style": "画風",
    "clothing": "衣装/小物",
    "concept": "その他",
}


@dataclass(frozen=True)
class TrainingSetStats:
    image_count: int
    caption_count: int
    missing_caption_count: int
    max_width: int = 0
    max_height: int = 0
    typical_max_side: int = 0


@dataclass(frozen=True)
class RecommendedSettings:
    resolution: int
    train_batch_size: int
    num_repeats: int
    max_train_steps: int
    network_dim: int
    network_alpha: int
    learning_rate: str
    network_train_unet_only: bool
    optimizer_type: str
    lr_scheduler: str
    mixed_precision: str
    timestep_sampling: str
    discrete_flow_shift: str
    vae_chunk_size: int
    caption_extension: str
    shuffle_caption: bool
    keep_tokens: int
    caption_dropout_rate: float
    caption_tag_dropout_rate: float
    token_warmup_step: float
    seed: int
    save_every_n_epochs: int
    gradient_checkpointing: bool
    cache_latents: bool
    cache_text_encoder_outputs: bool
    vae_disable_cache: bool
    train_llm_adapter: bool
    estimated_epochs: float
    notes: tuple[str, ...]


def count_training_set(image_dir: Path, caption_extension: str = ".txt") -> TrainingSetStats:
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        return TrainingSetStats(0, 0, 0)

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    caption_count = 0
    max_sides: list[int] = []
    max_width = 0
    max_height = 0
    for image in images:
        if image.with_suffix(caption_extension).exists():
            caption_count += 1
        size = read_image_size(image)
        if size is not None:
            width, height = size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            max_sides.append(max(width, height))

    max_sides.sort()
    typical_max_side = max_sides[len(max_sides) // 2] if max_sides else 0

    return TrainingSetStats(
        image_count=len(images),
        caption_count=caption_count,
        missing_caption_count=len(images) - caption_count,
        max_width=max_width,
        max_height=max_height,
        typical_max_side=typical_max_side,
    )


def recommend_settings(
    *,
    vram_gb: int,
    image_count: int,
    caption_count: int,
    goal: str,
    image_resolution: int = 1024,
) -> RecommendedSettings:
    vram = max(1, int(vram_gb))
    images = max(1, int(image_count))
    captions = max(0, int(caption_count))
    normalized_goal = goal if goal in GOAL_LABELS else "character"

    desired_resolution = _target_resolution_for_images(image_resolution)
    resolution, batch_size, dim, alpha, vae_chunk = _hardware_profile(vram, desired_resolution)
    steps = _target_steps(images, normalized_goal)
    repeats = _repeats_for_images(images)
    steps_per_epoch = max(1, ceil(images * repeats / batch_size))
    estimated_epochs = steps / steps_per_epoch

    notes: list[str] = []
    if image_count <= 0:
        notes.append("教師画像が見つからないため、画像1枚相当の安全な仮設定にしました。")
    if captions < images:
        notes.append(f"キャプションがない画像が {images - captions} 枚あります。")
    if vram < 8:
        notes.append("VRAM 8GB未満はかなり厳しめです。解像度を下げた低VRAM設定にしています。")
    if resolution < desired_resolution:
        notes.append(f"VRAMに合わせて学習解像度を {desired_resolution} から {resolution} に下げています。")
    if normalized_goal == "style":
        notes.append("画風LoRAは少し長めのステップにしていますが、Anima向けに軽めへ抑えています。")
    if normalized_goal == "clothing":
        notes.append("衣装/小物LoRAは過学習しやすいため、ステップを控えめにしています。")
    notes.append("Anima Base v1の作者推奨に合わせ、LLM Adapterは学習しません。")
    notes.append("Text Encoder出力をキャッシュするため、DiTのみ学習し、shuffle/tag warmup/tag dropoutはオフにしています。")

    return RecommendedSettings(
        resolution=resolution,
        train_batch_size=batch_size,
        num_repeats=repeats,
        max_train_steps=steps,
        network_dim=dim,
        network_alpha=alpha,
        learning_rate=_learning_rate_for_goal(normalized_goal, dim),
        network_train_unet_only=True,
        optimizer_type="AdamW8bit",
        lr_scheduler="constant",
        mixed_precision="bf16",
        timestep_sampling="sigmoid",
        discrete_flow_shift="1.0",
        vae_chunk_size=vae_chunk,
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
        estimated_epochs=round(estimated_epochs, 2),
        notes=tuple(notes),
    )


def _hardware_profile(vram_gb: int, desired_resolution: int) -> tuple[int, int, int, int, int]:
    if vram_gb < 8:
        return min(desired_resolution, 768), 1, 16, 1, 16
    if vram_gb < 12:
        return min(desired_resolution, 896), 1, 24, 1, 32
    if vram_gb < 16:
        return min(desired_resolution, 1024), 1, 32, 1, 48
    if vram_gb < 24:
        return min(desired_resolution, 1280), 1, 32, 1, 64
    return desired_resolution, 1, 32, 1, 64


def _target_resolution_for_images(image_resolution: int) -> int:
    if image_resolution >= 1500:
        return 1536
    if image_resolution >= 1200:
        return 1280
    if image_resolution >= 900:
        return 1024
    return 768


def _repeats_for_images(image_count: int) -> int:
    if image_count < 10:
        return 20
    if image_count < 25:
        return 12
    if image_count < 50:
        return 8
    if image_count < 100:
        return 6
    return 4


def _target_steps(image_count: int, goal: str) -> int:
    tables = {
        "character": ((5, 500), (10, 700), (25, 900), (50, 1200), (100, 1600), (10_000, 2200)),
        "style": ((5, 650), (10, 900), (25, 1200), (50, 1600), (100, 2200), (10_000, 3000)),
        "clothing": ((5, 350), (10, 500), (25, 700), (50, 1000), (100, 1400), (10_000, 1900)),
        "concept": ((5, 500), (10, 750), (25, 1000), (50, 1300), (100, 1800), (10_000, 2400)),
    }
    for limit, steps in tables[goal]:
        if image_count < limit:
            return steps
    return tables[goal][-1][1]


def _learning_rate_for_goal(goal: str, dim: int) -> str:
    if goal == "clothing":
        return "1.5e-5"
    if goal == "style" and dim >= 64:
        return "1.5e-5"
    return "2e-5"


def read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Path(path).open("rb") as file:
            header = file.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                return struct.unpack(">II", header[16:24])
            if header[:2] == b"\xff\xd8":
                return _read_jpeg_size(file)
            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                return _read_webp_size(header, file)
    except OSError:
        return None
    return None


def _read_jpeg_size(file) -> tuple[int, int] | None:
    while True:
        marker_prefix = file.read(1)
        if not marker_prefix:
            return None
        if marker_prefix != b"\xff":
            continue
        marker = file.read(1)
        while marker == b"\xff":
            marker = file.read(1)
        if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
            segment = file.read(7)
            if len(segment) < 7:
                return None
            height, width = struct.unpack(">HH", segment[3:7])
            return width, height
        length_bytes = file.read(2)
        if len(length_bytes) < 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        file.seek(max(0, length - 2), 1)


def _read_webp_size(header: bytes, file) -> tuple[int, int] | None:
    chunk = header[12:16]
    if chunk == b"VP8X":
        data = header[20:30]
        if len(data) < 10:
            data += file.read(10 - len(data))
        width = int.from_bytes(data[4:7], "little") + 1
        height = int.from_bytes(data[7:10], "little") + 1
        return width, height
    if chunk == b"VP8 ":
        data = header[20:30]
        if len(data) < 10:
            data += file.read(10 - len(data))
        if len(data) >= 10:
            width = int.from_bytes(data[6:8], "little") & 0x3FFF
            height = int.from_bytes(data[8:10], "little") & 0x3FFF
            return width, height
    if chunk == b"VP8L":
        data = header[20:25]
        if len(data) < 5:
            data += file.read(5 - len(data))
        if len(data) >= 5 and data[0] == 0x2F:
            bits = int.from_bytes(data[1:5], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
    return None
