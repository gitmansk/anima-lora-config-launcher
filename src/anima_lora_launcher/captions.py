from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .recommender import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class CaptionEntry:
    image_path: Path
    caption_path: Path
    tags: tuple[str, ...]


def parse_tags(text: str) -> list[str]:
    return [tag.strip() for tag in text.replace("\n", ",").split(",") if tag.strip()]


def format_tags(tags: list[str] | tuple[str, ...]) -> str:
    return ", ".join(tag.strip() for tag in tags if tag.strip())


def image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        return []
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_caption_entries(image_dir: Path, caption_extension: str = ".txt") -> list[CaptionEntry]:
    entries: list[CaptionEntry] = []
    for image_path in image_paths(image_dir):
        caption_path = image_path.with_suffix(caption_extension)
        text = ""
        try:
            text = caption_path.read_text(encoding="utf-8")
        except OSError:
            pass
        entries.append(CaptionEntry(image_path, caption_path, tuple(parse_tags(text))))
    return entries


def tag_counts(entries: list[CaptionEntry]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in entries:
        counts.update(entry.tags)
    return counts


def remove_tags(entries: list[CaptionEntry], tags_to_remove: set[str]) -> int:
    changed = 0
    for entry in entries:
        if not entry.caption_path.exists():
            continue
        tags = [tag for tag in entry.tags if tag not in tags_to_remove]
        if tuple(tags) == entry.tags:
            continue
        entry.caption_path.write_text(format_tags(tags), encoding="utf-8")
        changed += 1
    return changed


def add_tags(entries: list[CaptionEntry], tags_to_add: list[str], *, position: str = "top") -> int:
    clean_tags = [tag for tag in tags_to_add if tag.strip()]
    if not clean_tags:
        return 0

    changed = 0
    for entry in entries:
        existing = [tag for tag in entry.tags if tag not in clean_tags]
        if position == "bottom":
            new_tags = existing + clean_tags
        else:
            new_tags = clean_tags + existing
        if tuple(new_tags) == entry.tags and entry.caption_path.exists():
            continue
        entry.caption_path.write_text(format_tags(new_tags), encoding="utf-8")
        changed += 1
    return changed


def apply_caption_edits(
    entries: list[CaptionEntry],
    *,
    tags_to_remove: set[str],
    tags_to_add_top: list[str],
    tags_to_add_bottom: list[str],
) -> int:
    top_tags = unique_tags(tags_to_add_top)
    bottom_tags = [tag for tag in unique_tags(tags_to_add_bottom) if tag not in top_tags]
    added_tags = set(top_tags) | set(bottom_tags)

    changed = 0
    for entry in entries:
        existing = [tag for tag in entry.tags if tag not in tags_to_remove and tag not in added_tags]
        new_tags = top_tags + existing + bottom_tags
        if tuple(new_tags) == entry.tags and entry.caption_path.exists():
            continue
        if not entry.caption_path.exists() and not new_tags:
            continue
        entry.caption_path.write_text(format_tags(new_tags), encoding="utf-8")
        changed += 1
    return changed


def unique_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = tag.strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result
