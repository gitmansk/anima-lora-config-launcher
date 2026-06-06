from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import datetime
from math import ceil
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .config_writer import settings_from_form, write_configs
from .recommender import GOAL_LABELS, RecommendedSettings, TrainingSetStats, count_training_set, recommend_settings
from .runner import build_training_command, decode_process_output, popen_training, stop_process_tree
from .settings_store import load_user_settings, save_user_settings


GOAL_LABEL_TO_KEY = {label: key for key, label in GOAL_LABELS.items()}
GOAL_KEYS = ("character", "style", "clothing", "concept")
GOAL_LABELS_EN = {
    "character": "Character",
    "style": "Style",
    "clothing": "Clothing/Item",
    "concept": "Other",
}
GOAL_LABEL_TO_KEY.update({label: key for key, label in GOAL_LABELS_EN.items()})
DEFAULT_IMAGE_RESOLUTION = 1024


HELP = {
    "lora_name": "保存されるLoRAファイル名です。半角英数字、_、- を推奨します。",
    "goal": "目的により学習ステップだけを変えます。画風は長め、衣装/小物は短め、キャラは中間です。",
    "vram": "12GB以上で通常の1024学習設定です。8GB前後では解像度を下げます。16GB以上は高解像度教師画像の時に効きます。",
    "image_resolution": "教師画像の長辺の目安です。フォルダ選択時に可能なら自動入力します。学習解像度のおすすめに使います。",
    "image_count": "教師画像フォルダ内の jpg/png/webp/bmp を数えます。同名.txtがあるものをcaptionありとして数えます。",
    "train_resolution": "sd-scriptsが学習時に画像をリサイズ/バケット分けする基準解像度です。教師画像の元サイズではありません。Animaは通常1024から始めます。",
    "steps": "学習の総ステップ数です。多いほど覚えますが、過学習もしやすくなります。",
    "repeats": "教師画像を1epoch内で何回繰り返すかです。画像枚数が少ないほど大きめにします。",
    "batch": "同時に学習する画像枚数です。増やすとVRAMを使います。Animaでは通常1です。",
    "rank": "LoRAの容量です。大きいほど表現力は増えますが、重くなり過学習もしやすくなります。Animaは32から始めます。",
    "alpha": "LoRAのスケールです。Anima公式例はalpha=1系なので通常は1のままで十分です。",
    "lr": "学習率です。大きいほど強く学習しますが壊れやすくなります。Animaのrank32では2e-5から始めます。",
    "save": "何epochごとにLoRAを保存するかです。通常は1で、進捗表示の保存区間にも使います。",
    "optimizer": "最適化アルゴリズムです。通常はAdamW8bitです。",
    "scheduler": "学習率の変化方法です。通常はconstantです。",
    "precision": "学習時の精度です。Animaではbf16を基本にします。",
    "timestep": "Animaのtimestep samplingです。通常はsigmoidです。",
    "flow": "timestep_sampling=shift時の補助値です。通常は1.0です。",
    "vae_chunk": "VAE処理を分割するサイズです。小さいほどVRAMを抑えますが遅くなることがあります。",
    "caption_ext": "画像と同名のcaptionファイル拡張子です。通常は.txtです。",
    "keep_tokens": "caption shuffle時に先頭から固定するタグ数です。通常設定ではshuffleを使わないので影響は小さいです。",
    "seed": "乱数シードです。同じ条件で再現したい時に固定します。",
    "caption_dropout": "captionを空にする割合です。AnimaではText Encoder cacheと併用可能ですが、通常は0です。",
    "tag_dropout": "caption内タグをランダムに落とす割合です。Text Encoder cacheとは併用できないため通常は0です。",
    "token_warmup": "序盤だけタグ数を少なくして徐々に増やす設定です。Text Encoder cacheとは併用できないため通常は0です。",
    "dit_only": "Text Encoder出力をキャッシュする場合、Text Encoder LoRAは学習できません。通常はオンです。",
    "cache_text": "Qwen3出力を事前計算してVRAMを下げます。オンの時はshuffle/tag warmup/tag dropoutを使えません。",
    "cache_latents": "VAE出力を事前計算してVRAMを下げます。通常はオンです。",
    "gradient_checkpointing": "VRAMを下げますが少し遅くなります。通常はオンです。",
    "vae_disable_cache": "VAE内部キャッシュを無効化してVRAMを抑えます。通常はオンです。",
    "shuffle": "captionタグをシャッフルします。AnimaでText Encoder cacheを使う通常設定ではオフです。",
    "llm_adapter": "LLM AdapterにもLoRAを入れます。壊れやすいため通常ユーザーはオフ推奨です。",
}


HELP_EN = {
    "lora_name": "The LoRA file name to save. ASCII letters, numbers, underscores, and hyphens are recommended.",
    "goal": "Only the recommended step count changes by goal. Style is longer, clothing/items are shorter, character is in the middle.",
    "vram": "12GB or more uses the usual 1024 setting. Around 8GB lowers resolution. 16GB or more helps when source images are high resolution.",
    "image_resolution": "Estimated long side of the training images. It is read when you create recommended settings and used to choose training resolution.",
    "image_count": "Counts jpg/png/webp/bmp files in the training image folder. A matching .txt file counts as a caption.",
    "train_resolution": "The base resolution sd-scripts uses for resizing and buckets during training. It is not the original image size. Anima usually starts at 1024.",
    "steps": "Total training steps. Higher values learn more but can overfit.",
    "repeats": "How many times the images are repeated inside one epoch. Smaller datasets use larger repeats.",
    "batch": "Images processed per training step. Higher values use more VRAM. Anima usually uses 1.",
    "rank": "LoRA capacity. Higher values can express more, but are heavier and can overfit. Anima starts well at 32.",
    "alpha": "LoRA scale. Anima examples use alpha=1 style settings, so 1 is normally enough.",
    "lr": "Learning rate. Higher values learn harder but can break more easily. For Anima rank 32, start around 2e-5.",
    "save": "Save a LoRA every N epochs. Usually 1. This is also used for the save progress bar.",
    "optimizer": "Optimization algorithm. Usually AdamW8bit.",
    "scheduler": "Learning-rate schedule. Usually constant.",
    "precision": "Training precision. Anima defaults to bf16 here.",
    "timestep": "Timestep sampling for Anima. Usually sigmoid.",
    "flow": "Extra value used when timestep_sampling is shift. Usually 1.0.",
    "vae_chunk": "Chunk size for VAE processing. Smaller values reduce VRAM but can be slower.",
    "caption_ext": "Caption file extension for files with the same basename as images. Usually .txt.",
    "keep_tokens": "Number of leading caption tags to keep fixed when shuffle is used. It has little effect when shuffle is off.",
    "seed": "Random seed. Fix it when you want repeatable runs.",
    "caption_dropout": "Chance to use an empty caption. Normally 0.",
    "tag_dropout": "Chance to drop tags inside captions. It cannot be used with Text Encoder cache, so normally 0.",
    "token_warmup": "Starts with fewer tags and increases them gradually. It cannot be used with Text Encoder cache, so normally 0.",
    "dit_only": "When Text Encoder outputs are cached, Text Encoder LoRA cannot be trained. Usually on.",
    "cache_text": "Precomputes Qwen3 outputs to reduce VRAM. When on, shuffle/tag warmup/tag dropout cannot be used.",
    "cache_latents": "Precomputes VAE outputs to reduce VRAM. Usually on.",
    "gradient_checkpointing": "Reduces VRAM use, with a small speed cost. Usually on.",
    "vae_disable_cache": "Disables the internal VAE cache to reduce VRAM. Usually on.",
    "shuffle": "Shuffles caption tags. Off for the usual Anima setup with Text Encoder cache.",
    "llm_adapter": "Also trains the LLM Adapter. It is easier to break, so normal users should leave this off.",
}


UI_TEXT = {
    "ja": {
        "app_title": "Anima LoRA Launcher",
        "language_label": "表示言語",
        "language_ja": "日本語",
        "language_en": "English",
        "paths": "パス",
        "basic_settings": "基本設定",
        "recommended_settings": "おすすめ設定",
        "training_settings": "学習設定",
        "advanced_settings": "高度設定",
        "progress": "進捗",
        "log": "ログ",
        "sd_scripts": "sd-scripts",
        "anima": "Anima",
        "qwen3": "Qwen3",
        "vae": "VAE",
        "teacher_images": "教師画像",
        "output_dir": "出力先",
        "lora_name": "LoRA名",
        "vram": "VRAM",
        "goal": "目的",
        "train_resolution": "学習解像度",
        "steps": "steps",
        "repeats": "repeats",
        "batch": "batch",
        "rank": "rank",
        "alpha": "alpha",
        "learning_rate": "learning rate",
        "save_epochs": "save epochs",
        "optimizer": "optimizer",
        "scheduler": "scheduler",
        "precision": "precision",
        "timestep": "timestep",
        "flow_shift": "flow shift",
        "vae_chunk": "VAE chunk",
        "caption_ext": "caption ext",
        "keep_tokens": "keep tokens",
        "seed": "seed",
        "caption_dropout": "caption dropout",
        "tag_dropout": "tag dropout",
        "token_warmup": "token warmup",
        "dit_only": "DiTのみ学習",
        "cache_text": "cache text encoder",
        "cache_latents": "cache latents",
        "gradient_checkpointing": "gradient checkpointing",
        "vae_disable_cache": "VAE cache無効",
        "shuffle_caption": "shuffle caption",
        "train_llm_adapter": "train LLM adapter",
        "choose": "選択",
        "create_recommended": "おすすめ設定を作成する",
        "show_advanced": "高度設定を表示",
        "hide_advanced": "高度設定を隠す",
        "start_training": "学習開始",
        "stop": "停止",
        "open_output": "出力先フォルダを開く",
        "stats_initial": "教師画像: おすすめ設定作成時に読み込みます",
        "stats_missing": "未指定",
        "stats_no_size": "サイズ未取得",
        "stats_text": "{images}枚 / caption {captions} / 未caption {missing} / {size}",
        "stats_size": "中央長辺 {typical}px / 最大 {width}x{height}",
        "log_no_image_dir": "教師画像フォルダが未指定です。",
        "log_stats": "教師画像を確認しました: images={images}, captions={captions}, missing={missing}, image_resolution={resolution}",
        "log_recommended": "おすすめ設定を作成しました。推定epoch: {epochs}",
        "log_note": "注意: {note}",
        "log_config_created": "設定ファイルを作成しました: {run_dir}",
        "log_command": "コマンド:",
        "log_start": "学習を開始します。",
        "log_pid": "学習プロセスPID: {pid}",
        "log_stop_none": "停止対象のプロセスはありません。",
        "log_stop": "停止します。PID {pid} のプロセスツリーを終了します。",
        "log_stop_ok": "停止処理を実行しました。終了確認を待っています。",
        "log_stop_fail": "停止処理に失敗しました。ログを確認してください。",
        "log_process_stopped": "学習プロセスを停止しました。exit_code={code}",
        "log_process_finished": "学習プロセスが終了しました。exit_code={code}",
        "log_read_error": "ログ読み取りエラー: {error}",
        "log_open_output": "出力先フォルダを開きました: {path}",
        "log_saved_prefs": "パス設定を保存しました: {path}",
        "error_title": "設定エラー",
        "running_title": "実行中",
        "running_message": "すでに学習が実行中です。",
        "invalid_output": "出力先フォルダを正しく選択してください。",
        "open_output_error": "出力先フォルダを開けませんでした: {error}",
        "path_dir_error": "{label}フォルダを正しく選択してください。",
        "path_file_error": "{label}ファイルを正しく選択してください。",
        "qwen_error": "Qwen3ファイルまたはフォルダを正しく選択してください。",
        "lora_name_error": "LoRA名を入力してください。",
        "sd_script_error": "sd-scriptsフォルダ内に anima_train_network.py が見つかりません。",
        "cache_shuffle_error": "Animaでは cache text encoder と shuffle caption を同時に使えません。",
        "cache_dit_error": "Animaでは cache text encoder を使う場合、DiTのみ学習をオンにしてください。",
        "cache_warmup_error": "Animaでは cache text encoder と token warmup を同時に使えません。",
        "cache_tag_error": "Animaでは cache text encoder と tag dropout を同時に使えません。",
        "overall_progress_initial": "全体進捗: -",
        "save_progress_initial": "保存区間: -",
        "overall_progress": "全体進捗: {current} / {total} steps",
        "save_progress_start": "保存区間: 1 / {slots}",
        "save_progress": "保存区間: {slot} / {slots}  次の保存まで {remaining} steps",
    },
    "en": {
        "app_title": "Anima LoRA Launcher",
        "language_label": "Language",
        "language_ja": "日本語",
        "language_en": "English",
        "paths": "Paths",
        "basic_settings": "Basic Settings",
        "recommended_settings": "Recommended Settings",
        "training_settings": "Training Settings",
        "advanced_settings": "Advanced Settings",
        "progress": "Progress",
        "log": "Log",
        "sd_scripts": "sd-scripts",
        "anima": "Anima",
        "qwen3": "Qwen3",
        "vae": "VAE",
        "teacher_images": "Training Images",
        "output_dir": "Output Folder",
        "lora_name": "LoRA Name",
        "vram": "VRAM",
        "goal": "Goal",
        "train_resolution": "Resolution",
        "steps": "steps",
        "repeats": "repeats",
        "batch": "batch",
        "rank": "rank",
        "alpha": "alpha",
        "learning_rate": "learning rate",
        "save_epochs": "save epochs",
        "optimizer": "optimizer",
        "scheduler": "scheduler",
        "precision": "precision",
        "timestep": "timestep",
        "flow_shift": "flow shift",
        "vae_chunk": "VAE chunk",
        "caption_ext": "caption ext",
        "keep_tokens": "keep tokens",
        "seed": "seed",
        "caption_dropout": "caption dropout",
        "tag_dropout": "tag dropout",
        "token_warmup": "token warmup",
        "dit_only": "Train DiT only",
        "cache_text": "cache text encoder",
        "cache_latents": "cache latents",
        "gradient_checkpointing": "gradient checkpointing",
        "vae_disable_cache": "disable VAE cache",
        "shuffle_caption": "shuffle caption",
        "train_llm_adapter": "train LLM adapter",
        "choose": "Choose",
        "create_recommended": "Create Recommended Settings",
        "show_advanced": "Show Advanced Settings",
        "hide_advanced": "Hide Advanced Settings",
        "start_training": "Start Training",
        "stop": "Stop",
        "open_output": "Open Output Folder",
        "stats_initial": "Training images: read when recommendations are created",
        "stats_missing": "Not selected",
        "stats_no_size": "image size unavailable",
        "stats_text": "{images} images / captions {captions} / missing captions {missing} / {size}",
        "stats_size": "median long side {typical}px / max {width}x{height}",
        "log_no_image_dir": "Training image folder is not set.",
        "log_stats": "Checked training images: images={images}, captions={captions}, missing={missing}, image_resolution={resolution}",
        "log_recommended": "Created recommended settings. Estimated epochs: {epochs}",
        "log_note": "Note: {note}",
        "log_config_created": "Created config files: {run_dir}",
        "log_command": "Command:",
        "log_start": "Starting training.",
        "log_pid": "Training process PID: {pid}",
        "log_stop_none": "No process to stop.",
        "log_stop": "Stopping process tree for PID {pid}.",
        "log_stop_ok": "Stop command was sent. Waiting for exit confirmation.",
        "log_stop_fail": "Stop command failed. Check the log.",
        "log_process_stopped": "Training process was stopped. exit_code={code}",
        "log_process_finished": "Training process finished. exit_code={code}",
        "log_read_error": "Log read error: {error}",
        "log_open_output": "Opened output folder: {path}",
        "log_saved_prefs": "Saved path settings: {path}",
        "error_title": "Settings Error",
        "running_title": "Running",
        "running_message": "Training is already running.",
        "invalid_output": "Select a valid output folder.",
        "open_output_error": "Could not open output folder: {error}",
        "path_dir_error": "Select a valid {label} folder.",
        "path_file_error": "Select a valid {label} file.",
        "qwen_error": "Select a valid Qwen3 file or folder.",
        "lora_name_error": "Enter a LoRA name.",
        "sd_script_error": "anima_train_network.py was not found in the sd-scripts folder.",
        "cache_shuffle_error": "Anima cannot use cache text encoder and shuffle caption at the same time.",
        "cache_dit_error": "When cache text encoder is used, enable Train DiT only.",
        "cache_warmup_error": "Anima cannot use cache text encoder and token warmup at the same time.",
        "cache_tag_error": "Anima cannot use cache text encoder and tag dropout at the same time.",
        "overall_progress_initial": "Overall progress: -",
        "save_progress_initial": "Save interval: -",
        "overall_progress": "Overall progress: {current} / {total} steps",
        "save_progress_start": "Save interval: 1 / {slots}",
        "save_progress": "Save interval: {slot} / {slots}  {remaining} steps until next save",
    },
}


class AnimaLoraLauncher(ttk.Frame):
    def __init__(self, master: Tk) -> None:
        super().__init__(master, padding=12)
        self.master = master
        self.process: subprocess.Popen[bytes] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.last_command: list[str] = []
        self.last_run_dir: Path | None = None
        self.training_active = False
        self.stop_requested = False
        self.current_settings: RecommendedSettings | None = None
        self.current_stats = TrainingSetStats(0, 0, 0)
        self.save_interval_steps = 0
        self.total_save_slots = 0
        self.progress_visible = False

        self.english_ui = BooleanVar(value=False)
        self.sd_scripts_dir = StringVar()
        self.anima_model = StringVar()
        self.qwen3_model = StringVar()
        self.vae_model = StringVar()
        self.image_dir = StringVar()
        self.output_dir = StringVar()
        self.output_name = StringVar(value="anima_lora")
        self.vram_gb = IntVar(value=12)
        self.goal = StringVar(value=GOAL_LABELS["character"])
        self.image_resolution = IntVar(value=DEFAULT_IMAGE_RESOLUTION)
        self.stats_text = StringVar()

        self.resolution = IntVar(value=1024)
        self.train_batch_size = IntVar(value=1)
        self.num_repeats = IntVar(value=8)
        self.max_train_steps = IntVar(value=1200)
        self.network_dim = IntVar(value=32)
        self.network_alpha = IntVar(value=1)
        self.learning_rate = StringVar(value="2e-5")
        self.network_train_unet_only = BooleanVar(value=True)
        self.optimizer_type = StringVar(value="AdamW8bit")
        self.lr_scheduler = StringVar(value="constant")
        self.mixed_precision = StringVar(value="bf16")
        self.timestep_sampling = StringVar(value="sigmoid")
        self.discrete_flow_shift = StringVar(value="1.0")
        self.vae_chunk_size = IntVar(value=48)
        self.caption_extension = StringVar(value=".txt")
        self.keep_tokens = IntVar(value=1)
        self.caption_dropout_rate = DoubleVar(value=0.0)
        self.caption_tag_dropout_rate = DoubleVar(value=0.0)
        self.token_warmup_step = DoubleVar(value=0.0)
        self.seed = IntVar(value=42)
        self.save_every_n_epochs = IntVar(value=1)
        self.shuffle_caption = BooleanVar(value=False)
        self.gradient_checkpointing = BooleanVar(value=True)
        self.cache_latents = BooleanVar(value=True)
        self.cache_text_encoder_outputs = BooleanVar(value=True)
        self.vae_disable_cache = BooleanVar(value=True)
        self.train_llm_adapter = BooleanVar(value=False)
        self.advanced_visible = BooleanVar(value=False)
        self.overall_progress = DoubleVar(value=0.0)
        self.save_progress = DoubleVar(value=0.0)
        self.progress_text = StringVar()
        self.save_progress_text = StringVar()

        self._load_preferences()
        self._reset_static_text()
        self._build_ui()
        self.master.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_logs()

    def _language_code(self) -> str:
        return "en" if self.english_ui.get() else "ja"

    def _t(self, key: str, **values) -> str:
        text = UI_TEXT[self._language_code()].get(key, key)
        return text.format(**values) if values else text

    def _help(self, key: str) -> str:
        return (HELP_EN if self.english_ui.get() else HELP)[key]

    def _goal_label(self, key: str) -> str:
        if self.english_ui.get():
            return GOAL_LABELS_EN.get(key, GOAL_LABELS_EN["character"])
        return GOAL_LABELS.get(key, GOAL_LABELS["character"])

    def _goal_values(self) -> list[str]:
        return [self._goal_label(key) for key in GOAL_KEYS]

    def _reset_static_text(self) -> None:
        if self.current_stats.image_count:
            self._set_stats_text(self.current_stats)
        else:
            self.stats_text.set(self._t("stats_initial"))
        self.progress_text.set(self._t("overall_progress_initial"))
        self.save_progress_text.set(self._t("save_progress_initial"))

    def _load_preferences(self) -> None:
        data = load_user_settings()
        self.english_ui.set(data.get("language") == "en")
        string_vars = {
            "sd_scripts_dir": self.sd_scripts_dir,
            "anima_model": self.anima_model,
            "qwen3_model": self.qwen3_model,
            "vae_model": self.vae_model,
            "image_dir": self.image_dir,
            "output_dir": self.output_dir,
            "output_name": self.output_name,
        }
        for key, variable in string_vars.items():
            value = data.get(key)
            if isinstance(value, str):
                variable.set(value)

        vram = data.get("vram_gb")
        if isinstance(vram, int):
            self.vram_gb.set(vram)

        goal_key = data.get("goal_key")
        if isinstance(goal_key, str) and goal_key in GOAL_KEYS:
            self.goal.set(self._goal_label(goal_key))

    def _save_preferences(self, *, log: bool = False) -> None:
        data = {
            "language": self._language_code(),
            "sd_scripts_dir": self.sd_scripts_dir.get(),
            "anima_model": self.anima_model.get(),
            "qwen3_model": self.qwen3_model.get(),
            "vae_model": self.vae_model.get(),
            "image_dir": self.image_dir.get(),
            "output_dir": self.output_dir.get(),
            "output_name": self.output_name.get(),
            "vram_gb": self.vram_gb.get(),
            "goal_key": self._goal_key(),
        }
        try:
            saved_path = save_user_settings(data)
        except OSError as exc:
            if log:
                self._append_log(str(exc))
            return
        if log:
            self._append_log(self._t("log_saved_prefs", path=saved_path))

    def _rebuild_ui(self) -> None:
        goal_key = self._goal_key()
        self.goal.set(self._goal_label(goal_key))
        log_text = ""
        if hasattr(self, "log"):
            log_text = self.log.get("1.0", "end-1c")
        for child in self.winfo_children():
            child.destroy()
        self._reset_static_text()
        self._build_ui()
        if log_text:
            self.log.insert("end", log_text + "\n")
            self.log.see("end")
        self._set_training_controls(self.training_active)

    def change_language(self) -> None:
        self._save_preferences()
        self._rebuild_ui()

    def close(self) -> None:
        self._save_preferences()
        self.master.destroy()

    def _build_ui(self) -> None:
        self.master.title(self._t("app_title"))
        self.master.minsize(1100, 780)
        self.grid(sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(6, weight=1)

        language_bar = ttk.Frame(self)
        language_bar.grid(row=0, column=0, sticky="ew")
        language_toggle = ttk.Frame(language_bar)
        language_toggle.pack(side="right")
        ttk.Label(language_toggle, text=self._t("language_label")).pack(side="left", padx=(0, 6))
        ttk.Radiobutton(
            language_toggle,
            text=self._t("language_ja"),
            variable=self.english_ui,
            value=False,
            command=self.change_language,
            style="Toolbutton",
        ).pack(side="left")
        ttk.Radiobutton(
            language_toggle,
            text=self._t("language_en"),
            variable=self.english_ui,
            value=True,
            command=self.change_language,
            style="Toolbutton",
        ).pack(side="left")

        paths = ttk.LabelFrame(self, text=self._t("paths"), padding=10)
        paths.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, self._t("sd_scripts"), self.sd_scripts_dir, "dir")
        self._path_row(paths, 1, self._t("anima"), self.anima_model, "file")
        self._path_row(paths, 2, self._t("qwen3"), self.qwen3_model, "file")
        self._path_row(paths, 3, self._t("vae"), self.vae_model, "file")
        self._path_row(paths, 4, self._t("teacher_images"), self.image_dir, "dir")
        self._path_row(paths, 5, self._t("output_dir"), self.output_dir, "dir")

        basic = ttk.LabelFrame(self, text=self._t("basic_settings"), padding=10)
        basic.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        basic.columnconfigure(1, weight=1)
        basic.columnconfigure(5, weight=1)

        self._field(basic, 0, 0, self._t("lora_name"), self.output_name, self._help("lora_name"), width=24)
        self._combo_field(basic, 0, 2, self._t("vram"), self.vram_gb, [6, 8, 10, 12, 16, 20, 24, 32, 48], self._help("vram"), width=8)
        self._combo_field(basic, 0, 4, self._t("goal"), self.goal, self._goal_values(), self._help("goal"), width=14)

        recommend = ttk.LabelFrame(self, text=self._t("recommended_settings"), padding=10)
        recommend.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        recommend.columnconfigure(2, weight=1)
        ttk.Button(recommend, text=self._t("create_recommended"), command=self.apply_recommended).grid(row=0, column=0, sticky="w")
        self._label_with_help(recommend, self._t("teacher_images"), self._help("image_count")).grid(row=0, column=1, sticky="e", padx=(14, 4))
        ttk.Label(recommend, textvariable=self.stats_text).grid(row=0, column=2, sticky="w")

        visible_settings = ttk.LabelFrame(self, text=self._t("training_settings"), padding=10)
        visible_settings.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        visible_settings.columnconfigure(1, weight=1)
        visible_settings.columnconfigure(3, weight=1)
        self._setting_entry(visible_settings, 0, 0, self._t("train_resolution"), self.resolution, self._help("train_resolution"))
        self._setting_entry(visible_settings, 0, 2, self._t("steps"), self.max_train_steps, self._help("steps"))
        self._setting_entry(visible_settings, 1, 0, self._t("repeats"), self.num_repeats, self._help("repeats"))
        self._setting_entry(visible_settings, 1, 2, self._t("batch"), self.train_batch_size, self._help("batch"))
        self._setting_entry(visible_settings, 2, 0, self._t("rank"), self.network_dim, self._help("rank"))
        self._setting_entry(visible_settings, 2, 2, self._t("alpha"), self.network_alpha, self._help("alpha"))
        self._setting_entry(visible_settings, 3, 0, self._t("learning_rate"), self.learning_rate, self._help("lr"))
        self._setting_entry(visible_settings, 3, 2, self._t("save_epochs"), self.save_every_n_epochs, self._help("save"))

        advanced_toggle = ttk.Button(visible_settings, text=self._t("show_advanced"), command=self.toggle_advanced)
        advanced_toggle.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.advanced_toggle = advanced_toggle

        self.advanced_frame = ttk.LabelFrame(self, text=self._t("advanced_settings"), padding=10)
        self.advanced_frame.columnconfigure(1, weight=1)
        self.advanced_frame.columnconfigure(3, weight=1)
        self._setting_entry(self.advanced_frame, 0, 0, self._t("optimizer"), self.optimizer_type, self._help("optimizer"))
        self._setting_entry(self.advanced_frame, 0, 2, self._t("scheduler"), self.lr_scheduler, self._help("scheduler"))
        self._setting_entry(self.advanced_frame, 1, 0, self._t("precision"), self.mixed_precision, self._help("precision"))
        self._setting_entry(self.advanced_frame, 1, 2, self._t("timestep"), self.timestep_sampling, self._help("timestep"))
        self._setting_entry(self.advanced_frame, 2, 0, self._t("flow_shift"), self.discrete_flow_shift, self._help("flow"))
        self._setting_entry(self.advanced_frame, 2, 2, self._t("vae_chunk"), self.vae_chunk_size, self._help("vae_chunk"))
        self._setting_entry(self.advanced_frame, 3, 0, self._t("caption_ext"), self.caption_extension, self._help("caption_ext"))
        self._setting_entry(self.advanced_frame, 3, 2, self._t("keep_tokens"), self.keep_tokens, self._help("keep_tokens"))
        self._setting_entry(self.advanced_frame, 4, 0, self._t("seed"), self.seed, self._help("seed"))
        self._setting_entry(self.advanced_frame, 4, 2, self._t("caption_dropout"), self.caption_dropout_rate, self._help("caption_dropout"))
        self._setting_entry(self.advanced_frame, 5, 0, self._t("tag_dropout"), self.caption_tag_dropout_rate, self._help("tag_dropout"))
        self._setting_entry(self.advanced_frame, 5, 2, self._t("token_warmup"), self.token_warmup_step, self._help("token_warmup"))

        checks = ttk.Frame(self.advanced_frame)
        checks.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        check_items = [
            (self._t("dit_only"), self.network_train_unet_only, self._help("dit_only")),
            (self._t("cache_text"), self.cache_text_encoder_outputs, self._help("cache_text")),
            (self._t("cache_latents"), self.cache_latents, self._help("cache_latents")),
            (self._t("gradient_checkpointing"), self.gradient_checkpointing, self._help("gradient_checkpointing")),
            (self._t("vae_disable_cache"), self.vae_disable_cache, self._help("vae_disable_cache")),
            (self._t("shuffle_caption"), self.shuffle_caption, self._help("shuffle")),
            (self._t("train_llm_adapter"), self.train_llm_adapter, self._help("llm_adapter")),
        ]
        for index, (label, var, help_text) in enumerate(check_items):
            self._checkbutton(checks, index // 2, index % 2, label, var, help_text)

        if self.advanced_visible.get():
            self.advanced_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
            self.advanced_toggle.configure(text=self._t("hide_advanced"))

        body = ttk.PanedWindow(self, orient="horizontal")
        body.grid(row=6, column=0, sticky="nsew", pady=(10, 0))

        progress = ttk.LabelFrame(body, text=self._t("progress"), padding=10)
        progress.columnconfigure(0, weight=1)
        body.add(progress, weight=1)
        self.progress_label = ttk.Label(progress, textvariable=self.progress_text)
        self.progress_label.grid(row=0, column=0, sticky="w")
        self.overall_progress_bar = ttk.Progressbar(progress, variable=self.overall_progress, maximum=100)
        self.overall_progress_bar.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.save_progress_label = ttk.Label(progress, textvariable=self.save_progress_text)
        self.save_progress_label.grid(row=2, column=0, sticky="w")
        self.save_progress_bar_widget = ttk.Progressbar(progress, variable=self.save_progress, maximum=100)
        self.save_progress_bar_widget.grid(row=3, column=0, sticky="ew", pady=(4, 10))
        self._set_progress_widgets_visible(self.progress_visible)

        actions = ttk.Frame(progress)
        actions.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.start_button = ttk.Button(actions, text=self._t("start_training"), command=self.start_training)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text=self._t("stop"), command=self.stop_training, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text=self._t("open_output"), command=self.open_output_folder).pack(side="left", padx=(8, 0))

        log_frame = ttk.LabelFrame(body, text=self._t("log"), padding=8)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        body.add(log_frame, weight=2)
        self.log = ScrolledText(log_frame, height=18, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")

    def toggle_advanced(self) -> None:
        if self.advanced_visible.get():
            self.advanced_frame.grid_remove()
            self.advanced_visible.set(False)
            self.advanced_toggle.configure(text=self._t("show_advanced"))
        else:
            self.advanced_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
            self.advanced_visible.set(True)
            self.advanced_toggle.configure(text=self._t("hide_advanced"))

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: StringVar, kind: str) -> None:
        ttk.Label(parent, text=label, width=16).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=4, sticky="ew", padx=(4, 6), pady=3)
        command = (lambda: self._choose_dir(variable)) if kind == "dir" else (lambda: self._choose_file(variable))
        ttk.Button(parent, text=self._t("choose"), command=command).grid(row=row, column=5, sticky="e", pady=3)

    def _field(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        variable: StringVar | IntVar | DoubleVar,
        help_text: str,
        width: int = 16,
    ) -> None:
        self._label_with_help(parent, label, help_text).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=4)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=column + 1, sticky="ew", padx=(4, 12), pady=4)

    def _combo_field(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        variable: StringVar | IntVar,
        values: list[str] | list[int],
        help_text: str,
        width: int = 12,
    ) -> None:
        self._label_with_help(parent, label, help_text).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=4)
        ttk.Combobox(parent, textvariable=variable, values=values, width=width).grid(
            row=row, column=column + 1, sticky="w", padx=(4, 12), pady=4
        )

    def _setting_entry(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        variable: StringVar | IntVar | DoubleVar,
        help_text: str,
    ) -> None:
        self._label_with_help(parent, label, help_text).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=column + 1, sticky="ew", pady=3)

    def _checkbutton(self, parent: ttk.Frame, row: int, column: int, label: str, variable: BooleanVar, help_text: str) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky="w", padx=(0, 18), pady=3)
        ttk.Checkbutton(frame, text=label, variable=variable).pack(side="left")
        self._help_icon(frame, help_text).pack(side="left", padx=(4, 0))

    def _label_with_help(self, parent: ttk.Frame, label: str, help_text: str) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=label).pack(side="left")
        self._help_icon(frame, help_text).pack(side="left", padx=(4, 0))
        return frame

    def _help_icon(self, parent: ttk.Frame, text: str) -> ttk.Label:
        icon = ttk.Label(parent, text="?", width=2, anchor="center", relief="groove")
        ToolTip(icon, text)
        return icon

    def _choose_dir(self, variable: StringVar) -> None:
        value = filedialog.askdirectory()
        if value:
            variable.set(value)
            self._save_preferences()

    def _choose_file(self, variable: StringVar) -> None:
        value = filedialog.askopenfilename(
            filetypes=[
                ("Model files", "*.safetensors *.ckpt *.pt *.pth"),
                ("All files", "*.*"),
            ]
        )
        if value:
            variable.set(value)
            self._save_preferences()

    def _set_stats_text(self, stats: TrainingSetStats) -> None:
        size_text = self._t("stats_no_size")
        if stats.typical_max_side:
            size_text = self._t(
                "stats_size",
                typical=stats.typical_max_side,
                width=stats.max_width,
                height=stats.max_height,
            )
        self.stats_text.set(
            self._t(
                "stats_text",
                images=stats.image_count,
                captions=stats.caption_count,
                missing=stats.missing_caption_count,
                size=size_text,
            )
        )

    def refresh_stats(self) -> None:
        image_dir = self.image_dir.get().strip()
        if not image_dir:
            self.current_stats = TrainingSetStats(0, 0, 0)
            self.image_resolution.set(DEFAULT_IMAGE_RESOLUTION)
            self.stats_text.set(self._t("stats_missing"))
            self._append_log(self._t("log_no_image_dir"))
            return

        self.current_stats = count_training_set(Path(image_dir), self.caption_extension.get())
        stats = self.current_stats
        if stats.typical_max_side:
            self.image_resolution.set(nearest_resolution(stats.typical_max_side))
        else:
            self.image_resolution.set(DEFAULT_IMAGE_RESOLUTION)
        self._set_stats_text(stats)
        self._append_log(
            self._t(
                "log_stats",
                images=stats.image_count,
                captions=stats.caption_count,
                missing=stats.missing_caption_count,
                resolution=self.image_resolution.get(),
            )
        )

    def apply_recommended(self) -> None:
        self.refresh_stats()
        self._save_preferences()
        stats = self.current_stats
        settings = recommend_settings(
            vram_gb=self.vram_gb.get(),
            image_count=stats.image_count,
            caption_count=stats.caption_count,
            goal=self._goal_key(),
            image_resolution=self.image_resolution.get(),
        )
        self._apply_settings(settings)
        self._append_log(self._t("log_recommended", epochs=settings.estimated_epochs))
        for note in settings.notes:
            self._append_log(self._t("log_note", note=note))

    def _apply_settings(self, settings: RecommendedSettings) -> None:
        self.resolution.set(settings.resolution)
        self.train_batch_size.set(settings.train_batch_size)
        self.num_repeats.set(settings.num_repeats)
        self.max_train_steps.set(settings.max_train_steps)
        self.network_dim.set(settings.network_dim)
        self.network_alpha.set(settings.network_alpha)
        self.learning_rate.set(settings.learning_rate)
        self.network_train_unet_only.set(settings.network_train_unet_only)
        self.optimizer_type.set(settings.optimizer_type)
        self.lr_scheduler.set(settings.lr_scheduler)
        self.mixed_precision.set(settings.mixed_precision)
        self.timestep_sampling.set(settings.timestep_sampling)
        self.discrete_flow_shift.set(settings.discrete_flow_shift)
        self.vae_chunk_size.set(settings.vae_chunk_size)
        self.caption_extension.set(settings.caption_extension)
        self.keep_tokens.set(settings.keep_tokens)
        self.caption_dropout_rate.set(settings.caption_dropout_rate)
        self.caption_tag_dropout_rate.set(settings.caption_tag_dropout_rate)
        self.token_warmup_step.set(settings.token_warmup_step)
        self.seed.set(settings.seed)
        self.save_every_n_epochs.set(settings.save_every_n_epochs)
        self.shuffle_caption.set(settings.shuffle_caption)
        self.gradient_checkpointing.set(settings.gradient_checkpointing)
        self.cache_latents.set(settings.cache_latents)
        self.cache_text_encoder_outputs.set(settings.cache_text_encoder_outputs)
        self.vae_disable_cache.set(settings.vae_disable_cache)
        self.train_llm_adapter.set(settings.train_llm_adapter)

    def create_configs(self) -> tuple[Path, Path] | None:
        try:
            self._validate_required_paths()
            self.refresh_stats()
            self._save_preferences()
            settings = self._settings_from_form()
            self.current_settings = settings
            self._prepare_progress(settings)
            output_dir = Path(self.output_dir.get())
            run_name = f"{self.output_name.get()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            run_dir = output_dir / "_anima_lora_launcher_runs" / run_name
            dataset_config, train_config = write_configs(
                run_dir=run_dir,
                sd_scripts_dir=Path(self.sd_scripts_dir.get()),
                anima_model=Path(self.anima_model.get()),
                qwen3_model=Path(self.qwen3_model.get()),
                vae_model=Path(self.vae_model.get()),
                image_dir=Path(self.image_dir.get()),
                output_dir=output_dir,
                output_name=self.output_name.get(),
                settings=settings,
            )
        except Exception as exc:
            messagebox.showerror(self._t("error_title"), str(exc))
            return None

        self.last_run_dir = run_dir
        self.last_command = build_training_command(Path(self.sd_scripts_dir.get()), train_config)
        self._append_log(self._t("log_config_created", run_dir=run_dir))
        self._append_log(self._t("log_command"))
        self._append_log(command_to_text(self.last_command))
        return dataset_config, train_config

    def start_training(self) -> None:
        if self.training_active:
            messagebox.showwarning(self._t("running_title"), self._t("running_message"))
            return

        configs = self.create_configs()
        if configs is None:
            return

        self._append_log(self._t("log_start"))
        self.stop_requested = False
        self.process = popen_training(sd_scripts_dir=Path(self.sd_scripts_dir.get()), command=self.last_command)
        self.training_active = True
        self._set_training_controls(True)
        self._append_log(self._t("log_pid", pid=self.process.pid))
        thread = threading.Thread(target=self._read_process_output, daemon=True)
        thread.start()

    def stop_training(self) -> None:
        if not self.process or not self.training_active:
            self._append_log(self._t("log_stop_none"))
            return

        self.stop_requested = True
        self._append_log(self._t("log_stop", pid=self.process.pid))
        ok, output = stop_process_tree(self.process)
        if ok:
            self._append_log(self._t("log_stop_ok"))
        else:
            self._append_log(self._t("log_stop_fail"))
            if output:
                for line in output.splitlines():
                    self._append_log(line)

    def open_output_folder(self) -> None:
        output_dir = Path(self.output_dir.get().strip())
        if not output_dir.exists() or not output_dir.is_dir():
            messagebox.showerror(self._t("error_title"), self._t("invalid_output"))
            return
        try:
            open_folder(output_dir)
        except OSError as exc:
            messagebox.showerror(self._t("error_title"), self._t("open_output_error", error=exc))
            return
        self._save_preferences()
        self._append_log(self._t("log_open_output", path=output_dir))

    def copy_command(self) -> None:
        if not self.last_command:
            self.create_configs()
        if self.last_command:
            self.master.clipboard_clear()
            self.master.clipboard_append(command_to_text(self.last_command))
            self._append_log("コマンドをクリップボードにコピーしました。")

    def _settings_from_form(self) -> RecommendedSettings:
        settings = settings_from_form(
            {
                "resolution": self.resolution.get(),
                "train_batch_size": self.train_batch_size.get(),
                "num_repeats": self.num_repeats.get(),
                "max_train_steps": self.max_train_steps.get(),
                "network_dim": self.network_dim.get(),
                "network_alpha": self.network_alpha.get(),
                "learning_rate": self.learning_rate.get(),
                "network_train_unet_only": self.network_train_unet_only.get(),
                "optimizer_type": self.optimizer_type.get(),
                "lr_scheduler": self.lr_scheduler.get(),
                "mixed_precision": self.mixed_precision.get(),
                "timestep_sampling": self.timestep_sampling.get(),
                "discrete_flow_shift": self.discrete_flow_shift.get(),
                "vae_chunk_size": self.vae_chunk_size.get(),
                "caption_extension": self.caption_extension.get(),
                "shuffle_caption": self.shuffle_caption.get(),
                "keep_tokens": self.keep_tokens.get(),
                "caption_dropout_rate": self.caption_dropout_rate.get(),
                "caption_tag_dropout_rate": self.caption_tag_dropout_rate.get(),
                "token_warmup_step": self.token_warmup_step.get(),
                "seed": self.seed.get(),
                "save_every_n_epochs": self.save_every_n_epochs.get(),
                "gradient_checkpointing": self.gradient_checkpointing.get(),
                "cache_latents": self.cache_latents.get(),
                "cache_text_encoder_outputs": self.cache_text_encoder_outputs.get(),
                "vae_disable_cache": self.vae_disable_cache.get(),
                "train_llm_adapter": self.train_llm_adapter.get(),
            }
        )
        return validate_settings(settings)

    def _validate_required_paths(self) -> None:
        required_dirs = {
            "sd-scripts": self.sd_scripts_dir.get(),
            self._t("teacher_images"): self.image_dir.get(),
            self._t("output_dir"): self.output_dir.get(),
        }
        required_files = {
            "Anima": self.anima_model.get(),
            "VAE": self.vae_model.get(),
        }
        qwen3_path = Path(self.qwen3_model.get())

        for label, value in required_dirs.items():
            path = Path(value)
            if not value or not path.exists() or not path.is_dir():
                raise ValueError(self._t("path_dir_error", label=label))

        for label, value in required_files.items():
            path = Path(value)
            if not value or not path.exists() or not path.is_file():
                raise ValueError(self._t("path_file_error", label=label))

        if not self.qwen3_model.get() or not qwen3_path.exists():
            raise ValueError(self._t("qwen_error"))

        if not self.output_name.get().strip():
            raise ValueError(self._t("lora_name_error"))

        if not (Path(self.sd_scripts_dir.get()) / "anima_train_network.py").exists():
            raise ValueError(self._t("sd_script_error"))

    def _goal_key(self) -> str:
        value = self.goal.get()
        return GOAL_LABEL_TO_KEY.get(value, value)

    def _prepare_progress(self, settings: RecommendedSettings) -> None:
        image_count = max(1, self.current_stats.image_count)
        steps_per_epoch = max(1, ceil(image_count * settings.num_repeats / max(1, settings.train_batch_size)))
        self.save_interval_steps = max(1, steps_per_epoch * max(1, settings.save_every_n_epochs))
        self.total_save_slots = max(1, ceil(settings.max_train_steps / self.save_interval_steps))
        self.overall_progress.set(0)
        self.save_progress.set(0)
        self.progress_text.set(self._t("overall_progress_initial"))
        self.save_progress_text.set(self._t("save_progress_initial"))
        self._set_progress_widgets_visible(False)

    def _set_progress_widgets_visible(self, visible: bool) -> None:
        self.progress_visible = visible
        widgets = (
            "progress_label",
            "overall_progress_bar",
            "save_progress_label",
            "save_progress_bar_widget",
        )
        for name in widgets:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _read_process_output(self) -> None:
        if not self.process or not self.process.stdout:
            return
        process = self.process
        try:
            while True:
                read_chunk = getattr(process.stdout, "read1", process.stdout.read)
                chunk = read_chunk(4096)
                if not chunk:
                    break
                for line in split_process_text(decode_process_output(chunk)):
                    self.log_queue.put(line)
            code = process.wait()
            if self.stop_requested:
                self.log_queue.put(self._t("log_process_stopped", code=code))
            else:
                self.log_queue.put(self._t("log_process_finished", code=code))
        except Exception as exc:
            self.log_queue.put(self._t("log_read_error", error=exc))
        finally:
            self.training_active = False
            self.stop_requested = False
            self.log_queue.put("__TRAINING_FINISHED__")

    def _poll_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line == "__TRAINING_FINISHED__":
                self._set_training_controls(False)
                continue
            self._update_progress_from_line(line)
            self._append_log(line)
        self.after(120, self._poll_logs)

    def _update_progress_from_line(self, line: str) -> None:
        match = parse_step_progress(line)
        if not match:
            return
        current, total = match
        if total <= 0:
            return
        self._set_progress_widgets_visible(True)
        self.overall_progress.set(min(100.0, current * 100 / total))
        self.progress_text.set(self._t("overall_progress", current=current, total=total))
        if self.save_interval_steps:
            slot = min(self.total_save_slots, current // self.save_interval_steps + 1)
            slot_start = (slot - 1) * self.save_interval_steps
            slot_end = min(total, slot * self.save_interval_steps)
            span = max(1, slot_end - slot_start)
            self.save_progress.set(min(100.0, max(0, current - slot_start) * 100 / span))
            remaining = max(0, slot_end - current)
            self.save_progress_text.set(self._t("save_progress", slot=slot, slots=self.total_save_slots, remaining=remaining))

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{timestamp}] {text}\n")
        self.log.see("end")

    def _set_training_controls(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")


class ToolTip:
    def __init__(self, widget: ttk.Label, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: Toplevel | None = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 18
        self.tip = Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tip, text=self.text, padding=8, relief="solid", wraplength=360)
        label.pack()

    def hide(self, _event=None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def validate_settings(settings: RecommendedSettings) -> RecommendedSettings:
    if settings.train_llm_adapter:
        settings = replace(settings, cache_text_encoder_outputs=False, network_train_unet_only=False)
    if settings.cache_text_encoder_outputs and settings.shuffle_caption:
        raise ValueError("Animaでは cache text encoder と shuffle caption を同時に使えません。")
    if settings.cache_text_encoder_outputs and not settings.network_train_unet_only:
        raise ValueError("Animaでは cache text encoder を使う場合、DiTのみ学習をオンにしてください。")
    if settings.cache_text_encoder_outputs and settings.token_warmup_step > 0:
        raise ValueError("Animaでは cache text encoder と token warmup を同時に使えません。")
    if settings.cache_text_encoder_outputs and settings.caption_tag_dropout_rate > 0:
        raise ValueError("Animaでは cache text encoder と tag dropout を同時に使えません。")
    return settings


def nearest_resolution(value: int) -> int:
    choices = [768, 896, 1024, 1280, 1536]
    return min(choices, key=lambda item: abs(item - value))


def parse_step_progress(line: str) -> tuple[int, int] | None:
    if "steps:" not in line:
        return None
    match = re.search(r"(\d+)\s*/\s*(\d+)", line)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def command_to_text(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if not value:
        return '""'
    if any(char.isspace() for char in value) or any(char in value for char in ['"', "&", "(", ")"]):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def split_process_text(text: str) -> list[str]:
    parts = text.replace("\r", "\n").splitlines()
    return [part for part in parts if part.strip()]


def open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)])


def main() -> None:
    root = Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    AnimaLoraLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
