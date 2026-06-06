# Anima LoRA Config Launcher

  

  

  

Anima Base v1 LoRA向けの設定ファイルを作成し、`kohya-ss/sd-scripts` を起動するGUIツールです。

  

  

  

このアプリ自体はLoRA学習を行いません。インストール済みの `sd-scripts` とAnima用モデルを使って、キャプション作成・編集、`anima_train_network.py` の実行をします。

  

  

  

## 必須

  

  

  

- Windows

  

  

- Python 3.10+

  

  

- 学習環境構築済みの `kohya-ss/sd-scripts`

  

  

- Anima Base v1 / Qwen3 / VAE のモデルファイル

  

  

- 教師画像

  

  

  
  

  

  

## 導入

  

  

  

GitHubの `Code` -> `Download ZIP` からダウンロードして、ZIPを展開してください。

  

  

  

## 起動方法

  

  

  

run_app.bat

  

を実行

  

  

  

  

## 使い方

  

  

  

1. `sd-scripts`、Anima、Qwen3、VAE、教師画像、出力先のパスを指定する

  

  

2. LoRA名、VRAM、目的を選ぶ

  

  

3. キャプション作成・編集（※ onnxをインストールしていなくてキャプション作成ボタンが機能しない場合は後述を参照してください。）

  

  

4. `おすすめ設定を作成する` を押す（必要なら設定を調整する）

  

  

5. `学習開始` を押す

  

  

  

`学習開始` を押すと、学習実行の作業をまとめて行います。

  

  

※

キャプション作成には WD14 Tagger の ONNX 版を使用します。

キャプション作成を行う場合は、以下を実行してsd-scriptsを実行しているPython環境に

`onnx` と `onnxruntime` をインストールしてください。

  

venvを使用している場合(通常はこちらです):

  

sd-scriptsのあるフォルダ\venv\Scripts\python.exe -m pip install onnx onnxruntime-gpu

  

venvを使用していない場合:

  

python -m pip install onnx onnxruntime-gpu

  

例

```

C:\sd-scripts\venv\Scripts\python.exe -m pip install onnx onnxruntime-gpu

```

  

  

  

## References

  

  

  

- https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md

  

  

- https://huggingface.co/circlestone-labs/Anima

  

  
  
  

  

- ## 免責事項

  

  

このツールは非公式の補助ツールです。kohya-ss/sd-scripts および Anima Base v1 の開発元とは関係ありません。

  

  

本ツールはLoRA学習用の設定作成とコマンド実行を補助するものであり、学習結果や生成物の品質を保証するものではありません。

  

  

モデル、教師画像、生成物、学習済みLoRAの利用については、各モデルや素材のライセンス・利用規約を確認し、利用者の責任で行ってください。

  

  

本ツールの使用によって発生した損害やトラブルについて、作者は責任を負いません。

---

## English

# Anima LoRA Config Launcher

Anima LoRA Config Launcher is a GUI tool for creating LoRA training config files for Anima Base v1 and launching `kohya-ss/sd-scripts`.

This app does not train LoRA by itself. It uses your installed `sd-scripts` and Anima model files to create and edit captions, then run `anima_train_network.py`.

## Requirements

- Windows
- Python 3.10+
- A working `kohya-ss/sd-scripts` training environment
- Anima Base v1 / Qwen3 / VAE model files
- Training images

## Install

Download this repository from GitHub using `Code` -> `Download ZIP`, then extract the ZIP file.

## Run

Run:

```cmd
run_app.bat
```

## Usage

1. Set the paths for `sd-scripts`, Anima, Qwen3, VAE, training images, and the output folder.
2. Enter the LoRA name, VRAM, and purpose.
3. Create and edit captions. If the caption creation button does not work because ONNX is not installed, see the note below.
4. Click `Create Recommended Settings`, then adjust the settings if needed.
5. Click `Start Training`.

When `Start Training` is clicked, the tool handles the training launch steps together.

Note:

Caption creation uses the ONNX version of WD14 Tagger.

To use caption creation, install `onnx` and `onnxruntime` into the Python environment used by `sd-scripts`.

If you use venv, which is the usual setup:

```cmd
<sd-scripts folder>\venv\Scripts\python.exe -m pip install onnx onnxruntime-gpu
```

If you do not use venv:

```cmd
python -m pip install onnx onnxruntime-gpu
```

Example:

```cmd
C:\sd-scripts\venv\Scripts\python.exe -m pip install onnx onnxruntime-gpu
```

## References

- https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md
- https://huggingface.co/circlestone-labs/Anima

## Disclaimer

This is an unofficial helper tool. It is not affiliated with the developers of `kohya-ss/sd-scripts` or Anima Base v1.

This tool helps create LoRA training settings and run commands. It does not guarantee training results or output quality.

Please check the licenses and terms of use for any models, training images, generated images, and trained LoRAs. Use them at your own responsibility.

The author is not responsible for any damage or trouble caused by using this tool.
