# Anima LoRA Config Launcher

  

  

Anima Base v1 LoRA向けの設定ファイルを作成し、`kohya-ss/sd-scripts` を起動する小さなGUIツールです。

  

  

このアプリ自体はLoRA学習を行いません。インストール済みの `sd-scripts` とAnima用モデルを使って、`anima_train_network.py` を実行します。

  

  

## 必須

  

  

- Windows

  

- Python 3.10+

  

- インストール済みの `kohya-ss/sd-scripts`

  

- Anima Base v1 / Qwen3 / VAE のモデルファイル

  

- 画像と `.txt` caption が入った教師画像フォルダ

  

  

追加のPythonパッケージは不要です。

  

  

## 導入

  

  

GitHubの `Code` -> `Download ZIP` からダウンロードして、ZIPを展開してください。

  

  

## 起動方法

  

  
  
  

run_app.bat

を実行

  

  

  

## 使い方

  

  

1. `sd-scripts`、Anima、Qwen3、VAE、教師画像、出力先のパスを指定する

  

2. LoRA名、VRAM、目的を選ぶ

  

3. `おすすめ設定を作成する` を押す

  

4. （必要なら設定を調整する）

  

5. `学習開始` を押す

  

  

`学習開始` を押すと、学習実行の作業をまとめて行います。

  

  
  
  

  

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

A small GUI tool for creating LoRA training config files for Anima Base v1 and launching `kohya-ss/sd-scripts`.

This app does not train LoRA by itself. It uses your installed `sd-scripts` and Anima model files to run `anima_train_network.py`.

## Requirements

- Windows
- Python 3.10+
- Installed `kohya-ss/sd-scripts`
- Anima Base v1 / Qwen3 / VAE model files
- A training image folder containing images and `.txt` captions

No additional Python packages are required.

## Install

Download this repository from GitHub using `Code` -> `Download ZIP`, then extract the ZIP file.

## Run

Run `run_app.bat`.

## Usage

1. Set the paths for `sd-scripts`, Anima, Qwen3, VAE, training images, and the output folder.
2. Enter the LoRA name, VRAM, and purpose.
3. Click `Create Recommended Settings`.
4. Adjust the settings if needed.
5. Click `Srart Training`.

When `Srart Training` is clicked, the tool handles the training launch steps together.

## References

- https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md
- https://huggingface.co/circlestone-labs/Anima

## Disclaimer

This is an unofficial helper tool. It is not affiliated with the developers of `kohya-ss/sd-scripts` or Anima Base v1.

This tool helps create LoRA training settings and run commands. It does not guarantee training results or output quality.

Please check the licenses and terms of use for any models, training images, generated images, and trained LoRAs. Use them at your own responsibility.

The author is not responsible for any damage or trouble caused by using this tool.
