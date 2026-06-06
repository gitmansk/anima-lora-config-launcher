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