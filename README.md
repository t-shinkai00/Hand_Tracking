# Hand_Tracking

## 概要

[MediaPipe](https://google.github.io/mediapipe/solutions/hands.html)を使用して、カメラ画像から手の追跡を行います。
手首の座標にはかざされた手が右手か左手かを識別して表示します。
また画像の左上には fps が表示されています。
![](img/README_image.png)

## 使用手順

### パッケージのインストール

`pip install -r requirements.txt`

> mediapipe のインストールは各々の環境に合わせて適切に行ってください。
> `pip install mediapipe`では入らないみたいです。

### 実行

`python main.py`
