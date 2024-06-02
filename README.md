# Reversi-AlphaZero
## AlphaZeroの機械学習アルゴリズムを参考に作成したオセロAIと対戦するゲーム
  
* Google Colabで機械学習を行い学習データを作成する
```python
# ソースコード一式アップロード
from google.colab import files
uploaded = files.upload()

# 学習サイクル実行
!python TrainCycle.py

# 学習が完了したら学習データ(best.h5)をダウンロードする
from google.colab import files
files.download('./model/best.h5')
```

* HumanPlay.pyを実行することでAIと対戦できます。  
```bash
$ python HumanPlay.py
```
  
  
## ソースコード一覧 
### Game.py
* オセロの基本ルール設定
### ResidualNetwork.py
* Residual Network(ResNet)の構築
### MonteCarloTreeSearch.py
* モンテカルロ木探索プログラム
### SelfPlay.py
* 過去最強のプレイヤー同士で対戦させ学習データを作成する
### TrainNetwork.py
`SelfPlay.py`で作成された学習データを使ったResidual Network(ResNet)での学習
### EvaluateNetwork.py
* 最新プレイヤーと過去最強のプレイヤーを対戦させ強い方を残すプログラム
### TrainCycle.py
* 全てのスクリプトを合わせた学習サイクルの構築
### HumanPlay.py
* プログラムで作成されたAIとの対戦

## model
学習データ(best.h5)を`./model`に保存

# 将来の改善点
## 効率的な駒のカウント
現在、各プレイヤーの駒の数は、それぞれのリスト内の1の数を合計することで決定されています。この方法は正しく動作しますが、最適化する余地があります。今後の更新では以下の方法を計画しています：

1プレイヤーの駒の数を計算します。
空のマスの数と合計マス数（64）から、この数を引くことで相手の駒の数を取得します。
この方法により、駒の数をカウントするための計算量を減らし、ゲームの状態評価の効率を向上させることができます。
