# jsai2016_roboken2

## バージョンアップ
jsai_dialogue_small.py は LSTM 層が 1 層だけのモデルです。

Guglecus.jpg はローマ時代の賢者，ググレカスの肖像です。Wikipedia より。プレゼンでのウケ狙い？

jsai_baseline.py はパープレキシティを修正してみました。説明としては対話モデルとの対比のために
全部のパープレキシティを合算してますが，付け刃的に A/(Q+A) していますのでパープレキシティが小さく出ます。


関係者の皆様。我々のプロジェクトに関係するファイルを公開します。

# requirements
python 2.7
chainer

まず自分の実行環境を把握してくださいね。エラーの原因が推測できませんよ。
```shell
python -c 'import chainer; priint chainr.__version__'
```
すると Chainer のバージョンが分かります。必ず最新バージョンにしてください。
Anaconda も最新バージョンにしてください。普及率の高い邪悪な OS みたいにアップデートすると動かなくなるといことはありませんので，安心してアップデートしてください。以下にサンプルオペレーションを示します。
```shell
conda update conda
conda update anaconda
conda uddata --all
```
次に pip をアップデートします。
```shell
pip instal --upgrade pip
```
その後，Chainer をアップデートします。Theano は上の conda のアップデートで最新版になっているはず。
```shell
pip install --upgrade chainer
```
途中で setuptools のアップデートでコケることがありますが，慌てずもう一度
```shell
pip install --upgrade chainer 
```
すればなんとかなります。

# Data

all_mlq20160522.txt がオリジナルなスクレープした元データです。このデータを unk に置き換えたデータが
jsai_unk[0-5].{train,valid,test}.data になります。
unk の後の数字が unk のしきい値になります。unk0 は unk なし，すなわちまったく未知語なしです。反対に unk5 は頻度5以下の単語を unk トークンに置き換えたデータです。

これらのデータを作成したスクリプトが
jsai_make_dataset.sh です。ただしこのスクリプトを実行する必要はありません。あくまで，データ作成はどのようにしたのかを記録し，再現可能性を保証するためだけにあります。

# 2 つモデル
2つのモデルに対応した python scripts が jsai_baseline.py と jsai_dialogue.py です。前者がベースラインモデルで後者が対話モデルになります。

# 起動と終了

実行する際にはデータファイルを引数で指定する必要があります。--train, --valid, --test で，訓練データ，検証データ，テストデータを指定します。
オプションを何も指定しないと unk=0 すなわち未知語なしのデータを学習します。
```bash
python jsai_dialogue.py 
```
途中で終了する場合にはコントロールキーを押しながら C を押します。するとその時点までの学習結果を元にテストデータセットを評価し，結果を吐き出して
から終了します。
終了するときに jsai_baseline まはた jsai_dialogue を接頭にしてその時の時刻が続いたファイル名でモデルと状態ファイルを保存します。

もちろん，終了するまで待っていればこれらのファイルは生成されます。




また --help を指定すると，指定可能なオプションが表示されます。
```bash
python jsai_dialogue.py --help
usage: jsai_dialogue.py [-h] [--initmodel INITMODEL] [--initmodelQ INITMODELQ]
                        [--initmodelA INITMODELA] [--resume RESUME]
                        [--savefile SAVEFILE] [--gpu GPU] [--epoch EPOCH]
                        [--unit UNIT] [--batchsize BATCHSIZE]
                        [--bproplen BPROPLEN] [--gradclip GRADCLIP]
                        [--train TRAIN] [--valid VALID] [--test TEST]
                        [--lr LR] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --initmodel INITMODEL, -m INITMODEL
                        Initialize the model from given file
  --initmodelQ INITMODELQ, -q INITMODELQ
                        Initialize the modelQ from given file
  --initmodelA INITMODELA, -a INITMODELA
                        Initialize the modelA from given file
  --resume RESUME, -r RESUME
                        Resume the optimization from snapshot
  --savefile SAVEFILE, -s SAVEFILE
                        File name sufix to be saved
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --epoch EPOCH, -e EPOCH
                        number of epochs to learn
  --unit UNIT, -u UNIT  number of units
  --batchsize BATCHSIZE, -b BATCHSIZE
                        learning minibatch size
  --bproplen BPROPLEN, -l BPROPLEN
                        length of truncated BPTT
  --gradclip GRADCLIP, -c GRADCLIP
                        gradient norm threshold to clip
  --train TRAIN, -t TRAIN
                        train data file name
  --valid VALID, -v VALID
                        valid data file name
  --test TEST, -x TEST  test data file name
  --lr LR, -z LR        learning ratio a hyperparameter
  --seed SEED, -S SEED  seed of random number generator
  ```

このうち --seed オプションは重要です。同一の結果を得るために，乱数発生器の種 シードは決め打ちにしてあります。
従って，別の結果を得たければ別のシードを整数で指定してください。

以下簡単にオプション引数を説明します。

1. --initmodel 途中から始める場合に指定するモデルファイルです
2. --initmodelQ
3. --initmodelA 2 と 3 は対話モデルでしか使いません
4. --resume 途中から再開するスィッチです
5. --savefile セーブするファイル名の接頭辞です。
6. --epoch 学習終了までの総エポック数です
7. --unit 層内のユニット数です。デフォルトは 200 ですが，PC などでメモリが足りなければ少なくしてください。10 でも動作します。
8. --batchsize ミニバッチのサイズです
9. --bproplen LSTM の系列学習における系列長を指定します。長ければ長いほど性能は上がりますが学習もメモリも必要です
10. --gradclip 勾配爆発問題に対処するためのクリップの上限値
11. --train 訓練データセット名. デフォルトは jsai_unk0.train.data
12. --valid 検証データセット名. デフォルトは jsai_unk0.valid.data
13. --test テストデータセット名. デフォルトは jsai_unk0.test.data
14. --lr 学習率 デフォルトでは 0.1 にしてあります。急ぐなら 1.0 でも動作します
15. --seed 乱数の種

テスト的に実行するならデータ数が少ない方が良いので訓練データの代わりにデータ数の少ない検証データやテストデータを訓練データとして使う
方が勝負が早いです。例えば

```bash
python jsai_baseline.py --train jsai_unk0.valid.data --valid jsai_unk0.valid.data --test jsai_unk0.test.data --seed 3 --unit 10
```
のようにします。上例は一層内のユニット数が 10 だけなので終了も早いです。

# 実行例
```base
python jsai_baseline.py --train jsai_unk2.valid.data --test jsai_unk2.valid.data --valid jsai_unk2.valid.data --unit 10 
### train_data has  3771  words in this corpus.
### valid_data has  3771  words in this corpus.
### test_data has  3771  words in this corpus.
### vocab=691
### whole_len=3771
### n_epoch = 100
### n_units = 10
### batchsize = 1000
### bprop_len = 200.000000
### grad_clip = 1.000000
### save filename prefix = jsai_baseline
### learning ratio = 0.100000
### whole_len= 3771
### batchsize= 1000   # length of a minibatch
### jump= 3   # interval between minibatches
### going to train 300 iterations
iter 3 training perplexity: 694.519032 (16.790917 iters/sec)
iter 3 validation perplexity: 692.695520
iter 6 training perplexity: 694.054443 (31.921336 iters/sec)
iter 6 validation perplexity: 692.741501
iter 9 training perplexity: 694.311529 (33.204851 iters/sec)
iter 9 validation perplexity: 692.876499
iter 12 training perplexity: 694.085553 (29.397679 iters/sec)
iter 12 validation perplexity: 692.856240
iter 15 training perplexity: 693.972151 (31.209090 iters/sec)
iter 15 validation perplexity: 692.820403
iter 18 training perplexity: 693.911817 (27.828573 iters/sec)
iter 18 validation perplexity: 692.738682
learning rate = 0.0999000999001
iter 21 training perplexity: 693.780579 (29.984658 iters/sec)
iter 21 validation perplexity: 692.814926
learning rate = 0.0998002996005
iter 24 training perplexity: 693.712654 (30.207519 iters/sec)
iter 24 validation perplexity: 692.716460
^CTraining interupted
save the model
save the optimizer
test
test perplexity: 692.859615994
```


上例のように繰り返し3回ごとに訓練パープレキシティと検証パープレキシティを出力していきます。
途中を省略しましたが，コントロール＋C で強制終了させたため訓練が中断された(Training interupted)と表示され，その後
モデルを保存しています。

簡単ですが説明は以上です。
皆様のご協力に感謝いたします。

公開が遅れましたことをお詫びいたします。
