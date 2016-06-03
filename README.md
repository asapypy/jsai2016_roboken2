# jsai2016_roboken2
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
jsai_make_dataset.sh です。

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
テスト的に実行するならデータ数が少ない方が良いので訓練データの代わりにデータ数の少ない検証データやテストデータを訓練データとして使う
方が勝負が早いです。例えば

```bash
python jsai_baseline.py --train jsai_unk0.valid.data --valid jsai_unk0.valid.data --test jsai_unk0.test.data --seed 3 --unit 10
```
のようにします。上例は一層内のユニット数が 10 だけなので終了も早いです。

