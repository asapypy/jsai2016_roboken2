# jsai2016_roboken2
ロボケン関係者の皆様

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
