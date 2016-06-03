#!/bin/sh

rawdata=all_mlq20160522.txt
suffix=jsai

tail=/usr/bin/tail
head=/usr/bin/head
sed=/usr/bin/sed
sed_script=./jsai_delmarkers.sed

python=/Users/asakawa/anaconda2/bin/python
python_setunk=./jsai_setUNK.py
python_reform=./jsai_reform.py

${python} ${python_reform} --datafile ${rawdata} | ${sed} -E -f ${sed_script} > tmp$$

for unk in 0 1 2 3 4 5; do 
    ${python} ${python_setunk} --datafile tmp$$ --unk ${unk} > tmp2_$$
    ${head} -300 tmp2_$$               > ${suffix}_unk${unk}.train.data
    ${head} -320 tmp2_$$ | ${tail} -20 > ${suffix}_unk${unk}.valid.data
    ${tail}  -26 tmp2_$$               > ${suffix}_unk${unk}.test.data
done
rm tmp$$ tmp2_$$
