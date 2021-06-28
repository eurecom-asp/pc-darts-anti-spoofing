#!/bin/bash

let PERCENTAGE=30

let train_bonafide_head=$(expr 2580 \*  $PERCENTAGE / 100)
let train_spoof_head=$(expr 3800 \*  $PERCENTAGE / 100)
let dev_bonafide_head=$(expr 2548 \*  $PERCENTAGE / 100)
let dev_spoof_head=$(expr 3716 \*  $PERCENTAGE / 100)
let eval_head=$(expr 71237 \*  $PERCENTAGE / 100)

head -n 2580 ASVspoof2019.LA.cm.train.trn.txt | shuf | head -n $train_bonafide_head > train_bonafide_short.txt
head -n 6380 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf | head -n $train_spoof_head > A01_short.txt
head -n 10180 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf | head -n $train_spoof_head > A02_short.txt
head -n 13980 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf | head -n $train_spoof_head > A03_short.txt
head -n 17780 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf | head -n $train_spoof_head > A04_short.txt
head -n 21580 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf | head -n $train_spoof_head > A05_short.txt
head -n 25380 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf | head -n $train_spoof_head > A06_short.txt

cat train_bonafide_short.txt A01_short.txt A02_short.txt A03_short.txt A04_short.txt A05_short.txt A06_short.txt > train_short.txt

head -n 2548 ASVspoof2019.LA.cm.dev.trl.txt | shuf | head -n $dev_bonafide_head > train_bonafide_dev_short.txt
head -n 6264 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 | shuf | head -n $dev_spoof_head > A01_dev_short.txt
head -n 9980 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 | shuf | head -n $dev_spoof_head > A02_dev_short.txt
head -n 13696 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 | shuf | head -n $dev_spoof_head > A03_dev_short.txt
head -n 17412 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 | shuf | head -n $dev_spoof_head > A04_dev_short.txt
head -n 21128 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 | shuf | head -n $dev_spoof_head > A05_dev_short.txt
head -n 24844 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 | shuf | head -n $dev_spoof_head > A06_dev_short.txt

cat train_bonafide_dev_short.txt A01_dev_short.txt A02_dev_short.txt A03_dev_short.txt A04_dev_short.txt A05_dev_short.txt A06_dev_short.txt > dev_short.txt

shuf ASVspoof2019.LA.cm.eval.trl.txt | head -n $eval_head > eval_short.txt

rm train_bonafide_short.txt A01_short.txt A02_short.txt A03_short.txt A04_short.txt A05_short.txt A06_short.txt
rm train_bonafide_dev_short.txt A01_dev_short.txt A02_dev_short.txt A03_dev_short.txt A04_dev_short.txt A05_dev_short.txt A06_dev_short.txt
