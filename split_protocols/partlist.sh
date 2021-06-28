#!/bin/bash



head -n 2580 ASVspoof2019.LA.cm.train.trn.txt  > bonafide
head -n 6380 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 > a01
head -n 10180 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 > a02
head -n 13980 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 > a03
head -n 17780 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 > a04
head -n 21580 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 > a05
head -n 25380 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 > a06

head -n 2548 ASVspoof2019.LA.cm.dev.trl.txt > bonafide_dev
head -n 6264 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 > a01_dev
head -n 9980 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 > a02_dev
head -n 13696 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 > a03_dev
head -n 17412 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 > a04_dev
head -n 21128 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 > a05_dev
head -n 24844 ASVspoof2019.LA.cm.dev.trl.txt | tail -3716 > a06_dev
