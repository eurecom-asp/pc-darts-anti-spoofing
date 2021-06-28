#!/bin/bash


head -n 2580 ASVspoof2019.LA.cm.train.trn.txt | shuf  > train_bonafide.txt
head -n 6380 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf  > A01.txt
head -n 10180 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf  > A02.txt
head -n 13980 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf  > A03.txt
head -n 17780 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf  > A04.txt
head -n 21580 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf  > A05.txt
head -n 25380 ASVspoof2019.LA.cm.train.trn.txt | tail -3800 | shuf  > A06.txt


head -n 1290 train_bonafide.txt > train_bonafide_h.txt
tail -n 1290 train_bonafide.txt > train_bonafide_t.txt

head -n 1900 A01.txt > A01_h.txt
tail -n 1900 A01.txt > A01_t.txt
head -n 1900 A02.txt > A02_h.txt
tail -n 1900 A02.txt > A02_t.txt
head -n 1900 A03.txt > A03_h.txt
tail -n 1900 A03.txt > A03_t.txt
head -n 1900 A04.txt > A04_h.txt
tail -n 1900 A04.txt > A04_t.txt
head -n 1900 A05.txt > A05_h.txt
tail -n 1900 A05.txt > A05_t.txt
head -n 1900 A06.txt > A06_h.txt
tail -n 1900 A06.txt > A06_t.txt

cat train_bonafide_h.txt A01_h.txt A02_h.txt A03_h.txt A04_h.txt A05_h.txt A06_h.txt > train_h.txt
cat train_bonafide_t.txt A01_t.txt A02_t.txt A03_t.txt A04_t.txt A05_t.txt A06_t.txt > train_t.txt

rm train_bonafide.txt A01.txt A02.txt A03.txt A04.txt A05.txt A06.txt A01_h.txt A02_h.txt A03_h.txt A04_h.txt A05_h.txt A06_h.txt train_bonafide_h.txt train_bonafide_t.txt A01_t.txt A02_t.txt A03_t.txt A04_t.txt A05_t.txt A06_t.txt
