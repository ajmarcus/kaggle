#!/bin/sh

python create_features.py ../data/train_essay_no_header.txt > ../data/train_features.tsv
wc -l ../data/train*
head ../data/train_features.tsv
grep 'NA' ../data/train_features.tsv

python create_features.py ../data/test_essay_no_header.txt > ../data/test_features.tsv
wc -l ../data/test*
head ../data/test_features.tsv
grep 'NA' ../data/test_features.tsv

