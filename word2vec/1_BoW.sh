#!/bin/sh

for i in 7000 8000 9000
do
  echo $i
  python 0_BoW.py $i
done
