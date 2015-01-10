#!/bin/sh

for i in 3000 4000 5000 6000
do
  echo $i
  python 0_read_data.py $i
done
