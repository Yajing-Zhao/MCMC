#!/bin/bash

for FILE in *ed2.py; do
  echo ${FILE}
  sbatch -N1 -n1 -t 6:00:00 --mem=600 --wrap="time python ${FILE}" 
  sleep 1 # pause to be kind to the scheduler
done
